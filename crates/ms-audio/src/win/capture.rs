//! Opening one cpal WASAPI loopback stream, and the real-time callback state.
//!
//! Producer hand-off: the callback owns a [`ProducerSlot`] by value (moved in when the
//! stream is built), so the audio thread never locks to reach the ring. When cpal drops
//! the callback (stream dropped, or a failed build), the slot's `Drop` sends the
//! producer back to the service thread over a channel for the next stream.

use super::{bluetooth, Clock, Cmd, DeviceChoice};
use crate::{downmix, layout_for, Channel, RingProducer};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ErrorKind, SampleFormat};
use crossbeam_channel::Sender;
use ms_analysis::Frame;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::warn;
use windows::Win32::Media::Audio::IMMDeviceEnumerator;

/// A running capture stream and what it captures.
pub(super) struct Running {
    // Field order: the stream drops first.
    _stream: cpal::Stream,
    pub(super) gen: u64,
    /// WASAPI endpoint id.
    pub(super) device_id: String,
    pub(super) name: String,
    pub(super) sample_rate: u32,
    pub(super) channels: u16,
    pub(super) bluetooth: bool,
    /// Set by the callback once it has delivered any audio.
    pub(super) delivered: Arc<AtomicBool>,
}

pub(super) struct OpenArgs<'a> {
    pub(super) host: &'a cpal::Host,
    pub(super) choice: &'a DeviceChoice,
    pub(super) producer: RingProducer,
    pub(super) home: Sender<RingProducer>,
    pub(super) clock: Clock,
    pub(super) cmd_tx: Sender<Cmd>,
    pub(super) gen: u64,
    pub(super) enumerator: Option<&'a IMMDeviceEnumerator>,
}

/// Opens and starts a loopback stream. On error the producer has been (or is being)
/// sent back to `args.home`.
pub(super) fn open(args: OpenArgs<'_>) -> Result<Running, String> {
    // cpal 0.18's WASAPI backend `expect`s on some OS failures (property store, event
    // creation); turn such a panic into an ordinary failure. The producer slot is dropped
    // during unwinding, so the producer still goes home.
    catch_unwind(AssertUnwindSafe(|| open_inner(args)))
        .unwrap_or_else(|_| Err("audio backend panicked while opening the stream".into()))
}

fn open_inner(args: OpenArgs<'_>) -> Result<Running, String> {
    let slot = ProducerSlot {
        producer: Some(args.producer),
        home: args.home,
    };
    let device = resolve(args.host, args.choice)?;
    let device_id = device
        .id()
        .map(|id| id.id().to_string())
        .map_err(|e| format!("cannot read device id: {e}"))?;
    let name = device_name(&device).unwrap_or_else(|| "unknown device".into());
    let supported = device
        .default_output_config()
        .map_err(|e| format!("\"{name}\": no output format: {e}"))?;
    let sample_rate = supported.sample_rate();
    let channels = supported.channels();
    let format = supported.sample_format();
    if channels == 0 {
        return Err(format!("\"{name}\": device reports 0 channels"));
    }
    let delivered = Arc::new(AtomicBool::new(false));
    let mut capture = Capture::new(slot, channels, sample_rate, format, args.clock, &delivered);

    let gen = args.gen;
    let cmd_tx = args.cmd_tx;
    let on_error = move |err: cpal::Error| match err.kind() {
        // Glitches and scheduling notices: the stream keeps running.
        ErrorKind::Xrun | ErrorKind::RealtimeDenied | ErrorKind::DeviceChanged => {}
        _ => {
            let _ = cmd_tx.send(Cmd::StreamError {
                gen,
                reason: err.to_string(),
            });
        }
    };
    let config = supported.config();
    // Building an *input* stream on an *output* device makes cpal open WASAPI loopback.
    let stream = match format {
        SampleFormat::F32 => device.build_input_stream::<f32, _, _>(
            config,
            move |data, _| capture.process_f32(data),
            on_error,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream::<i16, _, _>(
            config,
            move |data, _| capture.process_i16(data),
            on_error,
            None,
        ),
        other => return Err(format!("\"{name}\": unsupported sample format {other}")),
    }
    .map_err(|e| format!("\"{name}\": cannot open loopback stream: {e}"))?;
    stream
        .play()
        .map_err(|e| format!("\"{name}\": cannot start loopback stream: {e}"))?;

    let bluetooth = bluetooth::is_bluetooth(args.enumerator, &device_id);
    Ok(Running {
        _stream: stream,
        gen,
        device_id,
        name,
        sample_rate,
        channels,
        bluetooth,
        delivered,
    })
}

/// Finds the device for `choice`. `Default` resolves to the *current* default endpoint
/// as a specific device, so a default change is handled by our own restart (which
/// reports the new device and rate) rather than by cpal's silent rerouting.
fn resolve(host: &cpal::Host, choice: &DeviceChoice) -> Result<cpal::Device, String> {
    match choice {
        DeviceChoice::Default => {
            let default = host
                .default_output_device()
                .ok_or("no default output device")?;
            let id = default
                .id()
                .map_err(|e| format!("cannot read default device id: {e}"))?;
            host.device_by_id(&id)
                .ok_or_else(|| "default output device not found".into())
        }
        DeviceChoice::Named(want) => host
            .output_devices()
            .map_err(|e| format!("cannot list output devices: {e}"))?
            .find(|d| device_name(d).as_deref() == Some(want.as_str()))
            .ok_or_else(|| format!("output device \"{want}\" not found")),
    }
}

/// cpal's display name for `device`, or `None` on any failure (including a panic inside
/// cpal's `description()`, which `expect`s on the property store).
fn device_name(device: &cpal::Device) -> Option<String> {
    catch_unwind(AssertUnwindSafe(|| device.description()))
        .ok()?
        .ok()
        .map(|d| d.name().to_string())
}

pub(super) fn list_output_devices() -> Vec<String> {
    let listed = catch_unwind(|| {
        cpal::default_host()
            .output_devices()
            .map(|devices| devices.filter_map(|d| device_name(&d)).collect())
    });
    match listed {
        Ok(Ok(names)) => names,
        Ok(Err(e)) => {
            warn!("ms-audio: cannot list output devices: {e}");
            Vec::new()
        }
        Err(_) => {
            warn!("ms-audio: audio backend panicked while listing devices");
            Vec::new()
        }
    }
}

/// Owns the ring producer inside the audio callback; sends it home when dropped.
struct ProducerSlot {
    producer: Option<RingProducer>,
    home: Sender<RingProducer>,
}

impl Drop for ProducerSlot {
    fn drop(&mut self) {
        if let Some(p) = self.producer.take() {
            // Bounded(1) channel that is empty here (only one producer exists).
            let _ = self.home.try_send(p);
        }
    }
}

/// Real-time state owned by the data callback.
struct Capture {
    slot: ProducerSlot,
    layout: Vec<Channel>,
    channels: usize,
    /// Downmix output.
    frames: Vec<Frame>,
    /// i16 → f32 conversion buffer (empty for f32 streams).
    samples: Vec<f32>,
    clock: Clock,
    delivered: Arc<AtomicBool>,
}

impl Capture {
    fn new(
        slot: ProducerSlot,
        channels: u16,
        sample_rate: u32,
        format: SampleFormat,
        clock: Clock,
        delivered: &Arc<AtomicBool>,
    ) -> Self {
        // 0.5 s: far more than any shared-mode WASAPI packet (~10 ms), so the scratch
        // buffers should never need to grow.
        let frames = (sample_rate as usize / 2).max(4096);
        let samples = if format == SampleFormat::I16 {
            frames * channels as usize
        } else {
            0
        };
        Self {
            slot,
            layout: layout_for(channels),
            channels: channels as usize,
            frames: vec![[0.0; 2]; frames],
            samples: vec![0.0; samples],
            clock,
            delivered: delivered.clone(),
        }
    }

    /// Audio callback for f32 streams. Never locks or blocks; allocates only if a packet
    /// is larger than any before it and larger than the 0.5 s preallocation (the scratch
    /// buffer then grows once).
    fn process_f32(&mut self, data: &[f32]) {
        let n = data.len() / self.channels;
        if n == 0 {
            return;
        }
        if n > self.frames.len() {
            // The only allocation on the audio thread (documented above).
            self.frames.resize(n, [0.0; 2]);
        }
        let out = &mut self.frames[..n];
        downmix(data, &self.layout, out);
        if let Some(p) = self.slot.producer.as_mut() {
            p.push(out, (self.clock)());
        }
        self.delivered.store(true, Ordering::Relaxed);
    }

    /// Audio callback for i16 streams: converts into the preallocated buffer, then as f32.
    fn process_i16(&mut self, data: &[i16]) {
        if data.len() > self.samples.len() {
            // Same growth rule as `frames` in `process_f32`.
            self.samples.resize(data.len(), 0.0);
        }
        for (d, &s) in self.samples.iter_mut().zip(data) {
            *d = f32::from(s) / 32768.0;
        }
        // `take` leaves an empty Vec (no allocation) and lets `process_f32` borrow `self`.
        let samples = std::mem::take(&mut self.samples);
        self.process_f32(&samples[..data.len()]);
        self.samples = samples;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring;
    use ms_analysis::AudioHistory;

    fn capture(
        format: SampleFormat,
        channels: u16,
    ) -> (
        Capture,
        crate::RingConsumer,
        crossbeam_channel::Receiver<RingProducer>,
    ) {
        let (producer, consumer) = ring(1024);
        let (home, home_rx) = crossbeam_channel::bounded(1);
        let slot = ProducerSlot {
            producer: Some(producer),
            home,
        };
        let clock: Clock = Arc::new(|| 1.0);
        let delivered = Arc::new(AtomicBool::new(false));
        let c = Capture::new(slot, channels, 1000, format, clock, &delivered);
        (c, consumer, home_rx)
    }

    fn drained(consumer: &mut crate::RingConsumer, n: usize, end: f64) -> Vec<Frame> {
        let mut h = AudioHistory::new(1000, 1.0);
        consumer.drain_into(&mut h);
        let mut out = vec![[9.0; 2]; n];
        h.window(end, &mut out);
        out
    }

    #[test]
    fn f32_callback_downmixes_and_pushes() {
        let (mut c, mut consumer, _home) = capture(SampleFormat::F32, 2);
        c.process_f32(&[0.1, 0.2, 0.3, 0.4]);
        assert!(c.delivered.load(Ordering::Relaxed));
        assert_eq!(drained(&mut consumer, 2, 1.0), vec![[0.1, 0.2], [0.3, 0.4]]);
    }

    #[test]
    fn i16_callback_converts() {
        let (mut c, mut consumer, _home) = capture(SampleFormat::I16, 1);
        c.process_i16(&[16384, -32768]);
        assert_eq!(
            drained(&mut consumer, 2, 1.0),
            vec![[0.5, 0.5], [-1.0, -1.0]]
        );
    }

    #[test]
    fn oversized_packet_grows_scratch() {
        let (mut c, mut consumer, _home) = capture(SampleFormat::F32, 2);
        let n = c.frames.len() + 10;
        let data: Vec<f32> = (0..n).flat_map(|i| [i as f32, 0.0]).collect();
        c.process_f32(&data);
        assert_eq!(c.frames.len(), n);
        // Ring holds 1024 frames: the packet's head arrives, its last written frame
        // (index 1023) stamped `n - 1024` frames before the clock time 1.0.
        let out = drained(&mut consumer, 1, 1.0 - (n - 1024) as f64 / 1000.0);
        assert_eq!(out[0][0], 1023.0);
    }

    #[test]
    fn dropping_callback_returns_producer() {
        let (c, _consumer, home) = capture(SampleFormat::F32, 2);
        assert!(home.try_recv().is_err());
        drop(c);
        assert!(home.try_recv().is_ok());
    }
}
