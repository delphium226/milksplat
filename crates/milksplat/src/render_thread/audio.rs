//! System-audio input for the render thread: the capture service (Windows
//! only), its ring and its events, drained into the analysis history.

use std::sync::Arc;

use ms_analysis::AudioHistory;

/// The capture service and the consuming end of its ring. Without capture
/// (other platforms) the history only ever gets silence.
pub(super) struct AudioInput {
    #[cfg(windows)]
    capture: Capture,
}

#[cfg(windows)]
struct Capture {
    // Field order: the ring and events go before the service stops.
    ring: ms_audio::RingConsumer,
    events: crossbeam_channel::Receiver<ms_audio::AudioEvent>,
    _service: ms_audio::AudioService,
}

impl AudioInput {
    /// Starts capturing `device` (`None`: the default output), stamping
    /// packets with `clock`.
    #[cfg(windows)]
    pub(super) fn start(device: Option<String>, clock: Arc<dyn Fn() -> f64 + Send + Sync>) -> Self {
        let choice = match device {
            Some(name) => ms_audio::DeviceChoice::Named(name),
            None => ms_audio::DeviceChoice::Default,
        };
        let (service, ring, events) = ms_audio::AudioService::start(choice, clock);
        AudioInput {
            capture: Capture {
                ring,
                events,
                _service: service,
            },
        }
    }

    #[cfg(not(windows))]
    pub(super) fn start(
        _device: Option<String>,
        _clock: Arc<dyn Fn() -> f64 + Send + Sync>,
    ) -> Self {
        tracing::warn!("system audio capture is only available on Windows");
        AudioInput {}
    }

    /// Handles capture events (a new sample rate resets `hist`; the
    /// analyzer follows the history's rate by itself), then moves every
    /// queued packet into `hist`.
    pub(super) fn drain(&mut self, hist: &mut AudioHistory) {
        #[cfg(windows)]
        {
            let c = &mut self.capture;
            while let Ok(ev) = c.events.try_recv() {
                match ev {
                    ms_audio::AudioEvent::Started {
                        name,
                        sample_rate,
                        channels,
                        bluetooth,
                    } => {
                        tracing::info!(
                            "audio capture started: {name}, {sample_rate} Hz, {channels} ch{}",
                            if bluetooth { ", Bluetooth" } else { "" }
                        );
                        if sample_rate != hist.sample_rate() {
                            hist.reset(sample_rate);
                        }
                    }
                    ms_audio::AudioEvent::Stopped { reason } => {
                        tracing::info!("audio capture stopped: {reason}");
                    }
                    ms_audio::AudioEvent::DefaultDeviceChanged => {
                        tracing::info!("the default output device changed");
                    }
                }
            }
            c.ring.drain_into(hist);
        }
        #[cfg(not(windows))]
        let _ = hist;
    }
}
