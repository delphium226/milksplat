//! Windows system-audio capture: WASAPI loopback through cpal, restarted on device changes.
//!
//! One service thread ("ms-audio") owns the cpal `Stream` (not `Send`), the
//! `IMMNotificationClient` registration and the restart logic. The ring producer is
//! moved into each stream's callback and comes back over a channel when that
//! stream is dropped (see `capture.rs`).

mod bluetooth;
mod capture;
mod notify;

use crate::restart::RestartPolicy;
use crate::{ring, RingConsumer, RingProducer};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use windows::Win32::Foundation::RPC_E_CHANGED_MODE;
use windows::Win32::Media::Audio::IMMDeviceEnumerator;
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED};

/// Ring capacity: 1 s at 192 kHz.
const RING_FRAMES: usize = 196_608;
/// Service-thread wake-up period while a stream runs (health checks).
const POLL: Duration = Duration::from_millis(250);
/// How long `Drop` waits for the service thread.
const STOP_TIMEOUT: Duration = Duration::from_secs(2);
/// How long to wait for the producer to come back from a dropped stream.
const RECLAIM_TIMEOUT: Duration = Duration::from_secs(1);

/// Which output device to capture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceChoice {
    /// The system default render device (console role); follows changes.
    Default,
    /// The first output device whose cpal name equals this string.
    Named(String),
}

/// Capture-state notifications.
#[derive(Clone, Debug, PartialEq)]
pub enum AudioEvent {
    /// A stream started (sent on every successful start, including restarts).
    Started {
        name: String,
        sample_rate: u32,
        channels: u16,
        bluetooth: bool,
    },
    /// Capture stopped; the service retries with backoff.
    Stopped { reason: String },
    /// The Windows default output device changed.
    DefaultDeviceChanged,
}

/// Clock giving capture timestamps in seconds (called from the audio callback).
pub(crate) type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

/// Messages to the service thread.
pub(crate) enum Cmd {
    Stop,
    /// New default render/console device id (`None`: no default device).
    DefaultChanged(Option<String>),
    /// An endpoint became active / inactive (disabled, unplugged, removed).
    DeviceState {
        id: String,
        active: bool,
    },
    /// Fatal error reported by the stream with this generation.
    StreamError {
        gen: u64,
        reason: String,
    },
}

/// Loopback capture service. Dropping it stops capture (waits at most 2 s).
pub struct AudioService {
    cmd_tx: Sender<Cmd>,
    done_rx: Receiver<()>,
    thread: Option<JoinHandle<()>>,
}

impl AudioService {
    /// Starts capturing `choice` on a new "ms-audio" thread. Frames arrive in the
    /// returned ring consumer, stamped with `clock()`; state changes arrive as events.
    pub fn start(
        choice: DeviceChoice,
        clock: Arc<dyn Fn() -> f64 + Send + Sync>,
    ) -> (Self, RingConsumer, Receiver<AudioEvent>) {
        let (producer, consumer) = ring(RING_FRAMES);
        let (events_tx, events_rx) = crossbeam_channel::unbounded();
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
        let (done_tx, done_rx) = crossbeam_channel::bounded::<()>(1);

        let thread_cmd_tx = cmd_tx.clone();
        let thread_events = events_tx.clone();
        let spawned = std::thread::Builder::new()
            .name("ms-audio".into())
            .spawn(move || {
                // Dropped on exit (also on panic), which wakes `Drop`.
                let _done = done_tx;
                service_main(
                    choice,
                    clock,
                    producer,
                    thread_cmd_tx,
                    cmd_rx,
                    thread_events,
                );
            });
        let thread = match spawned {
            Ok(h) => Some(h),
            Err(e) => {
                warn!("ms-audio: cannot spawn service thread: {e}");
                let _ = events_tx.send(AudioEvent::Stopped {
                    reason: format!("cannot spawn audio thread: {e}"),
                });
                None
            }
        };
        (
            Self {
                cmd_tx,
                done_rx,
                thread,
            },
            consumer,
            events_rx,
        )
    }

    /// Names of the output devices (as matched by [`DeviceChoice::Named`]).
    /// Errors yield an empty list.
    pub fn list_output_devices() -> Vec<String> {
        capture::list_output_devices()
    }
}

impl Drop for AudioService {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(Cmd::Stop);
        let Some(handle) = self.thread.take() else {
            return;
        };
        match self.done_rx.recv_timeout(STOP_TIMEOUT) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                let _ = handle.join();
            }
            Err(RecvTimeoutError::Timeout) => {
                warn!("ms-audio: service thread did not stop within {STOP_TIMEOUT:?}; detaching");
            }
        }
    }
}

/// COM apartment for the service thread; uninitialises on drop if we initialised it.
struct Com {
    initialised: bool,
}

impl Com {
    fn init_mta() -> Self {
        // SAFETY: plain COM initialisation of the current thread.
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if hr.is_err() && hr != RPC_E_CHANGED_MODE {
            warn!("ms-audio: CoInitializeEx(MTA) failed: {hr:?}");
        }
        Self {
            initialised: hr.is_ok(),
        }
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        if self.initialised {
            // SAFETY: balanced with the successful CoInitializeEx in `init_mta`.
            unsafe { CoUninitialize() };
        }
    }
}

fn service_main(
    choice: DeviceChoice,
    clock: Clock,
    producer: RingProducer,
    cmd_tx: Sender<Cmd>,
    cmd_rx: Receiver<Cmd>,
    events: Sender<AudioEvent>,
) {
    // Must come first so cpal's own (STA) initialisation on this thread is a no-op,
    // and must drop last (after every COM object below).
    let _com = Com::init_mta();
    let enumerator = match notify::create_enumerator() {
        Ok(e) => Some(e),
        Err(e) => {
            warn!("ms-audio: no IMMDeviceEnumerator ({e}); device notifications disabled");
            None
        }
    };
    let notifier =
        enumerator
            .as_ref()
            .and_then(|e| match notify::Notifier::register(e, cmd_tx.clone()) {
                Ok(n) => Some(n),
                Err(err) => {
                    warn!("ms-audio: cannot register device notifications: {err}");
                    None
                }
            });
    let (home_tx, home_rx) = crossbeam_channel::bounded(1);
    let mut service = Service {
        choice,
        clock,
        events,
        cmd_tx,
        cmd_rx,
        home_tx,
        home_rx,
        producer: Some(producer),
        host: cpal::default_host(),
        enumerator: enumerator.clone(),
        running: None,
        retry_at: Some(Instant::now()),
        policy: RestartPolicy::new(),
        gen: 0,
        last_default: None,
    };
    service.run();
    drop(service);
    drop(notifier);
    drop(enumerator);
}

struct Service {
    choice: DeviceChoice,
    clock: Clock,
    events: Sender<AudioEvent>,
    cmd_tx: Sender<Cmd>,
    cmd_rx: Receiver<Cmd>,
    /// Where a dropped stream's callback returns the producer.
    home_tx: Sender<RingProducer>,
    home_rx: Receiver<RingProducer>,
    /// Present while no stream owns it.
    producer: Option<RingProducer>,
    host: cpal::Host,
    enumerator: Option<IMMDeviceEnumerator>,
    running: Option<capture::Running>,
    /// When not running: when to attempt the next start.
    retry_at: Option<Instant>,
    policy: RestartPolicy,
    /// Generation of the newest stream (stale error reports are ignored).
    gen: u64,
    /// Last default-device id seen in a notification (dedups repeated callbacks).
    last_default: Option<Option<String>>,
}

impl Service {
    fn run(&mut self) {
        loop {
            let now = Instant::now();
            if let Some(r) = &self.running {
                if r.delivered.load(Ordering::Relaxed) {
                    self.policy.audio_delivered();
                }
                self.policy.tick(now);
            } else if let Some(at) = self.retry_at {
                if now >= at {
                    if !self.try_start() {
                        return;
                    }
                    continue;
                }
            }
            let timeout = match (&self.running, self.retry_at) {
                (None, Some(at)) => at.saturating_duration_since(now),
                _ => POLL,
            };
            match self.cmd_rx.recv_timeout(timeout) {
                Ok(Cmd::Stop) | Err(RecvTimeoutError::Disconnected) => break,
                Ok(cmd) => self.handle(cmd),
                Err(RecvTimeoutError::Timeout) => {}
            }
        }
        self.teardown();
    }

    fn emit(&self, event: AudioEvent) {
        let _ = self.events.send(event);
    }

    /// Attempts a start. Returns false if the service cannot continue.
    fn try_start(&mut self) -> bool {
        let now = Instant::now();
        self.retry_at = None;
        let Some(producer) = self.producer.take() else {
            // Unreachable unless a stream failed to hand the producer back.
            self.emit(AudioEvent::Stopped {
                reason: "internal error: ring producer lost".into(),
            });
            return false;
        };
        self.gen += 1;
        let opened = capture::open(capture::OpenArgs {
            host: &self.host,
            choice: &self.choice,
            producer,
            home: self.home_tx.clone(),
            clock: self.clock.clone(),
            cmd_tx: self.cmd_tx.clone(),
            gen: self.gen,
            enumerator: self.enumerator.as_ref(),
        });
        match opened {
            Ok(running) => {
                info!(
                    "ms-audio: capturing \"{}\" ({} Hz, {} ch, bluetooth: {})",
                    running.name, running.sample_rate, running.channels, running.bluetooth
                );
                self.emit(AudioEvent::Started {
                    name: running.name.clone(),
                    sample_rate: running.sample_rate,
                    channels: running.channels,
                    bluetooth: running.bluetooth,
                });
                self.policy.started(now);
                self.running = Some(running);
            }
            Err(reason) => {
                self.reclaim_producer();
                self.fail(reason);
            }
        }
        true
    }

    /// Reports a failure and schedules the next attempt with backoff.
    fn fail(&mut self, reason: String) {
        let now = Instant::now();
        let delay = self.policy.failed(now);
        warn!("ms-audio: capture stopped: {reason}; retrying in {delay:?}");
        self.emit(AudioEvent::Stopped { reason });
        self.retry_at = Some(now + delay);
    }

    /// Drops the running stream (if any) and takes the producer back.
    fn teardown(&mut self) {
        if let Some(running) = self.running.take() {
            // Joins cpal's capture thread, which drops our callback and sends the producer home.
            drop(running);
            self.reclaim_producer();
        }
    }

    fn reclaim_producer(&mut self) {
        if self.producer.is_some() {
            return;
        }
        match self.home_rx.recv_timeout(RECLAIM_TIMEOUT) {
            Ok(p) => self.producer = Some(p),
            Err(e) => warn!("ms-audio: ring producer not returned by stream: {e}"),
        }
    }

    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Stop => {}
            Cmd::DefaultChanged(id) => self.on_default_changed(id),
            Cmd::DeviceState { id, active } => self.on_device_state(id, active),
            Cmd::StreamError { gen, reason } => {
                if self.running.as_ref().is_some_and(|r| r.gen == gen) {
                    self.teardown();
                    self.fail(format!("stream error: {reason}"));
                } else {
                    debug!("ms-audio: ignoring error from stale stream {gen}: {reason}");
                }
            }
        }
    }

    fn on_default_changed(&mut self, id: Option<String>) {
        if self.last_default.as_ref() == Some(&id) {
            return;
        }
        self.last_default = Some(id.clone());
        self.emit(AudioEvent::DefaultDeviceChanged);
        if self.choice != DeviceChoice::Default {
            return;
        }
        match (&self.running, &id) {
            (Some(r), Some(new)) if r.device_id.eq_ignore_ascii_case(new) => {}
            (Some(_), _) => {
                info!("ms-audio: default output device changed; restarting capture");
                self.teardown();
                self.policy.stopped(Instant::now());
                self.retry_at = Some(Instant::now());
            }
            // Waiting to retry: try the new default now.
            (None, Some(_)) => self.retry_at = Some(Instant::now()),
            (None, None) => {}
        }
    }

    fn on_device_state(&mut self, id: String, active: bool) {
        match &self.running {
            Some(r) if !active && r.device_id.eq_ignore_ascii_case(&id) => {
                self.teardown();
                self.fail("device removed or disabled".into());
            }
            // A device appeared while we wait to retry: try now rather than after the backoff.
            None if active && self.retry_at.is_some() => self.retry_at = Some(Instant::now()),
            _ => {}
        }
    }
}
