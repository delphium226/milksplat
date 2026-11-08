//! Restart bookkeeping for the capture service. Pure: no OS calls, no clock reads.

use crate::backoff_delay;
use std::time::{Duration, Instant};

/// A stream that runs this long counts as healthy even if it never delivered audio
/// (WASAPI loopback delivers no packets while nothing is playing).
pub(crate) const HEALTHY_AFTER: Duration = Duration::from_secs(5);

/// Decides the backoff before the next start attempt.
///
/// `attempt` counts consecutive failures; it resets once a started stream proves
/// healthy (it delivered audio, or it ran for [`HEALTHY_AFTER`]).
#[derive(Debug, Default)]
pub(crate) struct RestartPolicy {
    attempt: u32,
    started_at: Option<Instant>,
}

impl RestartPolicy {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// A stream started successfully at `now`.
    pub(crate) fn started(&mut self, now: Instant) {
        self.started_at = Some(now);
    }

    /// The running stream delivered audio.
    pub(crate) fn audio_delivered(&mut self) {
        if self.started_at.is_some() {
            self.attempt = 0;
        }
    }

    /// Periodic check while a stream runs.
    pub(crate) fn tick(&mut self, now: Instant) {
        if let Some(t) = self.started_at {
            if now.saturating_duration_since(t) >= HEALTHY_AFTER {
                self.attempt = 0;
            }
        }
    }

    /// The stream was stopped on purpose (e.g. to follow a new default device).
    /// Not a failure: the attempt counter only changes if the stream had become healthy.
    pub(crate) fn stopped(&mut self, now: Instant) {
        self.tick(now);
        self.started_at = None;
    }

    /// A start attempt or a running stream failed at `now`. Returns the delay before the
    /// next attempt.
    pub(crate) fn failed(&mut self, now: Instant) -> Duration {
        self.stopped(now);
        let delay = backoff_delay(self.attempt);
        self.attempt = self.attempt.saturating_add(1);
        delay
    }
}

/// Where the ring producer is. While a stream runs, its callback owns the
/// producer and sends it home when the callback is dropped. It can come home
/// late (a driver slow to stop the stream) or unasked (cpal's capture thread
/// died), so the service accepts it whenever it shows up.
#[derive(Debug)]
pub(crate) struct ProducerSlot<P> {
    held: Option<P>,
}

impl<P> ProducerSlot<P> {
    pub(crate) fn new(producer: P) -> Self {
        Self {
            held: Some(producer),
        }
    }

    /// Whether the service holds the producer (no stream owns it).
    pub(crate) fn is_held(&self) -> bool {
        self.held.is_some()
    }

    /// Accepts a producer that came home (`returned`, from a non-blocking
    /// receive). Returns true if one did.
    pub(crate) fn accept(&mut self, returned: Option<P>) -> bool {
        match returned {
            Some(p) => {
                self.held = Some(p);
                true
            }
            None => false,
        }
    }

    /// Takes the producer for a new stream, first accepting one that came
    /// home late (`returned`). `None`: it is still out; the caller retries
    /// later (with backoff) rather than giving up.
    pub(crate) fn take_for_start(&mut self, returned: Option<P>) -> Option<P> {
        self.accept(returned);
        self.held.take()
    }
}

/// Health check while a stream runs: the producer coming home means the
/// stream's callback was dropped, so the stream is dead and must be
/// restarted (a failure, with backoff).
pub(crate) fn stream_died(running: bool, producer_came_home: bool) -> bool {
    running && producer_came_home
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_producer_is_used_for_the_next_start() {
        let mut slot = ProducerSlot::new(1);
        assert_eq!(slot.take_for_start(None), Some(1));
        // The teardown timed out: nothing came home yet, so no start now.
        assert_eq!(slot.take_for_start(None), None);
        assert!(!slot.is_held());
        // It arrives later and the retry picks it up.
        assert_eq!(slot.take_for_start(Some(1)), Some(1));
    }

    #[test]
    fn producer_home_while_running_is_a_stream_failure() {
        let mut slot = ProducerSlot::new(7);
        let p = slot.take_for_start(None);
        assert_eq!(p, Some(7));
        // Health ticks with nothing coming home: healthy.
        let came = slot.accept(None);
        assert!(!stream_died(true, came));
        // The callback was dropped: the producer is back while running.
        let came = slot.accept(p);
        assert!(stream_died(true, came));
        assert!(slot.is_held());
        // While not running, a returned producer is just collected.
        assert!(!stream_died(false, true));
    }

    #[test]
    fn missing_producer_retries_with_backoff() {
        // A start without the producer is a failure like any other: the
        // service schedules a retry instead of ending.
        let t = Instant::now();
        let mut p = RestartPolicy::new();
        let mut slot: ProducerSlot<u8> = ProducerSlot { held: None };
        assert_eq!(slot.take_for_start(None), None);
        assert_eq!(p.failed(t), ms(500));
        assert_eq!(slot.take_for_start(Some(3)), Some(3));
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn consecutive_failures_escalate() {
        let t = Instant::now();
        let mut p = RestartPolicy::new();
        let got: Vec<Duration> = (0..5).map(|_| p.failed(t)).collect();
        assert_eq!(got, vec![ms(500), ms(1000), ms(2000), ms(5000), ms(5000)]);
    }

    #[test]
    fn audio_resets_backoff() {
        let t = Instant::now();
        let mut p = RestartPolicy::new();
        p.failed(t);
        p.failed(t);
        p.started(t);
        p.audio_delivered();
        assert_eq!(p.failed(t + ms(100)), ms(500));
    }

    #[test]
    fn running_five_seconds_resets_backoff() {
        let t = Instant::now();
        let mut p = RestartPolicy::new();
        p.failed(t);
        p.failed(t);
        p.failed(t);
        p.started(t);
        // Failing at exactly HEALTHY_AFTER counts as healthy.
        assert_eq!(p.failed(t + HEALTHY_AFTER), ms(500));
    }

    #[test]
    fn short_silent_run_keeps_escalating() {
        let t = Instant::now();
        let mut p = RestartPolicy::new();
        p.failed(t);
        p.started(t);
        p.tick(t + ms(4000));
        assert_eq!(p.failed(t + ms(4999)), ms(1000));
    }

    #[test]
    fn audio_while_not_running_is_ignored() {
        let t = Instant::now();
        let mut p = RestartPolicy::new();
        p.failed(t);
        p.audio_delivered();
        assert_eq!(p.failed(t), ms(1000));
    }

    #[test]
    fn deliberate_stop_is_not_a_failure() {
        let t = Instant::now();
        let mut p = RestartPolicy::new();
        p.failed(t);
        p.started(t);
        p.stopped(t + ms(100));
        p.started(t + ms(100));
        // Previous failure still counts (the stopped stream never became healthy),
        // but the deliberate stop itself did not add one.
        assert_eq!(p.failed(t + ms(200)), ms(1000));
    }
}
