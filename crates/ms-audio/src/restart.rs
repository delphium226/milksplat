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

#[cfg(test)]
mod tests {
    use super::*;

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
