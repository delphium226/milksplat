//! Fixed-rate simulation clock driven by variable display-frame deltas.

const MIN_HZ: f64 = 1.0;
const MAX_HZ: f64 = 240.0;
const DEFAULT_HZ: f64 = 60.0;
const MAX_RUN: u32 = 4;

/// Result of one [`FixedClock::advance`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticks {
    /// Simulation ticks to run this frame (at most 4).
    pub run: u32,
    /// Due ticks that were discarded rather than run.
    pub dropped: u32,
}

/// Accumulator-based fixed-rate clock.
#[derive(Debug, Clone)]
pub struct FixedClock {
    rate: f64,
    /// Seconds into the current tick, in `[0, 1/rate)`.
    acc: f64,
}

fn clamp_rate(hz: f64) -> f64 {
    if hz.is_finite() {
        hz.clamp(MIN_HZ, MAX_HZ)
    } else {
        DEFAULT_HZ
    }
}

impl FixedClock {
    pub fn new(rate_hz: f64) -> Self {
        Self {
            rate: clamp_rate(rate_hz),
            acc: 0.0,
        }
    }

    /// Changes the tick rate, keeping the fractional phase (`alpha`).
    pub fn set_rate(&mut self, hz: f64) {
        let alpha = self.alpha();
        self.rate = clamp_rate(hz);
        self.acc = alpha / self.rate;
    }

    /// Adds `dt` seconds and returns how many ticks to run. Non-finite or
    /// negative `dt` counts as zero.
    pub fn advance(&mut self, dt: f64) -> Ticks {
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };
        let period = 1.0 / self.rate;
        let total = self.acc + dt;
        // fmod is exact, so the remainder always lands in [0, period).
        let rem = total % period;
        let due = ((total - rem) * self.rate).round();
        self.acc = if rem.is_finite() && rem >= 0.0 { rem } else { 0.0 };
        let run = due.min(f64::from(MAX_RUN));
        // `as` saturates for out-of-range floats.
        Ticks {
            run: run as u32,
            dropped: (due - run) as u32,
        }
    }

    /// Fraction of the way into the next tick, in `[0, 1)`.
    pub fn alpha(&self) -> f64 {
        let a = self.acc * self.rate;
        if a < 1.0 {
            a.max(0.0)
        } else {
            0.0
        }
    }
}
