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
    /// Creates a clock at `rate_hz` (clamped to 1..=240; non-finite -> 60).
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

    /// The tick rate in Hz.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// [`set_rate`](Self::set_rate) to `rate_hz` if it differs, then
    /// [`advance`](Self::advance) by `dt`: the per-frame step for a rate
    /// that follows the preset on screen.
    pub fn advance_at(&mut self, rate_hz: f64, dt: f64) -> Ticks {
        if clamp_rate(rate_hz) != self.rate {
            self.set_rate(rate_hz);
        }
        self.advance(dt)
    }

    /// Adds `dt` seconds and returns how many ticks to run (at most 4; the
    /// rest are dropped, not carried). Non-finite or negative `dt` counts as
    /// zero.
    pub fn advance(&mut self, dt: f64) -> Ticks {
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };
        let period = 1.0 / self.rate;
        let total = self.acc + dt;
        // Epsilon absorbs representation error in dt (e.g. 3/47 * 47 = 2.999...).
        let mut due = (total * self.rate + 1e-9).floor();
        let mut acc = total - due / self.rate;
        // Rounding can leave the remainder just outside [0, period).
        if acc < 0.0 {
            acc = 0.0;
        } else if acc >= period {
            due += 1.0;
            acc -= period;
        }
        self.acc = if acc >= 0.0 && acc < period { acc } else { 0.0 };
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
