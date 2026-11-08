//! Quality governor: steps the internal render scale down when frames run
//! over budget and back up when there is headroom.

use std::collections::VecDeque;

const STEPS: [f32; 5] = [1.0, 0.875, 0.75, 0.625, 0.5];
const WINDOW_S: f64 = 2.0;
const MIN_WINDOW_S: f64 = 1.5;
const MIN_STEP_GAP_S: f64 = 3.0;
const DOWN_FRAC: f64 = 0.9;
const UP_FRAC: f64 = 0.6;
const CAPACITY: usize = 512;

/// Steps the render scale through `[1.0, 0.875, 0.75, 0.625, 0.5]` from a
/// 2 s moving average of frame times. `now` must be monotonic non-decreasing;
/// going backwards is not supported.
#[derive(Debug, Clone)]
pub struct Governor {
    step: usize,
    samples: VecDeque<(f64, f64)>,
    last_step_at: Option<f64>,
}

impl Default for Governor {
    fn default() -> Self {
        Self::new()
    }
}

impl Governor {
    /// Creates a governor at scale 1.0 with no samples.
    pub fn new() -> Self {
        Self {
            step: 0,
            samples: VecDeque::with_capacity(CAPACITY),
            last_step_at: None,
        }
    }

    /// Current render scale.
    pub fn scale(&self) -> f32 {
        STEPS[self.step]
    }

    /// Number of samples currently held (bounded by the ring capacity).
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Records one frame time at `now` seconds. Returns the new scale only
    /// when it changed.
    pub fn record(&mut self, now: f64, frame_ms: f64, budget_ms: f64) -> Option<f32> {
        if !now.is_finite()
            || !frame_ms.is_finite()
            || frame_ms < 0.0
            || !budget_ms.is_finite()
            || budget_ms <= 0.0
        {
            return None;
        }
        if self.samples.len() == CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back((now, frame_ms));
        while let Some(&(t, _)) = self.samples.front() {
            if now - t > WINDOW_S {
                self.samples.pop_front();
            } else {
                break;
            }
        }

        let oldest = self.samples.front()?.0;
        if now - oldest < MIN_WINDOW_S {
            return None;
        }
        if let Some(at) = self.last_step_at {
            if now - at < MIN_STEP_GAP_S {
                return None;
            }
        }
        let avg = self.samples.iter().map(|s| s.1).sum::<f64>() / self.samples.len() as f64;

        let next = if avg > DOWN_FRAC * budget_ms && self.step + 1 < STEPS.len() {
            self.step + 1
        } else if avg < UP_FRAC * budget_ms && self.step > 0 {
            self.step - 1
        } else {
            return None;
        };
        self.step = next;
        self.last_step_at = Some(now);
        self.samples.clear();
        Some(STEPS[next])
    }
}
