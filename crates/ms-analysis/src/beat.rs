//! Bass-flux beat detection.

const HISTORY: usize = 60;
/// Flux values required before any beat may be reported.
const WARMUP: usize = 10;
const K: f64 = 1.5;
const MIN_FLUX: f64 = 1e-6;
const MIN_GAP: f64 = 0.150;
const DECAY_SECONDS: f64 = 0.2;
const MIN_THRESHOLD: f64 = 1e-12;

/// Onset detector over the bass band's positive energy flux.
pub(crate) struct BeatDetector {
    ring: [f64; HISTORY],
    len: usize,
    head: usize,
    prev_bass: f64,
    was_silent: bool,
    last_onset: f64,
    onset_strength: f64,
}

impl BeatDetector {
    pub(crate) fn new() -> Self {
        Self {
            ring: [0.0; HISTORY],
            len: 0,
            head: 0,
            prev_bass: 0.0,
            was_silent: false,
            last_onset: f64::NEG_INFINITY,
            onset_strength: 0.0,
        }
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns `(beat, beat_strength)` for this tick.
    pub(crate) fn update(
        &mut self,
        bass_energy: f32,
        now: f64,
        _dt: f64,
        silent: bool,
    ) -> (f32, f32) {
        let mut beat = 0.0;
        if silent {
            self.was_silent = true;
        } else {
            if self.was_silent {
                self.was_silent = false;
                self.prev_bass = 0.0;
            }
            let e = if bass_energy.is_finite() {
                bass_energy as f64
            } else {
                0.0
            };
            let flux = (e - self.prev_bass).max(0.0);
            self.prev_bass = e;

            let n = self.len as f64;
            let (mean, std) = if self.len == 0 {
                (0.0, 0.0)
            } else {
                let slice = &self.ring[..self.len];
                let mean = slice.iter().sum::<f64>() / n;
                let var = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
                (mean, var.sqrt())
            };
            let threshold = (mean + K * std).max(MIN_THRESHOLD);

            if self.len >= WARMUP
                && flux > threshold
                && flux > MIN_FLUX
                && now - self.last_onset >= MIN_GAP
            {
                beat = 1.0;
                self.last_onset = now;
                self.onset_strength = (flux / threshold).min(f32::MAX as f64);
            }

            self.ring[self.head] = flux;
            self.head = (self.head + 1) % HISTORY;
            self.len = (self.len + 1).min(HISTORY);
        }

        let elapsed = now - self.last_onset;
        let strength = if elapsed.is_finite() && (0.0..DECAY_SECONDS).contains(&elapsed) {
            self.onset_strength * (1.0 - elapsed / DECAY_SECONDS)
        } else {
            0.0
        };
        let strength = if strength.is_finite() {
            strength as f32
        } else {
            0.0
        };
        (beat, strength)
    }
}
