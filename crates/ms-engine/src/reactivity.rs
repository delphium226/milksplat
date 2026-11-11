//! Per-preset reactivity: how strongly a preset responds to the music.
//!
//! A level `k` (the preset's level times the global one, see
//! [`effective_reactivity`]) rescales the audio values a preset sees, in a
//! copy of the tick's [`AudioFrame`]:
//!
//! - **Bands** (`bass`, `mid`, `treb` and the `_att` versions) are the
//!   band energy over its long-term average, so they hover around 1 at
//!   average loudness. Their deviation from 1 is scaled:
//!   `v' = 1 + k (v - 1)`, never below 0. Average loudness stays at 1, so a
//!   preset does not simply get brighter or darker with the setting.
//! - **Waveform** samples are signed audio: scaled about 0, `w' = k w`.
//! - **Spectrum** bins are `ln(1 + |X| / 2)`, an absolute level that is 0
//!   in silence and has no running average to deviate from, so they are
//!   scaled proportionally, `s' = k s`: silence stays silent and louder
//!   bins get louder.
//! - `beat`, `beat_strength`, `silent_for` (and `time`, which is not
//!   audio) are not changed.
//!
//! **Flash-safety guard.** Turning a preset up must never push it past the
//! photosensitivity limit it was tuned to at 1x. A value turned up is held
//! to `max(ceiling, v)`: no higher than a plausible unscaled value
//! ([`BAND_CEILING`], [`WAVE_CEILING`], [`SPECTRUM_CEILING`]), unless the
//! unscaled value is itself higher, in which case it stays at that
//! unscaled value. A turned-up preset therefore never sees an input larger
//! than one a 1x preset could see at that moment, and the mapping stays
//! monotonic (a louder input never comes out quieter). Turning down only
//! moves values towards their resting point, which a 1x preset sees too.
//!
//! At exactly 1 the frame is passed through untouched (bit-identical).

use ms_analysis::AudioFrame;

/// The lowest per-preset level (Down arrow).
pub const REACTIVITY_MIN: f32 = 0.25;
/// The highest per-preset level (Up arrow), and the highest effective
/// level (per-preset times global).
pub const REACTIVITY_MAX: f32 = 2.5;
/// One Up/Down step.
pub const REACTIVITY_STEP: f32 = 0.25;
/// The guard's ceiling for a turned-up band value (`bass`, `bass_att`,
/// ...): about the largest the synthetic test music gives at 1x.
pub const BAND_CEILING: f32 = 3.0;
/// The guard's ceiling for a turned-up waveform sample's magnitude (full
/// scale).
pub const WAVE_CEILING: f32 = 1.0;
/// The guard's ceiling for a turned-up spectrum bin (about the loudest
/// bins of real music at 1x).
pub const SPECTRUM_CEILING: f32 = 3.0;

/// The level actually applied: `preset * global`, clamped to
/// `0..=REACTIVITY_MAX`. A non-finite factor counts as 1.
pub fn effective_reactivity(preset: f32, global: f32) -> f32 {
    let f = |x: f32| if x.is_finite() { x } else { 1.0 };
    (f(preset) * f(global)).clamp(0.0, REACTIVITY_MAX)
}

/// `level` moved by `steps` quarter steps: snapped to the nearest step
/// first (a hand-edited level may be off the grid; non-finite counts as
/// 1), then clamped to `REACTIVITY_MIN..=REACTIVITY_MAX`.
pub fn step_reactivity(level: f32, steps: i32) -> f32 {
    let level = if level.is_finite() { level } else { 1.0 };
    let n = (level / REACTIVITY_STEP).round() + steps as f32;
    (n * REACTIVITY_STEP).clamp(REACTIVITY_MIN, REACTIVITY_MAX)
}

/// A preset's reactivity: its level and a scratch frame for the scaled
/// audio, so applying it never allocates.
#[derive(Debug, Clone)]
pub struct Reactivity {
    k: f32,
    out: Box<AudioFrame>,
}

impl Reactivity {
    /// A reactivity at level `k` (non-finite or negative: see [`set`](Self::set)).
    pub fn new(k: f32) -> Self {
        let mut r = Reactivity {
            k: 1.0,
            out: Box::default(),
        };
        r.set(k);
        r
    }

    /// Sets the level: clamped to `0..=REACTIVITY_MAX`; non-finite is 1.
    pub fn set(&mut self, k: f32) {
        self.k = effective_reactivity(k, 1.0);
    }

    pub fn level(&self) -> f32 {
        self.k
    }

    /// The frame this preset sees: `a` itself at level 1, else a scaled
    /// copy (see the module docs).
    pub fn apply<'a>(&'a mut self, a: &'a AudioFrame) -> &'a AudioFrame {
        if self.k == 1.0 {
            return a;
        }
        let k = self.k;
        let o = &mut *self.out;
        *o = *a;
        let band = |v: f32| guard(1.0 + k * (v - 1.0), v, BAND_CEILING, 0.0);
        o.bass = band(a.bass);
        o.mid = band(a.mid);
        o.treb = band(a.treb);
        o.bass_att = band(a.bass_att);
        o.mid_att = band(a.mid_att);
        o.treb_att = band(a.treb_att);
        for (d, &s) in o.spectrum.iter_mut().zip(&a.spectrum) {
            *d = guard(k * s, s, SPECTRUM_CEILING, 0.0);
        }
        for (dc, sc) in o.wave.iter_mut().zip(&a.wave) {
            for (d, &w) in dc.iter_mut().zip(sc) {
                *d = if w < 0.0 {
                    -guard(-k * w, -w, WAVE_CEILING, 0.0)
                } else {
                    guard(k * w, w, WAVE_CEILING, 0.0)
                };
            }
        }
        o
    }
}

/// `scaled` held to `floor..=max(ceiling, unscaled)`; non-finite gives the
/// floor.
#[inline]
fn guard(scaled: f32, unscaled: f32, ceiling: f32, floor: f32) -> f32 {
    if !scaled.is_finite() {
        return floor;
    }
    let top = if unscaled.is_finite() {
        ceiling.max(unscaled)
    } else {
        ceiling
    };
    scaled.clamp(floor, top)
}
