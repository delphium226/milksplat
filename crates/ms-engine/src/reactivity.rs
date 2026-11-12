//! Per-preset reactivity: how strongly a preset responds to the music.
//!
//! A level `k` (the preset's level times the global one, see
//! [`effective_reactivity`]) rescales the audio values a preset sees, in a
//! copy of the tick's [`AudioFrame`]:
//!
//! - **Bands** (`bass`, `mid`, `treb` and the `_att` versions) are the
//!   band energy over its long-term average, so they hover around 1 at
//!   average loudness. Under the average they are scaled as a power,
//!   `v' = v^k` (the ratio's logarithm times `k`); over it their deviation
//!   is scaled, `v' = 1 + k (v - 1)`. The two meet at 1 with the same
//!   slope `k`. Average loudness stays at 1, so a preset does not simply
//!   get brighter or darker with the setting, and a quiet value stays
//!   graded and above 0 at any level: real music's bass sits under its
//!   average most of the time (median about 0.6 on techno), and scaling
//!   the deviation there too floored everything under `1 - 1/k` at 0,
//!   which at 5x left a preset a near on/off signal. Over the average the
//!   deviation scaling is kept because a power climbs much faster there
//!   (it reached the ceiling at 1.38 times the average at 5x, against 1.8)
//!   and flattened the loud hits into one value (ruling R78). Turned down
//!   (`k < 1`) both sides move towards 1; at 0 the bands are held at 1.
//! - **Waveform** samples are signed audio: scaled about 0, `w' = k w`.
//! - **Spectrum** bins are `ln(1 + |X| / 2)`, an absolute level that is 0
//!   in silence and has no running average to deviate from, so they are
//!   scaled proportionally, `s' = k s`: silence stays silent and louder
//!   bins get louder.
//! - `beat`, `beat_strength`, `silent_for` (and `time`, which is not
//!   audio) are not changed.
//!
//! **Flash-safety guard.** Turning a preset up must never push it past the
//! photosensitivity limits it was tuned to at 1x (no tick changing the
//! full-frame luminance by more than 0.06, no more than three full-frame
//! flashes a second). Two holds, both only on scaled values:
//!
//! - **Ceiling.** A value turned up is held to `max(ceiling, v)`
//!   ([`BAND_CEILING`], [`WAVE_CEILING`], [`SPECTRUM_CEILING`]): no higher
//!   than a loud value, unless the unscaled value is itself higher, in which
//!   case it stays at that unscaled value. The mapping stays monotonic (a
//!   louder input never comes out quieter). For bands the ceiling is a
//!   soft knee (ruling R79, `soft_ceiling`): the raise over the unscaled
//!   value passes unchanged up to half the headroom left under the
//!   ceiling, then bends smoothly towards the ceiling without reaching it,
//!   so loud hits stay graded (strictly: a louder value always comes out
//!   larger) instead of all landing on the ceiling. Waveform and spectrum
//!   keep a hard ceiling.
//! - **Slew limit.** A luminance step comes from how far an input moves in
//!   one tick, not from how high it is, and scaling by `k` makes every
//!   step `k` times larger. So each band and spectrum value may move at
//!   most [`SLEW_FACTOR`] times as far per tick as its unscaled value did,
//!   or [`SLEW_FACTOR`] times a small allowance ([`BAND_SLEW_ALLOWANCE`],
//!   [`SPECTRUM_SLEW_ALLOWANCE`]) when the unscaled value barely moves:
//!   `|v'(t) - v'(t-1)| <= SLEW_FACTOR * max(|v(t) - v(t-1)|, allowance)`.
//!   A preset turned up therefore steps at most `SLEW_FACTOR` times as far
//!   per tick as it does at 1x (for a preset whose look depends smoothly
//!   on its inputs), whatever the level. The limit does not undo the
//!   scaling: on a kick the turned-up value keeps climbing while the
//!   unscaled one decays (the decay moves too, so the limit is not zero),
//!   so it still peaks much higher, a few ticks later, and falls back as
//!   gradually. Up to `SLEW_FACTOR`x it never binds (the scaling is
//!   nowhere steeper than `k`, as `v^k` under 1 is no steeper than `k`
//!   for `k >= 1`, so `|dv'| <= k |dv| <= SLEW_FACTOR |dv|`), so those
//!   levels behave as plain scaling. A level change, including back to
//!   1x, also moves the values gradually; once
//!   back on the unscaled values at 1x, the frame is passed through again.
//!   The waveform is not slew-limited: its samples are new audio every
//!   tick, at 1x as much as turned up, and only the ceiling holds them.
//!   The first tick of a preset has nothing to step from, so it is not
//!   limited.
//!
//! Turning down only moves values towards their resting point, which a 1x
//! preset sees too, and binds the slew limit only for band values very
//! near 0 (where `v^k` is steep for `k < 1`).
//!
//! At exactly 1 (once settled, see above) the frame is passed through
//! untouched (bit-identical).

use ms_analysis::AudioFrame;

/// The lowest per-preset level (Down arrow).
pub const REACTIVITY_MIN: f32 = 0.25;
/// The highest per-preset level (Up arrow), and the highest effective
/// level (per-preset times global).
pub const REACTIVITY_MAX: f32 = 5.0;
/// One Up/Down step.
pub const REACTIVITY_STEP: f32 = 0.25;
/// The guard's ceiling for a turned-up band value (`bass`, `bass_att`,
/// ...). Average loudness is 1 and real music's beats reach 3 and well
/// beyond at 1x; with the slew limit holding the per-tick change, the
/// ceiling only bounds how far a turned-up value travels, and 5 lets a
/// preset keep gaining reaction all the way to 5x (chosen from every
/// bundled preset at 5x on real techno, see ruling R77). Approached
/// through a soft knee, never reached by a turned-up value (ruling R79).
pub const BAND_CEILING: f32 = 5.0;
/// The guard's ceiling for a turned-up waveform sample's magnitude (full
/// scale).
pub const WAVE_CEILING: f32 = 1.0;
/// The guard's ceiling for a turned-up spectrum bin (about the loudest
/// bins of real music at 1x).
pub const SPECTRUM_CEILING: f32 = 3.0;

/// How many times as far per tick as its unscaled value a turned-up band
/// or spectrum value may move (see the module docs).
pub const SLEW_FACTOR: f32 = 2.0;
/// The unscaled per-tick band change the slew limit allows for at least
/// (times [`SLEW_FACTOR`]), so a value can still travel to its scaled
/// value, or a new level, while the music moves little: about the median
/// per-tick change of `bass` on real techno at 1x (0.14), so a turned-up
/// value never steps further than 1x music routinely does.
pub const BAND_SLEW_ALLOWANCE: f32 = 0.1;
/// [`BAND_SLEW_ALLOWANCE`] for spectrum bins (`ln(1 + |X| / 2)`: real
/// techno's bins change by under 0.07 a tick 90% of the time).
pub const SPECTRUM_SLEW_ALLOWANCE: f32 = 0.05;

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

/// A preset's reactivity: its level, a scratch frame for the scaled
/// audio and the previous tick's input for the slew limit, so applying it
/// never allocates.
#[derive(Debug, Clone)]
pub struct Reactivity {
    k: f32,
    /// The last output when not passing through (what
    /// [`applied`](Self::applied) returns then).
    out: Box<AudioFrame>,
    /// The last input.
    prev: Box<AudioFrame>,
    /// Whether `prev` holds a previous tick.
    primed: bool,
    /// Whether the last output was the input itself (level 1, settled).
    through: bool,
}

impl Reactivity {
    /// A reactivity at level `k` (non-finite or negative: see [`set`](Self::set)).
    pub fn new(k: f32) -> Self {
        let mut r = Reactivity {
            k: 1.0,
            out: Box::default(),
            prev: Box::default(),
            primed: false,
            through: true,
        };
        r.set(k);
        r
    }

    /// Sets the level: clamped to `0..=REACTIVITY_MAX`; non-finite is 1.
    /// The values move to the new level gradually (the slew limit).
    pub fn set(&mut self, k: f32) {
        self.k = effective_reactivity(k, 1.0);
    }

    pub fn level(&self) -> f32 {
        self.k
    }

    /// The frame this preset sees: `a` itself at level 1 (once settled),
    /// else a scaled, guarded copy (see the module docs). Call it once a
    /// tick: the slew limit steps from the previous call.
    pub fn apply<'a>(&'a mut self, a: &'a AudioFrame) -> &'a AudioFrame {
        if self.k == 1.0 && self.through {
            // Remember the tick for the slew limit should the level change.
            *self.prev = *a;
            self.primed = true;
            return a;
        }
        let k = self.k;
        let first = !self.primed;
        let (o, p) = (&mut *self.out, &*self.prev);
        if self.through {
            // The last output was the input itself: step from it.
            *o = *p;
        }
        // `target` slew-limited from the last output `last`, the unscaled
        // value having moved from `pv` to `v`.
        let step = |v: f32, pv: f32, last: f32, target: f32, allowance: f32| {
            if first {
                return target;
            }
            let dv = (v - pv).abs();
            let dv = if dv.is_finite() { dv } else { 0.0 };
            let max = SLEW_FACTOR * dv.max(allowance);
            target.clamp(last - max, last + max)
        };
        let finite = |v: f32| if v.is_finite() { v } else { 0.0 };
        let mut settled = k == 1.0;
        let mut band = |v: f32, pv: f32, last: &mut f32| {
            let target = if k == 1.0 {
                finite(v)
            } else {
                soft_ceiling(scale_band(v, k), v, BAND_CEILING)
            };
            let n = step(v, pv, *last, target, BAND_SLEW_ALLOWANCE);
            settled &= n.to_bits() == v.to_bits();
            *last = n;
        };
        band(a.bass, p.bass, &mut o.bass);
        band(a.mid, p.mid, &mut o.mid);
        band(a.treb, p.treb, &mut o.treb);
        band(a.bass_att, p.bass_att, &mut o.bass_att);
        band(a.mid_att, p.mid_att, &mut o.mid_att);
        band(a.treb_att, p.treb_att, &mut o.treb_att);
        for ((d, &s), &ps) in o.spectrum.iter_mut().zip(&a.spectrum).zip(&p.spectrum) {
            let target = if k == 1.0 {
                finite(s)
            } else {
                guard(k * s, s, SPECTRUM_CEILING, 0.0)
            };
            let n = step(s, ps, *d, target, SPECTRUM_SLEW_ALLOWANCE);
            settled &= n.to_bits() == s.to_bits();
            *d = n;
        }
        if k == 1.0 {
            o.wave = a.wave;
        } else {
            for (dc, sc) in o.wave.iter_mut().zip(&a.wave) {
                for (d, &w) in dc.iter_mut().zip(sc) {
                    *d = if w < 0.0 {
                        -guard(-k * w, -w, WAVE_CEILING, 0.0)
                    } else {
                        guard(k * w, w, WAVE_CEILING, 0.0)
                    };
                }
            }
        }
        o.beat = a.beat;
        o.beat_strength = a.beat_strength;
        o.silent_for = a.silent_for;
        *self.prev = *a;
        self.primed = true;
        // Back on the unscaled values at 1x: pass through from the next tick.
        self.through = settled;
        &self.out
    }

    /// The frame the last [`apply`](Self::apply) of `a` gave, without
    /// computing it again: `a` itself when passing through.
    pub fn applied<'a>(&'a self, a: &'a AudioFrame) -> &'a AudioFrame {
        if self.k == 1.0 && self.through {
            a
        } else {
            &self.out
        }
    }
}

/// A band value at level `k`: `v^k` up to the average (1), `1 + k (v - 1)`
/// above it (0 for a negative or non-finite `v`; at `k = 0`, 1). The two
/// meet at 1 with the same slope, `k`. See the module docs.
#[inline]
fn scale_band(v: f32, k: f32) -> f32 {
    if !(v.is_finite() && v > 0.0) {
        return if k == 0.0 { 1.0 } else { 0.0 };
    }
    if v <= 1.0 {
        v.powf(k)
    } else {
        1.0 + k * (v - 1.0)
    }
}

/// A turned-up band value `scaled` held under `ceiling` by a soft knee
/// (ruling R79), relative to its unscaled value `unscaled`. With `a` the
/// excess over the unscaled value (`scaled - unscaled`) and `m` half the
/// headroom (`(ceiling - unscaled) / 2`), the excess passes unchanged up
/// to `m`, then bends smoothly (same slope at the knee) towards `2m`:
/// `unscaled + m (2 - m / a)`, so the value approaches the ceiling but
/// never reaches it. In `unscaled` it is strictly increasing (its slope is
/// `m/a + (k-1) m^2/a^2`, positive and at most `k`), so a louder hit
/// always gives a larger value and the slew limit's bound on the slope
/// still holds. A value not raised (`a <= 0`) passes; an unscaled value at
/// or over the ceiling is kept (never held below it). Never negative;
/// non-finite gives 0.
#[inline]
fn soft_ceiling(scaled: f32, unscaled: f32, ceiling: f32) -> f32 {
    if !scaled.is_finite() {
        return 0.0;
    }
    let s = scaled.max(0.0);
    if !unscaled.is_finite() {
        return s.min(ceiling);
    }
    let a = s - unscaled;
    if a <= 0.0 {
        return s;
    }
    let m = 0.5 * (ceiling - unscaled);
    if m <= 0.0 {
        return unscaled;
    }
    if a <= m {
        s
    } else {
        // Under the ceiling in exact arithmetic; `min` guards the rounding.
        (unscaled + m * (2.0 - m / a)).min(ceiling)
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
