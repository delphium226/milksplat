//! Per-preset reactivity: the scaling of a tick's audio values, the level
//! steps, and the effective level (per-preset level times the global one).

use ms_analysis::AudioFrame;
use ms_engine::{
    effective_reactivity, step_reactivity, Reactivity, BAND_CEILING, REACTIVITY_MAX,
    REACTIVITY_MIN, SPECTRUM_CEILING, WAVE_CEILING,
};

fn frame() -> AudioFrame {
    let mut a = AudioFrame {
        bass: 1.4,
        mid: 0.8,
        treb: 1.0,
        bass_att: 1.2,
        mid_att: 0.9,
        treb_att: 0.3,
        beat: 1.0,
        beat_strength: 2.2,
        silent_for: 0.5,
        ..AudioFrame::default()
    };
    for (i, s) in a.spectrum.iter_mut().enumerate() {
        *s = (i % 7) as f32 * 0.1;
    }
    for (i, w) in a.wave[0].iter_mut().enumerate() {
        *w = ((i % 9) as f32 - 4.0) * 0.05;
    }
    a.wave[1] = a.wave[0];
    a
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn level_one_is_bit_identical() {
    // Including values where `1 + (v - 1)` would round in f32.
    let mut a = frame();
    a.bass = 1e-9;
    a.treb = 123.456_79;
    a.spectrum[3] = 1e-30;
    a.wave[0][5] = -3e-12;
    let mut r = Reactivity::new(1.0);
    assert_eq!(*r.apply(&a), a);
    // And bit for bit (PartialEq would call -0.0 and 0.0 equal).
    let out = *r.apply(&a);
    assert_eq!(out.bass.to_bits(), a.bass.to_bits());
    assert_eq!(out.wave[0][5].to_bits(), a.wave[0][5].to_bits());
}

#[test]
fn bands_scale_their_deviation_from_one() {
    let a = frame();
    let mut r = Reactivity::new(2.0);
    let o = *r.apply(&a);
    assert!(approx(o.bass, 1.8), "{}", o.bass); // 1 + 2 * 0.4
    assert!(approx(o.mid, 0.6), "{}", o.mid); // 1 + 2 * -0.2
    assert!(approx(o.treb, 1.0));
    assert!(approx(o.bass_att, 1.4));
    assert!(approx(o.mid_att, 0.8));
    // 1 + 2 * -0.7 < 0: clamped at 0.
    assert_eq!(o.treb_att, 0.0);

    let mut r = Reactivity::new(0.5);
    let o = *r.apply(&a);
    assert!(approx(o.bass, 1.2));
    assert!(approx(o.mid, 0.9));
    assert!(approx(o.treb_att, 0.65));
}

#[test]
fn level_zero_holds_bands_at_one_and_silences_the_rest() {
    let o = *Reactivity::new(0.0).apply(&frame());
    for v in [o.bass, o.mid, o.treb, o.bass_att, o.mid_att, o.treb_att] {
        assert_eq!(v, 1.0);
    }
    assert!(o.spectrum.iter().all(|&s| s == 0.0));
    assert!(o.wave.iter().flatten().all(|&w| w == 0.0));
}

#[test]
fn beat_time_and_silence_are_untouched() {
    let a = frame();
    for k in [0.0, 0.25, 2.5] {
        let o = *Reactivity::new(k).apply(&a);
        assert_eq!(o.beat, a.beat);
        assert_eq!(o.beat_strength, a.beat_strength);
        assert_eq!(o.silent_for, a.silent_for);
    }
}

#[test]
fn wave_scales_about_zero_and_spectrum_proportionally() {
    let a = frame();
    let o = *Reactivity::new(2.0).apply(&a);
    for (x, y) in a.wave.iter().flatten().zip(o.wave.iter().flatten()) {
        assert!(approx(*y, 2.0 * x), "{x} -> {y}");
    }
    for (x, y) in a.spectrum.iter().zip(&o.spectrum) {
        assert!(approx(*y, 2.0 * x), "{x} -> {y}");
    }
}

#[test]
fn turning_up_never_exceeds_the_ceiling_or_the_unscaled_value() {
    let mut a = frame();
    a.bass = 2.5; // 1 + 2.5 * 1.5 = 4.75 > the ceiling
    a.mid = 40.0; // already above the ceiling: no further than 1x
    a.spectrum[0] = 2.0;
    a.spectrum[1] = 50.0;
    a.wave[0][0] = 0.9;
    a.wave[0][1] = -0.9;
    a.wave[0][2] = 3.0;
    let o = *Reactivity::new(REACTIVITY_MAX).apply(&a);
    assert_eq!(o.bass, BAND_CEILING);
    assert_eq!(o.mid, 40.0);
    assert_eq!(o.spectrum[0], SPECTRUM_CEILING);
    assert_eq!(o.spectrum[1], 50.0);
    assert_eq!(o.wave[0][0], WAVE_CEILING);
    assert_eq!(o.wave[0][1], -WAVE_CEILING);
    assert_eq!(o.wave[0][2], 3.0);
}

#[test]
fn turning_up_keeps_the_order_of_values() {
    // Monotonic: a louder input never comes out quieter.
    let mut r = Reactivity::new(REACTIVITY_MAX);
    let mut last = -1.0f32;
    for i in 0..2000 {
        let a = AudioFrame {
            bass: i as f32 * 0.01,
            ..AudioFrame::default()
        };
        let v = r.apply(&a).bass;
        assert!(v >= last, "bass {} -> {v} after {last}", a.bass);
        last = v;
    }
}

#[test]
fn non_finite_inputs_stay_finite() {
    let mut a = frame();
    a.bass = f32::NAN;
    a.mid = f32::INFINITY;
    a.spectrum[0] = f32::NAN;
    a.wave[0][0] = f32::NEG_INFINITY;
    let o = *Reactivity::new(2.0).apply(&a);
    assert!(o.bass.is_finite() && o.mid.is_finite());
    assert!(o.spectrum[0].is_finite() && o.wave[0][0].is_finite());
}

#[test]
fn set_changes_the_level() {
    let mut r = Reactivity::new(1.0);
    r.set(2.0);
    assert_eq!(r.level(), 2.0);
    assert!(approx(r.apply(&frame()).bass, 1.8));
    r.set(f32::NAN);
    assert_eq!(r.level(), 1.0);
}

#[test]
fn steps_are_quarters_clamped_to_the_range() {
    assert_eq!(step_reactivity(1.0, 1), 1.25);
    assert_eq!(step_reactivity(1.0, -1), 0.75);
    assert_eq!(step_reactivity(REACTIVITY_MAX, 1), REACTIVITY_MAX);
    assert_eq!(step_reactivity(REACTIVITY_MIN, -1), REACTIVITY_MIN);
    assert_eq!(step_reactivity(0.25, -3), 0.25);
    // An off-grid level (hand-edited config) snaps to the grid.
    assert_eq!(step_reactivity(1.1, 1), 1.25);
    assert_eq!(step_reactivity(1.1, -1), 0.75);
    assert_eq!(step_reactivity(f32::NAN, 1), 1.25);
}

#[test]
fn effective_level_is_the_clamped_product() {
    assert_eq!(effective_reactivity(1.0, 1.0), 1.0);
    assert_eq!(effective_reactivity(1.5, 0.5), 0.75);
    assert_eq!(effective_reactivity(2.5, 2.0), REACTIVITY_MAX);
    assert_eq!(effective_reactivity(3.0, 2.0), REACTIVITY_MAX);
    assert_eq!(REACTIVITY_MAX, 5.0);
    assert_eq!(effective_reactivity(0.5, 0.0), 0.0);
    assert_eq!(effective_reactivity(1.0, -1.0), 0.0);
    assert_eq!(effective_reactivity(f32::NAN, 1.0), 1.0);
    assert_eq!(effective_reactivity(1.0, f32::INFINITY), 1.0);
}
