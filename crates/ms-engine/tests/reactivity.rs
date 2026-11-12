//! Per-preset reactivity: the scaling of a tick's audio values, the level
//! steps, and the effective level (per-preset level times the global one).

use ms_analysis::AudioFrame;
use ms_engine::{
    effective_reactivity, step_reactivity, Reactivity, BAND_CEILING, BAND_SLEW_ALLOWANCE,
    REACTIVITY_MAX, REACTIVITY_MIN, SLEW_FACTOR, SPECTRUM_CEILING, SPECTRUM_SLEW_ALLOWANCE,
    WAVE_CEILING,
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
    a.bass = 2.5; // 1 + 5 * 1.5 = 8.5 > the ceiling
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

// ------------------------------------------------------------ slew limit

/// A deterministic pseudo-random value in [0, 1) for tick `n`, stream `s`.
fn rnd(s: u32, n: u32) -> f32 {
    let mut x = (u64::from(s) << 32 | u64::from(n)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

/// A kick-drum-like bass envelope at 2 Hz (a 60 Hz tick): a two-tick
/// attack to `peak`, then an exponential decay back towards `rest`, with a
/// little noise, as the analyser gives for techno.
fn kick(n: u32, rest: f32, peak: f32) -> f32 {
    let t = n % 30;
    let env = match t {
        0 => 0.5,
        1 => 1.0,
        _ => (-((t - 1) as f32) / 5.0).exp(),
    };
    rest + (peak - rest) * env + 0.05 * (rnd(9, n) - 0.5)
}

/// Ticks of audio: a band walk with kicks and jumps, and spectrum noise.
fn walk(ticks: u32) -> Vec<AudioFrame> {
    (0..ticks)
        .map(|n| {
            let mut a = frame();
            a.bass = kick(n, 0.8, 1.5);
            a.bass_att = 0.8 + 0.5 * (n as f32 * 0.05).sin();
            // Mostly smooth, with an occasional big jump.
            a.mid = if n % 97 < 3 {
                3.5
            } else {
                1.0 + 0.3 * (n as f32 * 0.11).sin()
            };
            a.mid_att = 1.0 + 0.2 * rnd(1, n);
            a.treb = 0.5 + 1.5 * rnd(2, n);
            a.treb_att = 1.0;
            for (i, s) in a.spectrum.iter_mut().enumerate() {
                *s = 1.2 * rnd(3 + i as u32, n);
            }
            a
        })
        .collect()
}

fn bands(a: &AudioFrame) -> [f32; 6] {
    [a.bass, a.mid, a.treb, a.bass_att, a.mid_att, a.treb_att]
}

/// The limit on one tick's change of a turned-up value whose unscaled
/// value moved by `d`, with a small tolerance for rounding.
fn limit(d: f32, allowance: f32) -> f32 {
    SLEW_FACTOR * d.abs().max(allowance) * (1.0 + 1e-5) + 1e-6
}

#[test]
fn every_scaled_step_is_slew_limited() {
    let audio = walk(600);
    for k in [REACTIVITY_MAX, 3.0, 2.0, 0.5] {
        let mut r = Reactivity::new(k);
        let mut last = *r.apply(&audio[0]);
        for (n, w) in audio.windows(2).enumerate() {
            let o = *r.apply(&w[1]);
            for ((p, q), (u, v)) in bands(&last)
                .iter()
                .zip(bands(&o))
                .zip(bands(&w[0]).iter().zip(bands(&w[1])))
            {
                assert!(
                    (q - p).abs() <= limit(v - u, BAND_SLEW_ALLOWANCE),
                    "{k}x tick {n}: band {p} -> {q} while unscaled {u} -> {v}"
                );
            }
            for (i, ((p, q), (u, v))) in last
                .spectrum
                .iter()
                .zip(&o.spectrum)
                .zip(w[0].spectrum.iter().zip(&w[1].spectrum))
                .enumerate()
            {
                assert!(
                    (q - p).abs() <= limit(v - u, SPECTRUM_SLEW_ALLOWANCE),
                    "{k}x tick {n}: bin {i} {p} -> {q} while unscaled {u} -> {v}"
                );
            }
            last = o;
        }
    }
}

fn std_dev(v: &[f32]) -> f32 {
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32).sqrt()
}

/// The slew limit must not undo the scaling: on kick-like bass, each step
/// up in level still clearly widens what the preset sees.
#[test]
fn turning_up_still_widens_the_reaction() {
    let audio = walk(1200);
    let spread = |k: f32| {
        let mut r = Reactivity::new(k);
        let bass: Vec<f32> = audio.iter().map(|a| r.apply(a).bass).skip(60).collect();
        let lo = bass.iter().copied().fold(f32::MAX, f32::min);
        let hi = bass.iter().copied().fold(f32::MIN, f32::max);
        (std_dev(&bass), hi - lo)
    };
    let (sd1, range1) = spread(1.0);
    let (sd25, range25) = spread(2.5);
    let (sd5, range5) = spread(REACTIVITY_MAX);
    eprintln!("bass sd {sd1:.3} {sd25:.3} {sd5:.3}, range {range1:.3} {range25:.3} {range5:.3}");
    assert!(sd25 > 2.0 * sd1, "2.5x: sd {sd25} vs 1x {sd1}");
    assert!(sd5 > 1.3 * sd25, "5x: sd {sd5} vs 2.5x {sd25}");
    assert!(
        range5 > 1.3 * range25,
        "5x: range {range5} vs 2.5x {range25}"
    );
    // Against 1x: the limit costs some of the peak (each kick's attack is
    // only two ticks long), but 5x still moves well over twice as far.
    assert!(sd5 > 3.0 * sd1, "5x: sd {sd5} vs 1x {sd1}");
    assert!(range5 > 2.5 * range1, "5x: range {range5} vs 1x {range1}");
}

/// A level change while the music holds steady moves the values gradually
/// to the new level instead of in one step.
#[test]
fn a_level_change_is_slewed() {
    let mut a = frame();
    a.bass = 1.6;
    a.spectrum[0] = 0.5;
    let mut r = Reactivity::new(1.0);
    for _ in 0..10 {
        r.apply(&a);
    }
    r.set(REACTIVITY_MAX);
    let mut last = a.bass;
    let mut s0 = a.spectrum[0];
    for _ in 0..1000 {
        let o = *r.apply(&a);
        assert!((o.bass - last).abs() <= limit(0.0, BAND_SLEW_ALLOWANCE));
        assert!((o.spectrum[0] - s0).abs() <= limit(0.0, SPECTRUM_SLEW_ALLOWANCE));
        last = o.bass;
        s0 = o.spectrum[0];
    }
    // Settled on the scaled value: 1 + 5 * 0.6 = 4, under the ceiling.
    assert!(approx(last, 1.0 + REACTIVITY_MAX * 0.6), "{last}");
    assert!(approx(s0, REACTIVITY_MAX * 0.5), "{s0}");
}

/// Going back to 1x (Backspace) slews back to the unscaled values, then
/// passes the frame through bit for bit again.
#[test]
fn back_to_one_slews_then_passes_through() {
    let mut a = frame();
    a.bass = 1.6;
    let mut r = Reactivity::new(REACTIVITY_MAX);
    for _ in 0..1000 {
        r.apply(&a);
    }
    let high = r.apply(&a).bass;
    assert!(high > 3.0, "{high}");
    r.set(1.0);
    let first = r.apply(&a).bass;
    assert!(first < high && first > a.bass, "{high} -> {first}");
    assert!((high - first) <= limit(0.0, BAND_SLEW_ALLOWANCE));
    for _ in 0..1000 {
        r.apply(&a);
    }
    a.wave[0][5] = -3e-12;
    let out = *r.apply(&a);
    assert_eq!(out, a);
    assert_eq!(out.bass.to_bits(), a.bass.to_bits());
    assert!(std::ptr::eq(r.applied(&a), &a));
}
