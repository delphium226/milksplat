use ms_analysis::{Analyzer, AudioHistory, Frame};

const RATE: u32 = 48000;
const EDGE: f64 = 0.002;
/// Measured onset lag is a constant 19.98 ms: the kick enters the Hann-windowed
/// FFT's tail (tiny flux on the first tick), and ticks run at 10 ms packet
/// boundaries rather than on the exact 1/60 s grid. That is 1 tick + 3.3 ms,
/// so the bound is 20.1 ms, not the nominal 1/60 s.
const MAX_LAG: f64 = 0.0201;
const KICK_LEN: f64 = 0.03;

/// 50 Hz sine burst, 30 ms, with 2 ms raised-cosine edges.
fn kick(t_rel: f64) -> f32 {
    if !(0.0..KICK_LEN).contains(&t_rel) {
        return 0.0;
    }
    let fade = if t_rel < EDGE {
        0.5 - 0.5 * (std::f64::consts::PI * t_rel / EDGE).cos()
    } else if t_rel > KICK_LEN - EDGE {
        0.5 - 0.5 * (std::f64::consts::PI * (KICK_LEN - t_rel) / EDGE).cos()
    } else {
        1.0
    };
    (0.8 * fade * (2.0 * std::f64::consts::PI * 50.0 * t_rel).sin()) as f32
}

/// Deterministic faint noise bed (about -60 dBFS) so gaps between kicks are not
/// digital silence (which the analyzer treats as silent and skips).
fn bed(n: u64) -> f32 {
    let mut x = n.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * 1e-3
}

/// Kicks starting at `first + k*period` while `active(start)` holds.
fn kick_train(first: f64, period: f64, until: f64) -> impl Fn(f64) -> f32 {
    move |t| {
        if t < first {
            return 0.0;
        }
        let k = ((t - first) / period).floor();
        if first + k * period >= until {
            return 0.0;
        }
        kick(t - first - k * period)
    }
}

/// Returns (tick time, beat, strength) per tick.
fn run(seconds: f64, sig: impl FnMut(f64) -> f32) -> Vec<(f64, f32, f32)> {
    run_bed(seconds, false, sig)
}

fn run_bed(seconds: f64, with_bed: bool, mut sig: impl FnMut(f64) -> f32) -> Vec<(f64, f32, f32)> {
    let mut hist = AudioHistory::new(RATE, 4.0);
    let mut an = Analyzer::new(RATE);
    let per_packet = (RATE / 100) as usize;
    let mut next_tick = 1.0 / 60.0;
    let mut n: u64 = 0;
    let mut buf: Vec<Frame> = Vec::new();
    let mut out = Vec::new();
    for _ in 0..(seconds * 100.0) as usize {
        buf.clear();
        for _ in 0..per_packet {
            let v = sig(n as f64 / RATE as f64) + if with_bed { bed(n) } else { 0.0 };
            buf.push([v, v]);
            n += 1;
        }
        let t = (n - 1) as f64 / RATE as f64;
        hist.push(&buf, t);
        while next_tick <= t {
            let f = an.tick(&hist, t, 0.0, 1.0 / 60.0);
            assert!(f.beat.is_finite() && f.beat_strength.is_finite());
            assert!(f.beat == 0.0 || f.beat == 1.0);
            assert!(f.beat_strength >= 0.0);
            out.push((t, f.beat, f.beat_strength));
            next_tick += 1.0 / 60.0;
        }
    }
    out
}

fn beat_times(rows: &[(f64, f32, f32)]) -> Vec<f64> {
    rows.iter().filter(|r| r.1 > 0.0).map(|r| r.0).collect()
}

#[test]
fn click_track_120bpm() {
    let rows = run_bed(10.0, true, kick_train(0.0, 0.5, f64::INFINITY));
    let beats = beat_times(&rows);
    let mut lags = Vec::new();
    for &b in &beats {
        let k = (b / 0.5).round();
        let lag = b - k * 0.5;
        lags.push(lag);
        assert!((0.0..=MAX_LAG).contains(&lag), "beat {b} lag {lag}");
    }
    eprintln!("beat times {beats:?}\nlags {lags:?}");
    assert!((19..=21).contains(&beats.len()), "{} beats", beats.len());
}

#[test]
fn refractory() {
    let rows = run_bed(6.0, true, kick_train(0.0, 0.1, f64::INFINITY));
    let beats = beat_times(&rows);
    eprintln!("refractory beats {}", beats.len());
    for w in beats.windows(2) {
        assert!(w[1] - w[0] >= 0.15 - 1e-9, "{w:?}");
    }
}

#[test]
fn steady_tone_no_beats() {
    let rows = run(6.0, |t| {
        (0.5 * (2.0 * std::f64::consts::PI * 60.0 * t).sin()) as f32
    });
    let late: Vec<_> = rows.iter().filter(|r| r.0 > 1.0 && r.1 > 0.0).collect();
    assert!(late.is_empty(), "{late:?}");
}

#[test]
fn strength_decays() {
    let rows = run_bed(5.0, true, kick_train(0.0, 0.5, f64::INFINITY));
    let mut checked = 0;
    for (i, r) in rows.iter().enumerate() {
        if r.1 > 0.0 {
            assert!(r.2 > 0.0);
            let mut prev = r.2;
            for later in rows[i + 1..].iter().take_while(|l| l.0 - r.0 < 0.15) {
                assert!(later.2 <= prev + 1e-6, "not monotone");
                prev = later.2;
            }
            if let Some(l) = rows[i + 1..].iter().find(|l| l.0 - r.0 >= 0.2) {
                assert_eq!(l.2, 0.0, "t={} onset {}", l.0, r.0);
                checked += 1;
            }
        }
    }
    assert!(checked >= 5);
}

#[test]
fn silence_no_beat_then_beat_after() {
    // Kicks for 2 s, silence 2..5.1 s, kick at 5.1 s.
    let train = kick_train(0.0, 0.5, 2.0);
    let mut n = 0u64;
    let rows = run(6.0, |t| {
        n += 1;
        let b = if t < 2.0 { bed(n) } else { 0.0 };
        train(t) + kick(t - 5.1) + b
    });
    for r in rows.iter().filter(|r| r.0 > 2.2 && r.0 < 5.1) {
        assert_eq!(r.1, 0.0, "beat during silence at {}", r.0);
    }
    let after: Vec<f64> = beat_times(&rows).into_iter().filter(|&t| t > 5.0).collect();
    eprintln!("beats after silence {after:?}");
    assert!(!after.is_empty());
    assert!((0.0..=MAX_LAG).contains(&(after[0] - 5.1)));
}

#[test]
fn all_silence_finite_no_beats() {
    let rows = run(3.0, |_| 0.0);
    assert!(rows.iter().all(|r| r.1 == 0.0 && r.2 == 0.0));
}
