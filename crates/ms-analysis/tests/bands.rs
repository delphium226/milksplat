use ms_analysis::{Analyzer, AudioFrame, AudioHistory, Frame};

struct Rng(u64);
impl Rng {
    fn white(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    }
}

/// Paul Kellet pink noise.
struct Pink {
    rng: Rng,
    b: [f32; 7],
}
impl Pink {
    fn new(seed: u64) -> Self {
        Self {
            rng: Rng(seed),
            b: [0.0; 7],
        }
    }
    fn next(&mut self) -> f32 {
        let w = self.rng.white();
        let b = &mut self.b;
        b[0] = 0.99886 * b[0] + w * 0.0555179;
        b[1] = 0.99332 * b[1] + w * 0.0750759;
        b[2] = 0.96900 * b[2] + w * 0.153852;
        b[3] = 0.86650 * b[3] + w * 0.3104856;
        b[4] = 0.55000 * b[4] + w * 0.5329522;
        b[5] = -0.7616 * b[5] - w * 0.0168980;
        let out = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362;
        b[6] = w * 0.115926;
        out * 0.11
    }
}

fn run<S, F>(rate: u32, seconds: f64, sig: S, each: F) -> (Analyzer, AudioFrame)
where
    S: FnMut(f64) -> f32,
    F: FnMut(f64, &Analyzer, &AudioFrame),
{
    run_from(rate, seconds, 1.0 / 60.0, sig, each)
}

/// Feeds `sig(t, channel_index_sample)` at `rate`, ticking at 60 Hz; calls `each` after each tick.
fn run_from<S, F>(
    rate: u32,
    seconds: f64,
    first_tick: f64,
    mut sig: S,
    mut each: F,
) -> (Analyzer, AudioFrame)
where
    S: FnMut(f64) -> f32,
    F: FnMut(f64, &Analyzer, &AudioFrame),
{
    let mut hist = AudioHistory::new(rate, 4.0);
    let mut an = Analyzer::new(rate);
    let r = rate as f64;
    let per_packet = (rate / 100) as usize;
    let packets = (seconds * 100.0) as usize;
    let mut next_tick = first_tick;
    let mut n: u64 = 0;
    let mut last = AudioFrame::default();
    let mut buf: Vec<Frame> = Vec::new();
    for _ in 0..packets {
        buf.clear();
        for _ in 0..per_packet {
            let v = sig(n as f64 / r);
            buf.push([v, v]);
            n += 1;
        }
        let t = (n - 1) as f64 / r;
        hist.push(&buf, t);
        while next_tick <= t {
            let f = *an.tick(&hist, t, 0.0, 1.0 / 60.0);
            each(t, &an, &f);
            last = f;
            next_tick += 1.0 / 60.0;
        }
    }
    (an, last)
}

fn sine(freq: f64) -> impl FnMut(f64) -> f32 {
    move |t| (0.5 * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32
}

fn dominant(freq: f64) {
    let (an, f) = run(48000, 3.0, sine(freq), |_, _, _| {});
    let e = an.raw_energy();
    let (b, m, t) = (e[0], e[1], e[2]);
    let main = b.max(m).max(t);
    assert!(main > 0.0);
    for (i, v) in e.iter().enumerate() {
        let want = (freq < 250.0 && i == 0)
            || ((250.0..4000.0).contains(&freq) && i == 1)
            || (freq >= 4000.0 && i == 2);
        if !want {
            assert!(*v < 0.1 * main, "band {i} leaks: {e:?}");
        }
    }
    let got = match freq {
        x if x < 250.0 => f.bass,
        x if x < 4000.0 => f.mid,
        _ => f.treb,
    };
    assert!(got > 0.5, "dominant band ratio {got}");
}

#[test]
fn sine_60hz_is_bass() {
    dominant(60.0);
}
#[test]
fn sine_1k_is_mid() {
    dominant(1000.0);
}
#[test]
fn sine_8k_is_treb() {
    dominant(8000.0);
}

#[test]
fn steady_signal_normalises_to_one() {
    // Mean level over 6..10 s (single-tick bass noise is far wider than 0.8..1.25).
    let mut p = Pink::new(0x1234_5678_9abc_def1);
    let (mut sum, mut n) = ([0.0f64; 3], 0);
    run(
        48000,
        10.0,
        |_| p.next(),
        |t, _, f| {
            if t >= 6.0 {
                sum[0] += f.bass as f64;
                sum[1] += f.mid as f64;
                sum[2] += f.treb as f64;
                n += 1;
            }
        },
    );
    eprintln!("steady mean {:?}", sum.map(|s| s / n as f64));
    for s in sum {
        assert!((0.8..1.25).contains(&(s / n as f64)), "{}", s / n as f64);
    }
}

#[test]
fn loud_hit_above_one() {
    let mut p = Pink::new(42);
    let mut peak = 0.0f32;
    let mut n = 0u64;
    run(
        48000,
        10.5,
        |_| {
            let g = if n >= 480_000 { 2.0 } else { 1.0 };
            n += 1;
            p.next() * g
        },
        |t, _, f| {
            if (10.0..=10.2).contains(&t) {
                peak = peak.max(f.bass);
            }
        },
    );
    assert!(peak > 1.5, "peak {peak}");
}

#[test]
fn att_smoother_than_instant() {
    let mut p = Pink::new(7);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    run(
        48000,
        8.0,
        |t| {
            let g = if (t * 2.0).floor() as i64 % 2 == 0 {
                1.0
            } else {
                0.1
            };
            p.next() * g
        },
        |t, _, f| {
            if t > 3.0 {
                a.push(f.bass_att as f64);
                b.push(f.bass as f64);
            }
        },
    );
    let var = |v: &Vec<f64>| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64
    };
    assert!(var(&a) < var(&b), "{} vs {}", var(&a), var(&b));
}

fn all_finite(f: &AudioFrame) -> bool {
    f.wave.iter().flatten().all(|v| v.is_finite())
        && f.spectrum.iter().all(|v| v.is_finite())
        && [
            f.bass,
            f.mid,
            f.treb,
            f.bass_att,
            f.mid_att,
            f.treb_att,
            f.beat,
            f.beat_strength,
            f.silent_for,
        ]
        .iter()
        .all(|v| v.is_finite())
}

#[test]
fn silence_no_nan() {
    let (_, f) = run(48000, 5.0, |_| 0.0, |_, _, f| assert!(all_finite(f)));
    assert!(all_finite(&f));
}

#[test]
fn silence_before_first_sound_is_zero() {
    let mut n = 0u64;
    run(
        48000,
        2.0,
        |_| {
            n += 1;
            if n > 48000 {
                0.3 * ((n as f32) * 0.05).sin()
            } else {
                0.0
            }
        },
        |t, _, f| {
            if t < 0.99 {
                assert_eq!(
                    [f.bass, f.mid, f.treb, f.bass_att, f.mid_att, f.treb_att],
                    [0.0; 6]
                );
            }
        },
    );
}

#[test]
fn rates_consistent() {
    // A single tick of noise is itself random (the bass band has only ~10-20
    // bins), so compare the mean level over the last 4 s of a 10 s run.
    // Pink-noise bass varies slowly and the band holds few bins, so average
    // over several seeds as well.
    let measure_seed = |rate: u32, seed: u64| {
        let mut p = Pink::new(seed);
        let (mut sum, mut n) = ([0.0f64; 3], 0usize);
        let (_, f) = run_from(
            rate,
            10.0,
            0.5,
            |_| p.next(),
            |t, _, f| {
                if t >= 6.0 {
                    sum[0] += f.bass as f64;
                    sum[1] += f.mid as f64;
                    sum[2] += f.treb as f64;
                    n += 1;
                }
            },
        );
        assert!(all_finite(&f));
        sum.map(|s| s / n as f64)
    };
    let measure = |rate: u32| {
        let mut acc = [0.0f64; 3];
        for seed in 1..=8u64 {
            let m = measure_seed(rate, seed * 7919);
            for (a, v) in acc.iter_mut().zip(m) {
                *a += v / 8.0;
            }
        }
        eprintln!("rate {rate}: mean {acc:?}");
        acc
    };
    let base = measure(48000);
    for rate in [22050u32, 44100, 192000] {
        let m = measure(rate);
        for (i, name) in ["bass", "mid", "treb"].iter().enumerate() {
            assert!(m[i] > 0.0, "{rate} {name}");
            assert!(
                (m[i] / base[i] - 1.0).abs() < 0.15,
                "{rate} {name}: {} vs {}",
                m[i],
                base[i]
            );
        }
    }
}

/// Instant pink-noise bass legitimately swings 0.45..2.7 (few FFT bins), so
/// the 0.5..2.0 check applies to `_att` for all bands and to instant mid and
/// treb; instant bass only has to stay clear of the old 10-20x inflation.
fn check_band_window(from: f64, to: f64, rows: &[(f64, [f32; 6])]) -> f32 {
    let mut max = 0.0f32;
    let mut seen = 0;
    for (t, b) in rows {
        if *t >= from && *t <= to {
            seen += 1;
            for (i, v) in b.iter().enumerate() {
                // `_att` (tau 0.4 s) is still rising from 0 just after onset.
                let lo = if i >= 3 && *t < from + 0.75 { 0.0 } else { 0.5 };
                let ok = if i == 0 {
                    (0.0..4.0).contains(v)
                } else {
                    (lo..2.0).contains(v)
                };
                assert!(ok, "t={t}: {b:?}");
                max = max.max(*v);
            }
        }
    }
    assert!(seen > 10);
    max
}

#[test]
fn onset_after_silence_not_inflated() {
    let mut p = Pink::new(5);
    let mut rows = Vec::new();
    run(
        48000,
        6.0,
        |t| if t < 1.0 { 0.0 } else { p.next() },
        |t, _, f| {
            rows.push((
                t,
                [f.bass, f.mid, f.treb, f.bass_att, f.mid_att, f.treb_att],
            ))
        },
    );
    let max = check_band_window(1.25, 6.0, &rows);
    eprintln!("onset max {max}");
}

#[test]
fn resume_after_long_gap_not_inflated() {
    let mut p = Pink::new(6);
    let mut rows = Vec::new();
    run(
        48000,
        28.0,
        |t| {
            if (5.0..25.0).contains(&t) {
                0.0
            } else {
                p.next()
            }
        },
        |t, _, f| {
            rows.push((
                t,
                [f.bass, f.mid, f.treb, f.bass_att, f.mid_att, f.treb_att],
            ))
        },
    );
    let max = check_band_window(25.25, 28.0, &rows);
    eprintln!("resume max {max}");
}

#[test]
fn empty_band_reads_one() {
    let mut p = Pink::new(8);
    let (an, f) = run(
        8000,
        2.0,
        |_| p.next(),
        |_, _, f| {
            assert_eq!((f.treb, f.treb_att), (1.0, 1.0));
        },
    );
    assert_eq!((f.treb, f.treb_att), (1.0, 1.0));
    assert_eq!(an.raw_energy()[2], 0.0);
    assert!(f.mid > 0.0 && f.bass > 0.0);
}

#[test]
fn av_offset_shifts_wave_window() {
    let rate = 48000u32;
    let mut hist = AudioHistory::new(rate, 2.0);
    let frames: Vec<Frame> = (0..rate as usize)
        .map(|i| [i as f32 / rate as f32, -(i as f32) / rate as f32])
        .collect();
    let now = (rate - 1) as f64 / rate as f64;
    hist.push(&frames, now);
    let mut an = Analyzer::new(rate);
    let a = *an.tick(&hist, now, 0.0, 1.0 / 60.0);
    let b = *an.tick(&hist, now, 0.1, 1.0 / 60.0);
    assert!((a.wave[0][575] - b.wave[0][575] - 0.1).abs() < 1e-4);
    assert!((a.wave[1][575] - b.wave[1][575] + 0.1).abs() < 1e-4);
    assert!(b.wave[0][0] < b.wave[0][575]);
}
