use ms_engine::{FixedClock, Ticks};

#[test]
fn sixty_hz_from_144hz_frames() {
    let mut c = FixedClock::new(60.0);
    let run: u32 = (0..144).map(|_| c.advance(1.0 / 144.0).run).sum();
    assert!((59..=61).contains(&run), "run {run}");
}

#[test]
fn catch_up_capped() {
    let mut c = FixedClock::new(60.0);
    assert_eq!(c.advance(1.0), Ticks { run: 4, dropped: 56 });
    // dropped ticks are not carried
    assert_eq!(c.advance(0.0).run, 0);
}

#[test]
fn alpha_range() {
    let mut c = FixedClock::new(60.0);
    for i in 0..10_000 {
        c.advance(0.001 + (i % 7) as f64 * 0.003);
        let a = c.alpha();
        assert!((0.0..1.0).contains(&a), "alpha {a}");
    }
}

#[test]
fn rate_change() {
    let mut c = FixedClock::new(60.0);
    let a: u32 = (0..120).map(|_| c.advance(1.0 / 120.0).run).sum();
    c.set_rate(30.0);
    let b: u32 = (0..120).map(|_| c.advance(1.0 / 120.0).run).sum();
    assert!((59..=61).contains(&a), "a {a}");
    assert!((29..=31).contains(&b), "b {b}");
}

#[test]
fn bad_dt_ignored() {
    let mut c = FixedClock::new(60.0);
    c.advance(0.01);
    let before = c.alpha();
    for dt in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert_eq!(c.advance(dt), Ticks { run: 0, dropped: 0 });
        assert_eq!(c.alpha(), before);
    }
}

#[test]
fn set_rate_keeps_alpha_and_clamps() {
    let mut c = FixedClock::new(60.0);
    c.advance(0.01);
    let a = c.alpha();
    c.set_rate(30.0);
    assert!((c.alpha() - a).abs() < 1e-9);
    c.set_rate(f64::NAN);
    assert!((c.alpha() - a).abs() < 1e-9);
    // 60 Hz after NaN: one second yields 60 ticks (cap aside)
    let mut n = 0;
    for _ in 0..60 {
        n += c.advance(1.0 / 60.0).run;
    }
    assert!((59..=61).contains(&n));
    // rate clamps to 240 Hz: one second is 240 due ticks, 4 run
    let mut d = FixedClock::new(1e9);
    assert_eq!(d.advance(1.0), Ticks { run: 4, dropped: 236 });
    assert!(c.alpha() < 1.0);
}
