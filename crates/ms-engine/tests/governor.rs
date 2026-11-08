use ms_engine::Governor;

const BUDGET: f64 = 16.7;

/// Feeds `frames` samples at 60 fps starting at frame index `start`,
/// returning (time, new_scale) for each change.
fn feed(g: &mut Governor, start: usize, frames: usize, ms: impl Fn(f64) -> f64) -> Vec<(f64, f32)> {
    let mut out = Vec::new();
    for i in start..start + frames {
        let t = i as f64 / 60.0;
        if let Some(s) = g.record(t, ms(t), BUDGET) {
            out.push((t, s));
        }
    }
    out
}

// NOTE: the brief says 15 ms vs 16.7 ms, but 15 < 0.9 * 16.7 = 15.03, so the
// rule "avg > 0.9 * budget" cannot fire; 16 ms is used instead.
#[test]
fn governor_steps_down() {
    let mut g = Governor::new();
    let ch = feed(&mut g, 0, 60 * 8, |_| 16.0);
    assert!(ch.len() >= 2, "{ch:?}");
    assert!(ch[0].0 >= 1.5 && ch[0].0 <= 2.2, "{ch:?}");
    assert_eq!(ch[0].1, 0.875);
    assert!(ch[1].0 - ch[0].0 >= 3.0, "{ch:?}");
    assert_eq!(ch[1].1, 0.75);
}

#[test]
fn governor_floor() {
    let mut g = Governor::new();
    let ch = feed(&mut g, 0, 60 * 60, |_| 40.0);
    let scales: Vec<f32> = ch.iter().map(|c| c.1).collect();
    assert_eq!(scales, vec![0.875, 0.75, 0.625, 0.5]);
}

#[test]
fn governor_recovers() {
    let mut g = Governor::new();
    feed(&mut g, 0, 60 * 30, |_| 40.0);
    let ch = feed(&mut g, 60 * 30, 60 * 60, |_| 5.0);
    assert_eq!(ch.last().map(|c| c.1), Some(1.0));
    assert!(ch.iter().all(|c| c.1 > 0.5));
}

#[test]
fn governor_hysteresis() {
    let mut g = Governor::new();
    let ch = feed(&mut g, 0, 60 * 60, |t| if (t as u64).is_multiple_of(2) { 15.0 } else { 5.0 });
    assert!(ch.is_empty(), "{ch:?}");
}

#[test]
fn governor_ignores_bad_samples() {
    let mut g = Governor::new();
    for i in 0..600 {
        let t = i as f64 / 60.0;
        assert_eq!(g.record(t, f64::NAN, BUDGET), None);
        assert_eq!(g.record(t, -5.0, BUDGET), None);
        assert_eq!(g.record(t, 100.0, 0.0), None);
        assert_eq!(g.record(t, 100.0, -1.0), None);
        assert_eq!(g.record(t, 100.0, f64::NAN), None);
        assert_eq!(g.record(f64::NAN, 100.0, BUDGET), None);
    }
    // bad samples never entered the window: still no decision history
    assert_eq!(g.scale(), 1.0);
}

#[test]
fn governor_ring_is_bounded() {
    let mut g = Governor::new();
    for i in 0..10_000 {
        g.record(i as f64 / 1000.0, 10.0, BUDGET);
    }
    assert!(g.sample_count() <= 512, "{}", g.sample_count());
}
