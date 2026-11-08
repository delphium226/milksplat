use ms_expr::Rng;
use ms_preset::*;
use std::collections::HashMap;
use std::path::PathBuf;

fn lib(n: usize) -> Library {
    Library {
        entries: (0..n)
            .map(|i| Entry {
                id: format!("p{i:02}.splat"),
                path: PathBuf::from(format!("p{i:02}.splat")),
            })
            .collect(),
    }
}

#[test]
fn empty_library_none() {
    let mut s = Selector::new(1);
    assert_eq!(s.next(&lib(0), &Prefs::default()), None);
}

#[test]
fn single_preset_repeats() {
    let mut s = Selector::new(1);
    let l = lib(1);
    for _ in 0..5 {
        assert_eq!(s.next(&l, &Prefs::default()).as_deref(), Some("p00.splat"));
    }
}

#[test]
fn all_banned_none() {
    let l = lib(3);
    let mut prefs = Prefs::default();
    prefs.bans.insert("p00.splat".into());
    prefs.bans.insert("p01.splat".into());
    prefs.blocklist.insert("p02.splat".into());
    assert_eq!(Selector::new(1).next(&l, &prefs), None);
}

#[test]
fn no_recent_repeat() {
    let l = lib(10);
    let mut s = Selector::new(7);
    let picks: Vec<String> = (0..1000)
        .map(|_| s.next(&l, &Prefs::default()).unwrap())
        .collect();
    for w in picks.windows(4) {
        for i in 0..4 {
            for j in i + 1..4 {
                assert_ne!(w[i], w[j]);
            }
        }
    }
}

#[test]
fn favourites_twice_as_often() {
    let l = lib(10);
    let mut prefs = Prefs::default();
    prefs.favourites.insert("p03.splat".into());
    let mut s = Selector::new(99);
    let mut counts: HashMap<String, f64> = HashMap::new();
    for _ in 0..20_000 {
        *counts.entry(s.next(&l, &prefs).unwrap()).or_default() += 1.0;
    }
    let fav = counts["p03.splat"];
    let others: f64 = counts
        .iter()
        .filter(|(k, _)| *k != "p03.splat")
        .map(|(_, v)| v)
        .sum::<f64>()
        / 9.0;
    let ratio = fav / others;
    assert!((1.5..=2.5).contains(&ratio), "ratio {ratio}");
}

#[test]
fn prev_walks_history() {
    let l = lib(10);
    let mut s = Selector::new(3);
    let a = s.next(&l, &Prefs::default()).unwrap();
    let b = s.next(&l, &Prefs::default()).unwrap();
    let c = s.next(&l, &Prefs::default()).unwrap();
    assert_eq!(s.prev().as_deref(), Some(b.as_str()));
    assert_eq!(s.prev().as_deref(), Some(a.as_str()));
    assert_eq!(s.prev(), None);
    let _ = c;
    // next after prev picks fresh and appends
    assert!(s.next(&l, &Prefs::default()).is_some());
}

#[test]
fn push_manual_then_prev() {
    let l = lib(10);
    let mut s = Selector::new(3);
    let a = s.next(&l, &Prefs::default()).unwrap();
    s.push_manual("p09.splat".into());
    assert_eq!(s.prev().as_deref(), Some(a.as_str()));
}

#[test]
fn history_capped_50() {
    let l = lib(100);
    let mut s = Selector::new(5);
    for _ in 0..200 {
        s.next(&l, &Prefs::default());
    }
    let mut n = 0;
    while s.prev().is_some() {
        n += 1;
    }
    assert_eq!(n, 49); // 50 entries; the oldest has no predecessor
}

#[test]
fn duration_range() {
    let mut rng = Rng::seeded(4);
    for _ in 0..10_000 {
        let d = next_duration(20.0, &mut rng);
        assert!((15.0..=25.0).contains(&d));
    }
}
