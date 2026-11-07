use ms_analysis::{AudioHistory, Frame};

const R: u32 = 48_000;

fn ramp(n: usize) -> Vec<Frame> {
    (0..n)
        .map(|i| [i as f32 + 1.0, -(i as f32) - 1.0])
        .collect()
}

#[test]
fn window_reads_latest() {
    let mut h = AudioHistory::new(R, 2.0);
    let f = ramp(1000);
    h.push(&f, 1.0);
    let mut out = [[0.0; 2]; 100];
    h.window(1.0, &mut out);
    assert_eq!(&out[..], &f[900..]);
}

#[test]
fn window_with_offset() {
    let mut h = AudioHistory::new(R, 2.0);
    let f = ramp(1000);
    h.push(&f, 1.0);
    let mut out = [[0.0; 2]; 100];
    h.window(1.0 - 0.001, &mut out);
    assert_eq!(&out[..], &f[852..952]);
}

#[test]
fn gap_filled() {
    let mut h = AudioHistory::new(R, 2.0);
    h.push(&ramp(1000), 1.0);
    h.fill_silence_until(1.2);
    let mut out = [[1.0; 2]; 256];
    h.window(1.2, &mut out);
    assert!(out.iter().all(|f| *f == [0.0, 0.0]));
}

#[test]
fn no_fill_under_50ms() {
    let mut h = AudioHistory::new(R, 2.0);
    let f = ramp(1000);
    h.push(&f, 1.0);
    h.fill_silence_until(1.04);
    let mut out = [[0.0; 2]; 100];
    h.window(1.0, &mut out);
    assert_eq!(&out[..], &f[900..]);
}

#[test]
fn silent_for() {
    let mut h = AudioHistory::new(R, 2.0);
    assert_eq!(h.silent_for(5.0), 0.0);
    h.push(&vec![[0.0, 0.0]; 480], 1.0);
    assert!((h.silent_for(1.5) - 0.5).abs() < 1e-9);
    h.push(&[[0.5, 0.0]], 1.01);
    assert!(h.silent_for(1.01).abs() < 1e-9);
    h.push(&vec![[0.0, 0.0]; 480], 1.02);
    assert!((h.silent_for(1.02) - 0.01).abs() < 1e-3);
    h.fill_silence_until(2.0);
    assert!((h.silent_for(2.0) - 0.99).abs() < 1e-3);
}

#[test]
fn wraps() {
    let mut h = AudioHistory::new(R, 1.0);
    let f = ramp(3 * R as usize);
    h.push(&f, 3.0);
    let mut out = vec![[0.0; 2]; 2 * R as usize];
    h.window(3.0, &mut out);
    let n = R as usize;
    assert!(out[..n].iter().all(|f| *f == [0.0, 0.0]));
    assert_eq!(&out[n..], &f[2 * n..]);
}

#[test]
fn rate_change_resets() {
    let mut h = AudioHistory::new(R, 1.0);
    h.push(&ramp(1000), 1.0);
    h.reset(44_100);
    assert_eq!(h.sample_rate(), 44_100);
    let mut out = [[1.0; 2]; 64];
    h.window(1.0, &mut out);
    assert!(out.iter().all(|f| *f == [0.0, 0.0]));
}

#[test]
fn zero_gap_padding_between_packets() {
    let mut h = AudioHistory::new(R, 2.0);
    h.push(&[[1.0, 1.0]; 10], 1.0);
    // second packet's first frame is ~0.1 s later than expected
    h.push(&[[2.0, 2.0]; 10], 1.1);
    let mut out = vec![[9.0; 2]; 4800 + 20];
    h.window(1.1, &mut out);
    // newest 10 are 2.0; the gap before them is zero; the older 1.0s precede it
    assert!(out[out.len() - 10..].iter().all(|f| *f == [2.0, 2.0]));
    let gap = &out[out.len() - 10 - 4700..out.len() - 10];
    assert!(gap.iter().all(|f| *f == [0.0, 0.0]));
    let ones = out.iter().filter(|f| **f == [1.0, 1.0]).count();
    assert_eq!(ones, 10);
}

#[test]
fn small_jitter_just_appends() {
    let mut h = AudioHistory::new(R, 2.0);
    h.push(&[[1.0, 1.0]; 10], 1.0);
    h.push(&[[2.0, 2.0]; 10], 1.0 + 10.0 / R as f64 + 0.01);
    let mut out = [[0.0; 2]; 20];
    h.window(1.0 + 10.0 / R as f64 + 0.01, &mut out);
    assert_eq!(out[9], [1.0, 1.0]);
    assert_eq!(out[10], [2.0, 2.0]);
}

#[test]
fn sample_rate_zero_does_not_panic() {
    let mut h = AudioHistory::new(0, 0.0);
    assert_eq!(h.sample_rate(), 1);
    h.push(&[[1.0, 1.0]; 3], 1.0);
    h.fill_silence_until(5.0);
    let mut out = [[0.0; 2]; 4];
    h.window(5.0, &mut out);
    h.reset(0);
    assert_eq!(h.sample_rate(), 1);
}

#[test]
fn nan_times_ignored() {
    let mut h = AudioHistory::new(R, 1.0);
    h.push(&[[1.0, 1.0]; 3], f64::NAN);
    h.fill_silence_until(f64::INFINITY);
    let mut out = [[0.0; 2]; 4];
    h.window(f64::NAN, &mut out);
    assert!(out.iter().all(|f| *f == [0.0, 0.0]));
    assert_eq!(h.silent_for(f64::NAN), 0.0);
}

#[test]
fn window_longer_than_capacity_zero_padded_front() {
    let mut h = AudioHistory::new(1000, 1.0);
    let f = ramp(500);
    h.push(&f, 0.5);
    let mut out = vec![[7.0; 2]; 3000];
    h.window(0.5, &mut out);
    assert!(out[..2500].iter().all(|f| *f == [0.0, 0.0]));
    assert_eq!(&out[2500..], &f[..]);
}
