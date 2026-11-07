use ms_analysis::{AudioHistory, Frame};
use ms_audio::{backoff_delay, downmix, layout_for, ring, Channel};
use std::time::Duration;

const K: f32 = 0.70710677;

fn run(layout: &[Channel], interleaved: &[f32], n: usize) -> Vec<Frame> {
    let mut out = vec![[9.0f32; 2]; n];
    downmix(interleaved, layout, &mut out);
    out
}

fn impulse(layout: &[Channel], ch: usize) -> Frame {
    let mut v = vec![0.0; layout.len()];
    v[ch] = 1.0;
    run(layout, &v, 1)[0]
}

#[test]
fn layouts() {
    use Channel::*;
    assert_eq!(layout_for(1), vec![FC]);
    assert_eq!(layout_for(2), vec![FL, FR]);
    assert_eq!(layout_for(6), vec![FL, FR, FC, LFE, SL, SR]);
    assert_eq!(layout_for(8), vec![FL, FR, FC, LFE, BL, BR, SL, SR]);
    assert_eq!(layout_for(3), vec![FL, FR, Other]);
}

#[test]
fn stereo_passthrough() {
    let l = layout_for(2);
    assert_eq!(run(&l, &[0.1, 0.2, 0.3, 0.4], 2), vec![[0.1, 0.2], [0.3, 0.4]]);
}

#[test]
fn mono_to_both() {
    let l = layout_for(1);
    assert_eq!(run(&l, &[0.5, -0.25], 2), vec![[0.5, 0.5], [-0.25, -0.25]]);
}

#[test]
fn five_one() {
    let l = layout_for(6);
    let exp = [[1.0, 0.0], [0.0, 1.0], [K, K], [K, K], [K, 0.0], [0.0, K]];
    for (i, e) in exp.iter().enumerate() {
        assert_eq!(impulse(&l, i), *e, "channel {i}");
    }
}

#[test]
fn seven_one() {
    let l = layout_for(8);
    let exp = [
        [1.0, 0.0],
        [0.0, 1.0],
        [K, K],
        [K, K],
        [K, 0.0],
        [0.0, K],
        [K, 0.0],
        [0.0, K],
    ];
    for (i, e) in exp.iter().enumerate() {
        assert_eq!(impulse(&l, i), *e, "channel {i}");
    }
}

#[test]
fn three_channels_third_ignored() {
    let l = layout_for(3);
    assert_eq!(run(&l, &[0.1, 0.2, 5.0], 1), vec![[0.1, 0.2]]);
}

#[test]
fn partial_frames_and_short_out() {
    let l = layout_for(2);
    // 2.5 frames of input: trailing half frame ignored, rest of out untouched.
    let out = run(&l, &[1.0, 2.0, 3.0, 4.0, 5.0], 4);
    assert_eq!(out, vec![[1.0, 2.0], [3.0, 4.0], [9.0, 9.0], [9.0, 9.0]]);
    // out shorter than input.
    assert_eq!(run(&l, &[1.0, 2.0, 3.0, 4.0], 1), vec![[1.0, 2.0]]);
    // empty layout writes nothing.
    assert_eq!(run(&[], &[1.0, 2.0], 1), vec![[9.0, 9.0]]);
}

#[test]
fn no_clamping() {
    let l = layout_for(6);
    let out = run(&l, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1);
    assert!(out[0][0] > 1.0);
}

fn frames(start: usize, n: usize) -> Vec<Frame> {
    (start..start + n).map(|i| [i as f32, -(i as f32)]).collect()
}

#[test]
fn drain_preserves_order_and_time() {
    let rate = 1000u32;
    let (mut p, mut c) = ring(1024);
    let mut h = AudioHistory::new(rate, 1.0);
    // Contiguous packets: last-frame times 0.010, 0.020, 0.030 at 1 kHz.
    p.push(&frames(0, 10), 0.010);
    p.push(&frames(10, 10), 0.020);
    p.push(&frames(20, 10), 0.030);
    c.drain_into(&mut h);
    let mut out = vec![[0.0; 2]; 30];
    h.window(0.030, &mut out);
    assert_eq!(out, frames(0, 30));
    // Newest time equals last stamp: window ending at 0.030 aligned (above);
    // shifting by one sample must shift the data.
    let mut o2 = vec![[0.0; 2]; 1];
    h.window(0.029, &mut o2);
    assert_eq!(o2[0], [28.0, -28.0]);
    // A second drain with nothing queued is a no-op.
    c.drain_into(&mut h);
    h.window(0.030, &mut out);
    assert_eq!(out, frames(0, 30));
}

#[test]
fn full_ring_drops_not_blocks() {
    let (mut p, mut c) = ring(8);
    let mut h = AudioHistory::new(1000, 1.0);
    p.push(&frames(0, 6), 0.006);
    p.push(&frames(6, 6), 0.012); // only 2 fit
    p.push(&frames(12, 6), 0.018); // none fit
    c.drain_into(&mut h);
    let mut w = vec![[0.0; 2]; 8];
    h.window(0.008, &mut w);
    assert_eq!(w, frames(0, 8));
}

#[test]
fn oversize_packet_keeps_head_with_correct_time() {
    let (mut p, mut c) = ring(8);
    let mut h = AudioHistory::new(1000, 1.0);
    // 12 frames, last frame at t=0.100; only first 8 fit; last written = 0.096.
    p.push(&frames(0, 12), 0.100);
    c.drain_into(&mut h);
    let mut w = vec![[0.0; 2]; 8];
    h.window(0.096, &mut w);
    assert_eq!(w, frames(0, 8));
}

#[test]
fn stamp_ring_full_drops_frames_without_desync() {
    let (mut p, mut c) = ring(4096);
    let mut h = AudioHistory::new(1000, 10.0);
    // 256 stamps fit; the 257th push must be dropped entirely.
    for i in 0..257usize {
        p.push(&frames(i, 1), (i + 1) as f64 * 0.001);
    }
    c.drain_into(&mut h);
    let mut w = vec![[0.0; 2]; 256];
    h.window(0.256, &mut w);
    assert_eq!(w, frames(0, 256));
    // Rings are in sync afterwards: a new packet comes out intact.
    p.push(&frames(500, 3), 0.300);
    c.drain_into(&mut h);
    let mut w = vec![[0.0; 2]; 3];
    h.window(0.300, &mut w);
    assert_eq!(w, frames(500, 3));
}

#[test]
fn drain_chunks_large_packets_with_correct_times() {
    let n = 20_000usize;
    let (mut p, mut c) = ring(n);
    let mut h = AudioHistory::new(48_000, 1.0);
    let t_last = 1.0;
    p.push(&frames(0, n), t_last);
    c.drain_into(&mut h);
    let mut w = vec![[0.0; 2]; n];
    h.window(t_last, &mut w);
    assert_eq!(w, frames(0, n));
}

#[test]
fn backoff_sequence() {
    let got: Vec<Duration> = (0..6).map(backoff_delay).collect();
    let want: Vec<Duration> = [0.5, 1.0, 2.0, 5.0, 5.0, 5.0]
        .iter()
        .map(|s| Duration::from_secs_f64(*s))
        .collect();
    assert_eq!(got, want);
    assert_eq!(backoff_delay(u32::MAX), Duration::from_secs(5));
}
