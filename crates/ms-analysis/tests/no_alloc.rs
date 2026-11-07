use ms_analysis::{Analyzer, AudioHistory};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.realloc(p, l, n)
    }
}

#[global_allocator]
static A: Counting = Counting;

#[test]
fn tick_does_not_allocate() {
    let rate = 48000u32;
    let mut hist = AudioHistory::new(rate, 2.0);
    let mut an = Analyzer::new(rate);
    let frames: Vec<[f32; 2]> = (0..rate as usize)
        .map(|i| {
            let v = (i as f32 * 0.05).sin() * 0.5;
            [v, v]
        })
        .collect();
    hist.push(&frames, 1.0);
    for _ in 0..10 {
        an.tick(&hist, 1.0, 0.0, 1.0 / 60.0);
    }
    let before = ALLOCS.load(Ordering::SeqCst);
    for _ in 0..100 {
        let f = an.tick(&hist, 1.0, 0.0, 1.0 / 60.0);
        assert!(f.bass.is_finite());
    }
    assert_eq!(ALLOCS.load(Ordering::SeqCst), before);
}
