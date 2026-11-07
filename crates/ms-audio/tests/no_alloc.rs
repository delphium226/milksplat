use ms_analysis::{AudioHistory, Frame};
use ms_audio::{downmix, layout_for, ring};
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
fn downmix_push_drain_do_not_allocate() {
    let layout = layout_for(6);
    let input = vec![0.25f32; 6 * 1000];
    let mut scratch: Vec<Frame> = vec![[0.0; 2]; 1000];
    let (mut p, mut c) = ring(30_000);
    let mut h = AudioHistory::new(48_000, 1.0);

    let big = vec![[0.1f32; 2]; 40_000];

    let before = ALLOCS.load(Ordering::SeqCst);
    for i in 0..20 {
        downmix(&input, &layout, &mut scratch);
        p.push(&scratch, i as f64 * 0.02);
    }
    // Oversize packet, partially dropped.
    p.push(&big, 5.0);
    c.drain_into(&mut h);
    let after = ALLOCS.load(Ordering::SeqCst);
    assert_eq!(after - before, 0, "hot paths allocated");
}
