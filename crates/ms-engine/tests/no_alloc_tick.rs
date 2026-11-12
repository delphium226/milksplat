//! After the first tick, ticking a preset with a batchable vertex program
//! does not allocate: the mesh, draw list and batching buffers are reused.
//! The audio goes through a turned-up [`Reactivity`] first (scaling, guard
//! and slew limit), which must not allocate either.
//!
//! A separate test binary because it installs a counting global allocator.
//! It holds a single test so no other test allocates while it measures.

use ms_analysis::AudioFrame;
use ms_engine::{Globals, PresetInstance, Reactivity, TickInput, VertexPath, REACTIVITY_MAX};
use ms_preset::{compile_preset, parse};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

fn count() {
    if COUNTING.load(Ordering::Relaxed) {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations made by ticks 2..=n+1 (the first tick is not counted).
fn allocations(toml: &str, ticks: usize) -> usize {
    let (p, _) = parse(toml.as_bytes()).unwrap();
    let mut inst = PresetInstance::new(Arc::new(compile_preset(p)), 3, [192, 144]);
    // Varying audio, so the slew limit has something to limit.
    let frames: Vec<AudioFrame> = (0..4)
        .map(|i| AudioFrame {
            bass: 0.5 + i as f32 * 0.6,
            mid: 2.0 - i as f32 * 0.4,
            ..AudioFrame::default()
        })
        .collect();
    let mut reactivity = Reactivity::new(REACTIVITY_MAX);
    let mut g = Globals::default();
    let mut tick = |i: u64| {
        let audio = reactivity.apply(&frames[i as usize % frames.len()]);
        let input = TickInput {
            time: i as f64 / 60.0,
            fps: 60.0,
            frame: i,
            progress: 0.0,
            audio,
            aspect: [1.0, 1.0],
            pixels: [1280, 720],
        };
        inst.tick(&input, &mut g).mesh.uv.len()
    };
    let n = tick(0);
    ALLOCS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let mut total = 0;
    for i in 1..=ticks as u64 {
        total += tick(i);
    }
    COUNTING.store(false, Ordering::SeqCst);
    assert_eq!(total, n * ticks);
    assert_eq!(inst.vertex_path(), VertexPath::Batched);
    ALLOCS.load(Ordering::SeqCst)
}

const CODE: &str = "[code]\n\
    frame = \"q1 = sin(time)*0.1; wave_r = bass\"\n\
    vertex = \"zoom = 1 + q1*rad + x*0.01; rot = if(y > 0.5, 0.01, ang*0.01); m = sin(x); dx = m*0.001\"\n\
    [[wave]]\nsamples = 64\npoint = \"x = sample; y = 0.5 + value1*0.2\"\n\
    [[shape]]\ninstances = 8\nframe = \"x = instance/8\"\n";

#[test]
fn no_mesh_allocation_after_first_tick() {
    // One batching chunk, run on the calling thread: no allocation at all.
    let small = allocations(&format!("[params]\nmesh = [8, 6]\n{CODE}"), 50);
    assert_eq!(small, 0, "allocations in 50 ticks of an 8x6 mesh");

    // 64x48 runs its chunks on rayon. The engine allocates nothing per tick;
    // rayon's job queues may allocate a block now and then (one per many
    // injected jobs), so allow a handful over 200 ticks. A per-tick mesh,
    // lane or scratch allocation would show up as at least 200.
    let big = allocations(&format!("[params]\nmesh = [64, 48]\n{CODE}"), 200);
    eprintln!("allocations after the first tick: 8x6 {small}, 64x48 {big}");
    assert!(big < 20, "{big} allocations in 200 ticks of a 64x48 mesh");
}
