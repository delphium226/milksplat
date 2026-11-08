//! GPU frame time from timestamp queries, for the quality governor (§6.6).
//!
//! [`begin`](GpuTimer::begin) and [`end`](GpuTimer::end) each submit an
//! empty compute pass that writes a timestamp, so the span covers every
//! submission in between (the ticks, the composite and the text). The two
//! values are resolved and copied into a small readback buffer that is
//! mapped asynchronously; [`collect`](GpuTimer::collect) returns the
//! frame times whose results have arrived (usually one or two frames later)
//! and never blocks. A few slots rotate, and a frame is simply not timed
//! when they are all still in flight.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use ms_render::Gpu;

const SLOTS: u32 = 4;
const SLOT_STRIDE: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;

const FREE: u8 = 0;
const IN_FLIGHT: u8 = 1;
const MAPPED: u8 = 2;
const FAILED: u8 = 3;

struct Slot {
    readback: wgpu::Buffer,
    state: Arc<AtomicU8>,
}

/// Measures GPU time per frame (see the module docs).
pub(super) struct GpuTimer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    slots: Vec<Slot>,
    /// The slot the current frame writes to, between `begin` and `end`.
    open: Option<u32>,
    next: u32,
    /// Nanoseconds per timestamp tick.
    period_ns: f64,
}

impl GpuTimer {
    /// A timer for `gpu`, or `None` if it has no timestamp queries.
    pub(super) fn new(gpu: &Gpu) -> Option<GpuTimer> {
        if !gpu
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        let period_ns = f64::from(gpu.queue.get_timestamp_period());
        if !period_ns.is_finite() || period_ns <= 0.0 {
            return None;
        }
        let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame timer"),
            ty: wgpu::QueryType::Timestamp,
            count: SLOTS * 2,
        });
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame timer resolve"),
            size: u64::from(SLOTS) * SLOT_STRIDE,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = (0..SLOTS)
            .map(|_| Slot {
                readback: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("frame timer readback"),
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(FREE)),
            })
            .collect();
        Some(GpuTimer {
            queries,
            resolve,
            slots,
            open: None,
            next: 0,
            period_ns,
        })
    }

    /// Writes the start timestamp, if a slot is free.
    pub(super) fn begin(&mut self, gpu: &Gpu) {
        self.open = None;
        let slot = self.next;
        if self.slots[slot as usize].state.load(Ordering::Acquire) != FREE {
            return;
        }
        self.write_timestamp(gpu, slot * 2, None);
        self.open = Some(slot);
    }

    /// Writes the end timestamp of the slot `begin` opened and starts
    /// reading the pair back.
    pub(super) fn end(&mut self, gpu: &Gpu) {
        let Some(slot) = self.open.take() else {
            return;
        };
        let s = &self.slots[slot as usize];
        let offset = u64::from(slot) * SLOT_STRIDE;
        self.write_timestamp(
            gpu,
            slot * 2 + 1,
            Some((slot * 2..slot * 2 + 2, offset, &s.readback)),
        );
        s.state.store(IN_FLIGHT, Ordering::Release);
        let state = Arc::clone(&s.state);
        s.readback.map_async(wgpu::MapMode::Read, .., move |r| {
            state.store(if r.is_ok() { MAPPED } else { FAILED }, Ordering::Release);
        });
        self.next = (slot + 1) % SLOTS;
    }

    /// Frame times (ms) whose timestamps have arrived since the last call.
    /// Does not block.
    pub(super) fn collect(&mut self, gpu: &Gpu, out: &mut Vec<f64>) {
        poll_nonblocking(gpu);
        for s in &self.slots {
            match s.state.load(Ordering::Acquire) {
                MAPPED => {
                    if let Ok(view) = s.readback.get_mapped_range(..) {
                        let read = |i: usize| {
                            let mut b = [0u8; 8];
                            b.copy_from_slice(&view[i * 8..i * 8 + 8]);
                            u64::from_le_bytes(b)
                        };
                        let (t0, t1) = (read(0), read(1));
                        drop(view);
                        if t1 > t0 {
                            let ms = (t1 - t0) as f64 * self.period_ns / 1.0e6;
                            if ms.is_finite() && ms < 10_000.0 {
                                out.push(ms);
                            }
                        }
                    }
                    s.readback.unmap();
                    s.state.store(FREE, Ordering::Release);
                }
                FAILED => s.state.store(FREE, Ordering::Release),
                _ => {}
            }
        }
    }

    /// Submits an empty compute pass writing timestamp `index`, then
    /// optionally resolves `range` into the slot at `offset` and copies it
    /// to `readback`.
    fn write_timestamp(
        &self,
        gpu: &Gpu,
        index: u32,
        resolve: Option<(std::ops::Range<u32>, u64, &wgpu::Buffer)>,
    ) {
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame timer"),
            });
        drop(enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("frame timer"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.queries,
                beginning_of_pass_write_index: Some(index),
                end_of_pass_write_index: None,
            }),
        }));
        if let Some((range, offset, readback)) = resolve {
            enc.resolve_query_set(&self.queries, range, &self.resolve, offset);
            enc.copy_buffer_to_buffer(&self.resolve, offset, readback, 0, 16);
        }
        gpu.queue.submit([enc.finish()]);
    }
}

/// Lets wgpu run finished-work callbacks (buffer maps). wgpu 30 panics in
/// `Device::poll` on a lost device, so this checks the flag and catches a
/// panic; the render loop notices the loss itself.
fn poll_nonblocking(gpu: &Gpu) {
    if gpu.lost.load(Ordering::SeqCst) {
        return;
    }
    let polled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.device.poll(wgpu::PollType::Poll)
    }));
    if !matches!(polled, Ok(Ok(_))) {
        tracing::debug!("non-blocking device poll failed");
    }
}
