use ms_analysis::{AudioHistory, Frame};
use rtrb::{Consumer, Producer, RingBuffer};

const STAMP_CAPACITY: usize = 256;
const SCRATCH_FRAMES: usize = 4096;

/// Describes one pushed packet.
#[derive(Clone, Copy)]
struct Stamp {
    /// Time of the packet's last frame (including any dropped tail).
    time: f64,
    /// Frames actually written to the frame ring.
    frames: u32,
    /// Frames cut from the packet's tail because the ring was full.
    dropped_tail: u32,
}

/// Capture-side half (real-time safe: never allocates, locks or blocks).
pub struct RingProducer {
    frames: Producer<Frame>,
    stamps: Producer<Stamp>,
}

/// Render-side half.
pub struct RingConsumer {
    frames: Consumer<Frame>,
    stamps: Consumer<Stamp>,
    scratch: Box<[Frame]>,
}

/// Creates a frame ring holding `capacity_frames` plus its packet-stamp ring.
pub fn ring(capacity_frames: usize) -> (RingProducer, RingConsumer) {
    let (fp, fc) = RingBuffer::new(capacity_frames);
    let (sp, sc) = RingBuffer::new(STAMP_CAPACITY);
    (
        RingProducer {
            frames: fp,
            stamps: sp,
        },
        RingConsumer {
            frames: fc,
            stamps: sc,
            scratch: vec![[0.0; 2]; SCRATCH_FRAMES].into_boxed_slice(),
        },
    )
}

impl RingProducer {
    /// Queues `frames` whose last frame is at `time`. Excess frames (the packet
    /// tail) are dropped when the ring is full; never blocks.
    pub fn push(&mut self, frames: &[Frame], time: f64) {
        if self.stamps.is_full() {
            return;
        }
        let n = frames
            .len()
            .min(self.frames.slots())
            .min(u32::MAX as usize);
        if n == 0 {
            return;
        }
        let Ok(chunk) = self.frames.write_chunk_uninit(n) else {
            return;
        };
        chunk.fill_from_iter(frames[..n].iter().copied());
        let dropped_tail = u32::try_from(frames.len() - n).unwrap_or(u32::MAX);
        // Cannot fail: checked above and we are the only producer.
        let _ = self.stamps.push(Stamp {
            time,
            frames: n as u32,
            dropped_tail,
        });
    }
}

impl RingConsumer {
    /// Moves all queued packets into `hist`, in order. Does not allocate.
    pub fn drain_into(&mut self, hist: &mut AudioHistory) {
        let rate = hist.sample_rate() as f64;
        while let Ok(stamp) = self.stamps.pop() {
            let total = stamp.frames as usize;
            let last_written = stamp.time - stamp.dropped_tail as f64 / rate;
            let mut done = 0usize;
            while done < total {
                let n = (total - done).min(self.scratch.len());
                let Ok(chunk) = self.frames.read_chunk(n) else {
                    return;
                };
                let (a, b) = chunk.as_slices();
                self.scratch[..a.len()].copy_from_slice(a);
                self.scratch[a.len()..n].copy_from_slice(b);
                chunk.commit_all();
                done += n;
                let t = last_written - (total - done) as f64 / rate;
                hist.push(&self.scratch[..n], t);
            }
        }
    }
}
