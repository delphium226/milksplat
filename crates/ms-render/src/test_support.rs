//! Test helpers shared by the GPU test binaries. Not part of the API.
//!
//! libtest runs tests on parallel threads, and several headless D3D12
//! devices alive at once in one process are not reliable (in debug builds
//! each new instance enables the D3D12 debug layer, which removes devices
//! that already exist; release builds were seen to hang). The app only ever
//! has one device at a time. Every test that creates a [`Gpu`] holds
//! [`gpu_lock`] for as long as it uses the device, which serialises the
//! GPU tests of a binary while other tests still run in parallel (cargo
//! runs test binaries one at a time).

use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::Gpu;

static GPU_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    /// How many [`GpuLock`]s this thread holds, and the mutex guard while
    /// that is non-zero. The lock is re-entrant per thread (a test may
    /// create a second device while it holds one) and is released when the
    /// thread's last [`GpuLock`] drops, whichever order they drop in.
    static HELD: RefCell<(u32, Option<MutexGuard<'static, ()>>)> = const { RefCell::new((0, None)) };
}

/// Holds the process-wide GPU test lock until dropped.
pub struct GpuLock {
    // Not Send: the lock belongs to the thread that took it.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for GpuLock {
    fn drop(&mut self) {
        let guard = HELD.with(|h| {
            let mut h = h.borrow_mut();
            h.0 = h.0.saturating_sub(1);
            if h.0 == 0 {
                h.1.take()
            } else {
                None
            }
        });
        drop(guard);
    }
}

/// Takes the process-wide GPU test lock (re-entrant on one thread; a
/// poisoned lock, from a failed test, is taken anyway).
pub fn gpu_lock() -> GpuLock {
    let first = HELD.with(|h| h.borrow().0 == 0);
    // Block outside the RefCell borrow.
    let guard = first.then(|| GPU_LOCK.lock().unwrap_or_else(PoisonError::into_inner));
    HELD.with(|h| {
        let mut h = h.borrow_mut();
        h.0 += 1;
        if guard.is_some() {
            h.1 = guard;
        }
    });
    GpuLock {
        _not_send: std::marker::PhantomData,
    }
}

/// A headless [`Gpu`] that holds [`gpu_lock`] for its lifetime (the device
/// is dropped before the lock is released).
pub struct TestGpu {
    gpu: Gpu,
    _lock: GpuLock,
}

impl Deref for TestGpu {
    type Target = Gpu;
    fn deref(&self) -> &Gpu {
        &self.gpu
    }
}

impl DerefMut for TestGpu {
    fn deref_mut(&mut self) -> &mut Gpu {
        &mut self.gpu
    }
}

/// [`Gpu::headless`] under [`gpu_lock`]; panics if there is no adapter.
pub fn test_gpu() -> TestGpu {
    let lock = gpu_lock();
    match Gpu::headless() {
        Ok(gpu) => TestGpu { gpu, _lock: lock },
        Err(e) => panic!("Gpu::headless() failed (no hardware adapter and no WARP): {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_is_reentrant_and_released() {
        {
            let a = gpu_lock();
            let b = gpu_lock();
            assert!(GPU_LOCK.try_lock().is_err());
            // Dropping the first one keeps the lock while the second lives.
            drop(a);
            assert!(GPU_LOCK.try_lock().is_err());
            drop(b);
        }
        assert!(GPU_LOCK.try_lock().is_ok());
        let t = std::thread::spawn(|| {
            let _a = gpu_lock();
        });
        assert!(t.join().is_ok());
    }
}
