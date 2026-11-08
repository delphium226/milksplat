//! Device-loss recovery (§6.7) on the render thread; see `gpu_state` for
//! why the old state is dropped before the new device is opened.

use std::sync::atomic::Ordering;

use super::gpu_state::GpuState;
use super::{RenderThread, GPU_RETRY_S};

impl RenderThread {
    /// `--debug-lose-device`: destroys the device once, after the given
    /// seconds of rendering, and raises the lost flag at once (the callback
    /// may come later) so nothing renders on the destroyed device.
    pub(super) fn maybe_lose_device(&mut self, wall: f64) {
        let (Some(after), Some(start), Some(gs)) =
            (self.lose_device_after, self.first_frame, &self.gs)
        else {
            return;
        };
        if wall - start >= after {
            self.lose_device_after = None;
            tracing::warn!("--debug-lose-device: destroying the GPU device");
            gs.gpu.device.destroy();
            gs.gpu.lost.store(true, Ordering::SeqCst);
            self.synthetic_loss = true;
        }
    }

    /// Device loss (§6.7): blame the preset if it just started, release
    /// every GPU object (the director's shader sets, then the whole
    /// `GpuState`), then build everything on a new device (retrying every
    /// second, with nothing of the old device alive).
    pub(super) fn recover(&mut self, wall: f64) {
        if !self.loss_handled {
            self.loss_handled = true;
            tracing::warn!("the GPU device was lost; recreating it");
            if std::mem::take(&mut self.synthetic_loss) {
                // `--debug-lose-device`: no preset is to blame.
                tracing::info!("the loss was forced; no preset is blocklisted");
            } else if let Some(id) = self.director.on_device_lost(self.tick_time) {
                // Task 24 persists the blocklist.
                self.cfg.blocklist.insert(id);
            }
            // The director keeps only shader sources from here on.
            self.director.release_gpu();
        }
        // Drop the whole old state (see `gpu_state`): joins the loader,
        // releases the swapchain, every renderer resource and the device.
        if let Some(old) = self.gs.take() {
            drop(old);
        }
        if wall < self.gpu_retry_at {
            self.idle_wait();
            return;
        }
        // A new surface for the window comes from the main thread (winit
        // only gives out the window handle there), created after the old
        // one was dropped above.
        let Some(surface) = self.new_surface.take() else {
            if wall >= self.surface_request_at {
                (self.request_surface)();
                self.surface_request_at = wall + GPU_RETRY_S;
            }
            self.idle_wait();
            return;
        };
        match GpuState::new(&self.instance, surface, &self.resolver, self.internal) {
            Ok(gs) => {
                self.director.on_renderer_recreated(
                    &gs.renderer.shared(),
                    &self.resolver,
                    gs.format,
                );
                self.gs = Some(gs);
                self.loss_handled = false;
                self.surface_dirty = true;
                self.last_wall = None;
                self.next_frame_due = None;
                self.stats.reset();
                tracing::info!("GPU device recreated");
            }
            Err(e) => {
                tracing::error!(
                    "cannot recreate the GPU device ({e}); retrying in {GPU_RETRY_S} s"
                );
                self.gpu_retry_at = wall + GPU_RETRY_S;
                // The surface went with the failed attempt; ask again.
                self.surface_request_at = 0.0;
            }
        }
    }
}
