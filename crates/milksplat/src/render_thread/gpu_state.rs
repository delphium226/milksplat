//! Everything tied to one GPU device, kept together so that a device loss
//! replaces all of it in one place.
//!
//! After a loss, wgpu-core panics if an object of the old device meets the
//! new one, and `Renderer` methods take the caller's `Gpu`. So the `Gpu`
//! lives only here, next to the surface, renderer, loader and timer built
//! on it, and the render thread drops a lost `GpuState` as a whole and
//! builds a new one; no other code keeps a `Gpu`
//! (or anything created from it) across frames.
//!
//! **Recovery path:** the wgpu instance outlives the device and is reused.
//! Nothing of the old device may be alive when the new one is requested:
//! D3D12 devices are per-adapter singletons, and the driver hands back the
//! same, removed, device while references to it remain; a flip-model
//! swapchain also holds the window ("window is in use"). So, in order:
//! `Director::on_device_lost`, `Director::release_gpu` (the director keeps
//! only shader sources), drop the whole old `GpuState` (loader joined,
//! renderer, surface and swapchain, device), ask the main thread for a new
//! surface from the same instance (winit only gives out the window handle
//! on the main thread), [`GpuState::new`] (the renderer is built fresh with
//! `Renderer::new`, as `Renderer::recreate` would need the old renderer
//! alive), then `Director::on_renderer_recreated`. A failed attempt is
//! retried a second later, still with nothing old alive.

use ms_engine::Loader;
use ms_render::{FeedbackFormat, Gpu, RenderConfig, Renderer, TextureResolver};

use super::gpu_timer::GpuTimer;

/// The GPU device and everything built on it. Fields drop in declaration
/// order: loader and timer, the renderer, the surface, then the device.
pub(super) struct GpuState {
    pub loader: Loader,
    pub timer: Option<GpuTimer>,
    pub renderer: Renderer,
    pub surface: wgpu::Surface<'static>,
    pub gpu: Gpu,
    /// The swapchain (and composite target) format.
    pub format: wgpu::TextureFormat,
    present_mode: wgpu::PresentMode,
    alpha_mode: wgpu::CompositeAlphaMode,
}

/// The surface's present mode, alpha mode and format for `gpu`'s adapter.
struct SurfaceChoice {
    format: wgpu::TextureFormat,
    present_mode: wgpu::PresentMode,
    alpha_mode: wgpu::CompositeAlphaMode,
}

impl GpuState {
    /// Opens a device for `surface` (created from `instance`) and builds
    /// the renderer (at `internal_size`), the loader and the frame timer.
    pub(super) fn new(
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        resolver: &TextureResolver,
        internal_size: [u32; 2],
    ) -> Result<GpuState, String> {
        let gpu =
            Gpu::with_instance(instance.clone(), Some(&surface)).map_err(|e| e.to_string())?;
        let choice = choose(&surface, &gpu)?;
        let renderer = Renderer::new(
            &gpu,
            RenderConfig {
                internal_size,
                format: FeedbackFormat::Rgba8,
            },
        );
        let loader = Loader::spawn(renderer.shared(), resolver.clone(), choice.format);
        let timer = GpuTimer::new(&gpu);
        Ok(GpuState {
            loader,
            timer,
            renderer,
            surface,
            gpu,
            format: choice.format,
            present_mode: choice.present_mode,
            alpha_mode: choice.alpha_mode,
        })
    }

    /// Configures the surface for `size` (both sides non-zero).
    pub(super) fn configure(&self, size: [u32; 2]) {
        self.surface.configure(
            &self.gpu.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: size[0],
                height: size[1],
                present_mode: self.present_mode,
                desired_maximum_frame_latency: 1,
                alpha_mode: self.alpha_mode,
                view_formats: Vec::new(),
            },
        );
    }

    /// Whether frames are presented with Mailbox (which does not pace them).
    pub(super) fn mailbox(&self) -> bool {
        self.present_mode == wgpu::PresentMode::Mailbox
    }

    /// Whether the device-lost callback has fired.
    pub(super) fn is_lost(&self) -> bool {
        self.gpu.lost.load(std::sync::atomic::Ordering::SeqCst)
    }
}

fn choose(surface: &wgpu::Surface<'_>, gpu: &Gpu) -> Result<SurfaceChoice, String> {
    let caps = surface.get_capabilities(&gpu.adapter);
    let format = pick_format(&caps.formats)
        .ok_or_else(|| "the window surface supports no formats on this GPU".to_string())?;
    let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
        wgpu::PresentMode::Mailbox
    } else {
        wgpu::PresentMode::Fifo
    };
    let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
        wgpu::CompositeAlphaMode::Opaque
    } else {
        caps.alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto)
    };
    tracing::info!(?format, ?present_mode, ?alpha_mode, "surface");
    if format.is_srgb() {
        tracing::warn!(
            "the surface only offers sRGB formats ({format:?}); colours will look washed out"
        );
    }
    Ok(SurfaceChoice {
        format,
        present_mode,
        alpha_mode,
    })
}

/// The swapchain format: a non-sRGB 8-bit format (MilkDrop-style colour
/// math is done in gamma space, so the values are written as they are),
/// `Bgra8Unorm` first; else any non-sRGB format; else the first one.
pub fn pick_format(formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as F;
    [F::Bgra8Unorm, F::Rgba8Unorm]
        .into_iter()
        .find(|f| formats.contains(f))
        .or_else(|| formats.iter().copied().find(|f| !f.is_srgb()))
        .or_else(|| formats.first().copied())
}
