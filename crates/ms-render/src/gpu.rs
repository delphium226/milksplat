//! Device creation (hardware or WARP software fallback) and texture readback.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use image::{Rgba, RgbaImage};

/// Environment variable that, when set to `1`, makes [`Gpu::headless`] skip
/// the hardware adapter and go straight to the software (WARP) fallback.
pub const FORCE_WARP_ENV: &str = "MS_FORCE_WARP";

/// Everything needed to render: instance, adapter, device and queue, plus a
/// flag set by the device-lost callback.
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Set to `true` by the device-lost callback.
    pub lost: Arc<AtomicBool>,
}

/// Why a [`Gpu`] could not be created.
#[derive(Debug)]
pub enum GpuError {
    /// No adapter matched the request (the message says which kind).
    NoAdapter(String),
    /// An adapter was found but refused to create a device.
    RequestDevice(wgpu::RequestDeviceError),
    /// Neither a hardware adapter (DX12 or Vulkan) nor the software
    /// fallback could be used.
    NoHardwareOrFallback {
        dx12: Box<GpuError>,
        vulkan: Box<GpuError>,
        fallback: Box<GpuError>,
    },
    /// The device is lost (or a blocking poll failed); recreate the [`Gpu`].
    DeviceLost(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoAdapter(wanted) => write!(f, "no {wanted}"),
            GpuError::RequestDevice(e) => write!(f, "GPU device request failed: {e}"),
            GpuError::NoHardwareOrFallback {
                dx12,
                vulkan,
                fallback,
            } => write!(
                f,
                "no usable GPU: DX12 hardware failed ({dx12}); Vulkan hardware failed \
                 ({vulkan}); software fallback failed ({fallback})"
            ),
            GpuError::DeviceLost(why) => write!(f, "GPU device lost: {why}"),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GpuError::NoAdapter(_) | GpuError::DeviceLost(_) => None,
            GpuError::RequestDevice(e) => Some(e),
            GpuError::NoHardwareOrFallback { fallback, .. } => Some(fallback.as_ref()),
        }
    }
}

impl Gpu {
    /// Creates a wgpu instance limited to DX12 and Vulkan.
    ///
    /// A surface passed to [`Gpu::with_instance`] must be created from the
    /// same instance, so windowed callers create the instance first.
    pub fn create_instance() -> wgpu::Instance {
        instance_for(wgpu::Backends::DX12 | wgpu::Backends::VULKAN)
    }

    /// Creates a GPU on a fresh instance, preferring a high-performance
    /// hardware adapter compatible with `compatible_surface`.
    ///
    /// Note: a surface belongs to the instance that created it; to present to
    /// a window, use [`Gpu::create_instance`] + [`Gpu::with_instance`].
    pub fn new(compatible_surface: Option<&wgpu::Surface<'_>>) -> Result<Gpu, GpuError> {
        Self::with_instance(Self::create_instance(), compatible_surface)
    }

    /// Like [`Gpu::new`], on an existing instance (the one that created
    /// `compatible_surface`). After a device loss, call it again with the
    /// same instance and surface (a surface outlives the device).
    ///
    /// Setting [`FORCE_WARP_ENV`] (`MS_FORCE_WARP=1`) picks the software
    /// fallback adapter (WARP) instead, as for [`Gpu::headless`].
    pub fn with_instance(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Gpu, GpuError> {
        let kind = if force_warp() {
            tracing::info!("{FORCE_WARP_ENV}=1: using the software fallback adapter");
            AdapterKind::Fallback
        } else {
            AdapterKind::Any
        };
        Self::open(instance, compatible_surface, kind)
    }

    /// Creates an offscreen GPU: a DX12 hardware adapter, else a Vulkan
    /// hardware adapter, else the software fallback (WARP on DX12).
    ///
    /// Setting [`FORCE_WARP_ENV`] (`MS_FORCE_WARP=1`) skips the hardware
    /// attempts, e.g. `MS_FORCE_WARP=1 cargo test -p ms-render` runs the GPU
    /// tests on WARP.
    ///
    /// Each attempt uses a single-backend instance, so a Vulkan instance is
    /// only created when DX12 has no hardware adapter: creating and dropping
    /// Vulkan instances concurrently (e.g. parallel tests) crashes some
    /// Windows drivers/layers with an access violation.
    pub fn headless() -> Result<Gpu, GpuError> {
        if force_warp() {
            tracing::info!("{FORCE_WARP_ENV}=1: using the software fallback adapter");
            return Self::open(
                instance_for(wgpu::Backends::DX12),
                None,
                AdapterKind::Fallback,
            );
        }
        let dx12 = match Self::open(
            instance_for(wgpu::Backends::DX12),
            None,
            AdapterKind::Hardware,
        ) {
            Ok(gpu) => return Ok(gpu),
            Err(e) => e,
        };
        tracing::info!("no DX12 hardware GPU ({dx12}); trying Vulkan");
        let vulkan = match Self::open(
            instance_for(wgpu::Backends::VULKAN),
            None,
            AdapterKind::Hardware,
        ) {
            Ok(gpu) => return Ok(gpu),
            Err(e) => e,
        };
        tracing::warn!("no hardware GPU ({vulkan}); trying software fallback");
        Self::open(
            instance_for(wgpu::Backends::DX12),
            None,
            AdapterKind::Fallback,
        )
        .map_err(|fallback| GpuError::NoHardwareOrFallback {
            dx12: Box::new(dx12),
            vulkan: Box::new(vulkan),
            fallback: Box::new(fallback),
        })
    }

    /// Blocks until all submitted GPU work has finished.
    ///
    /// **Every blocking poll in the codebase must go through this helper**
    /// rather than calling `device.poll(..)` directly: wgpu 30 *panics* inside
    /// `Device::poll` when the device is lost (removed, out of memory, ...),
    /// and this catches that.
    ///
    /// Returns `Err(GpuError::DeviceLost)` without polling if [`Gpu::lost`]
    /// is already set. On a poll error or a caught panic, sets `lost`, logs
    /// and returns `Err`. Also returns `Err` if the device-lost callback fired
    /// during the poll.
    pub fn poll_wait(&self) -> Result<(), GpuError> {
        poll_wait_on(&self.device, &self.lost)
    }

    fn open(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
        kind: AdapterKind,
    ) -> Result<Gpu, GpuError> {
        let adapter = select_adapter(&instance, compatible_surface, kind)?;

        let info = adapter.get_info();
        tracing::info!(
            name = %info.name,
            backend = ?info.backend,
            device_type = ?info.device_type,
            driver = %info.driver,
            "selected GPU adapter"
        );

        let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ms-render device"),
            required_features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(GpuError::RequestDevice)?;

        device.on_uncaptured_error(Arc::new(|error: wgpu::Error| {
            tracing::error!("uncaptured wgpu error: {error}");
        }));

        let lost = Arc::new(AtomicBool::new(false));
        let lost_flag = Arc::clone(&lost);
        device.set_device_lost_callback(move |reason, message| {
            lost_flag.store(true, Ordering::SeqCst);
            match reason {
                wgpu::DeviceLostReason::Destroyed => {
                    tracing::info!("wgpu device destroyed: {message}");
                }
                wgpu::DeviceLostReason::Unknown => {
                    tracing::error!("wgpu device lost: {message}");
                }
            }
        });

        Ok(Gpu {
            instance,
            adapter,
            device,
            queue,
            lost,
        })
    }
}

/// Whether [`FORCE_WARP_ENV`] is set to `1`.
fn force_warp() -> bool {
    std::env::var(FORCE_WARP_ENV).is_ok_and(|v| v == "1")
}

fn instance_for(backends: wgpu::Backends) -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

/// Which adapters [`select_adapter`] may return.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AdapterKind {
    /// Best available, hardware preferred.
    Any,
    /// Hardware only (no CPU adapters).
    Hardware,
    /// Software only (CPU adapters such as WARP).
    Fallback,
}

/// Picks an adapter: hardware before CPU, then DX12 before Vulkan, then by
/// HighPerformance device type (discrete > integrated > other > virtual).
/// Done by hand because `request_adapter` keeps wgpu's backend order, which
/// puts Vulkan first.
fn select_adapter(
    instance: &wgpu::Instance,
    compatible_surface: Option<&wgpu::Surface<'_>>,
    kind: AdapterKind,
) -> Result<wgpu::Adapter, GpuError> {
    fn rank(device_type: wgpu::DeviceType) -> u8 {
        match device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::Other => 2,
            wgpu::DeviceType::VirtualGpu => 3,
            wgpu::DeviceType::Cpu => 4,
        }
    }

    let mut candidates = Vec::new();
    for backend in [wgpu::Backends::DX12, wgpu::Backends::VULKAN] {
        candidates.extend(pollster::block_on(instance.enumerate_adapters(backend)));
    }
    candidates.retain(|adapter| {
        let is_cpu = adapter.get_info().device_type == wgpu::DeviceType::Cpu;
        let kind_ok = match kind {
            AdapterKind::Any => true,
            AdapterKind::Hardware => !is_cpu,
            AdapterKind::Fallback => is_cpu,
        };
        kind_ok && compatible_surface.is_none_or(|surface| adapter.is_surface_supported(surface))
    });
    // DX12 is the primary backend: any DX12 hardware adapter beats Vulkan.
    candidates.sort_by_key(|adapter| {
        let info = adapter.get_info();
        (
            info.device_type == wgpu::DeviceType::Cpu,
            info.backend != wgpu::Backend::Dx12,
            rank(info.device_type),
        )
    });
    candidates.into_iter().next().ok_or_else(|| {
        let what = match kind {
            AdapterKind::Any => "GPU adapter",
            AdapterKind::Hardware => "hardware GPU adapter",
            AdapterKind::Fallback => "software fallback adapter",
        };
        let surface = if compatible_surface.is_some() {
            " compatible with the surface"
        } else {
            ""
        };
        GpuError::NoAdapter(format!("{what}{surface}"))
    })
}

/// How a readable texture format is turned into RGBA8.
#[derive(Clone, Copy)]
enum Readback {
    Rgba8,
    Bgra8,
    Rgba16Float,
}

impl Readback {
    fn for_format(format: wgpu::TextureFormat) -> Option<Readback> {
        use wgpu::TextureFormat as F;
        match format {
            F::Rgba8Unorm | F::Rgba8UnormSrgb => Some(Readback::Rgba8),
            F::Bgra8Unorm | F::Bgra8UnormSrgb => Some(Readback::Bgra8),
            F::Rgba16Float => Some(Readback::Rgba16Float),
            _ => None,
        }
    }

    fn bytes_per_pixel(self) -> u32 {
        match self {
            Readback::Rgba8 | Readback::Bgra8 => 4,
            Readback::Rgba16Float => 8,
        }
    }

    fn convert(self, texel: &[u8]) -> [u8; 4] {
        match self {
            Readback::Rgba8 => [texel[0], texel[1], texel[2], texel[3]],
            Readback::Bgra8 => [texel[2], texel[1], texel[0], texel[3]],
            Readback::Rgba16Float => {
                let channel = |i: usize| {
                    let v = half::f16::from_le_bytes([texel[2 * i], texel[2 * i + 1]]).to_f32();
                    // NaN saturates to 0 in the cast.
                    (v.clamp(0.0, 1.0) * 255.0).round() as u8
                };
                [channel(0), channel(1), channel(2), channel(3)]
            }
        }
    }
}

/// [`Gpu::poll_wait`] for a device and its lost flag (also used where only a
/// [`RendererShared`](crate::RendererShared) is at hand, e.g. the loader
/// thread's warm-up draw).
pub(crate) fn poll_wait_on(device: &wgpu::Device, lost: &AtomicBool) -> Result<(), GpuError> {
    if lost.load(Ordering::SeqCst) {
        return Err(GpuError::DeviceLost("device already lost".into()));
    }
    let polled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device.poll(wgpu::PollType::wait_indefinitely())
    }));
    let failure = match polled {
        Ok(Ok(_)) => None,
        Ok(Err(e)) => Some(format!("device poll failed: {e}")),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            Some(format!("device poll panicked: {msg}"))
        }
    };
    if let Some(why) = failure {
        lost.store(true, Ordering::SeqCst);
        tracing::error!("{why}");
        return Err(GpuError::DeviceLost(why));
    }
    if lost.load(Ordering::SeqCst) {
        return Err(GpuError::DeviceLost("device lost during poll".into()));
    }
    Ok(())
}

/// Reads mip 0 / layer 0 of a 2D texture back to the CPU as RGBA8, blocking
/// until the copy completes.
///
/// Supports `Rgba8Unorm[Srgb]`, `Bgra8Unorm[Srgb]` (swizzled to RGBA) and
/// `Rgba16Float` (clamped to [0, 1], scaled by 255 and rounded). The texture
/// needs `COPY_SRC` usage. Anything unsupported, or any GPU failure, logs an
/// error and returns an opaque black image of the texture's size.
pub fn read_texture(gpu: &Gpu, tex: &wgpu::Texture) -> RgbaImage {
    let (width, height) = (tex.width(), tex.height());
    let black = || RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 255]));

    let Some(readback) = Readback::for_format(tex.format()) else {
        tracing::error!("read_texture: unsupported format {:?}", tex.format());
        return black();
    };
    if !tex.usage().contains(wgpu::TextureUsages::COPY_SRC) {
        tracing::error!("read_texture: texture lacks COPY_SRC usage");
        return black();
    }
    if gpu.lost.load(Ordering::SeqCst) {
        tracing::error!("read_texture: device is lost");
        return black();
    }
    if tex.sample_count() != 1 || tex.dimension() != wgpu::TextureDimension::D2 {
        tracing::error!("read_texture: only single-sampled 2D textures can be read back");
        return black();
    }

    match copy_and_map(gpu, tex, readback) {
        Some(img) => img,
        None => black(),
    }
}

fn copy_and_map(gpu: &Gpu, tex: &wgpu::Texture, readback: Readback) -> Option<RgbaImage> {
    let (width, height) = (tex.width(), tex.height());
    let unpadded_row = width * readback.bytes_per_pixel();
    let padded_row = unpadded_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_texture staging"),
        size: u64::from(padded_row) * u64::from(height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_texture"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let (tx, rx) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        // The receiver only disappears if we already gave up waiting.
        let _ = tx.send(result);
    });
    if let Err(e) = gpu.poll_wait() {
        tracing::error!("read_texture: {e}");
        return None;
    }
    match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::error!("read_texture: buffer map failed: {e}");
            return None;
        }
        Err(e) => {
            tracing::error!("read_texture: buffer map never completed: {e}");
            return None;
        }
    }

    let img = {
        let view = match buffer.get_mapped_range(..) {
            Ok(view) => view,
            Err(e) => {
                tracing::error!("read_texture: mapped range unavailable: {e}");
                return None;
            }
        };
        let bpp = readback.bytes_per_pixel() as usize;
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for row in view.chunks_exact(padded_row as usize) {
            for texel in row[..unpadded_row as usize].chunks_exact(bpp) {
                pixels.extend_from_slice(&readback.convert(texel));
            }
        }
        RgbaImage::from_raw(width, height, pixels)
    };
    buffer.unmap();
    if img.is_none() {
        tracing::error!("read_texture: readback size mismatch");
    }
    img
}
