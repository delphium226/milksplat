//! GPU setup, readback and golden-harness tests.
//!
//! These run on the machine's real adapter, or on WARP when no hardware
//! adapter exists (or when `MS_FORCE_WARP=1` is set). If no device can be
//! created at all, the tests fail loudly rather than skipping.

use ms_render::test_support::{test_gpu, TestGpu};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use image::{Rgba, RgbaImage};
use ms_render::{golden, read_texture, Gpu};

/// A headless GPU holding the GPU test lock (see `ms_render::test_support`).
fn gpu() -> TestGpu {
    test_gpu()
}

fn texture(gpu: &Gpu, w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test texture"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn upload(gpu: &Gpu, tex: &wgpu::Texture, bytes_per_pixel: u32, data: &[u8]) {
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(tex.width() * bytes_per_pixel),
            rows_per_image: Some(tex.height()),
        },
        tex.size(),
    );
}

/// Recognisable per-pixel pattern for row-padding checks.
fn pattern(x: u32, y: u32) -> [u8; 4] {
    [(x * 7) as u8, (y * 41) as u8, (x + y) as u8, 255]
}

fn unique_temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("ms-render-{tag}-{}-{nanos}", std::process::id()))
}

fn panic_message(result: std::thread::Result<()>) -> String {
    match result {
        Ok(()) => panic!("expected a panic, but the call returned normally"),
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    }
}

#[test]
fn headless_device() {
    let _lock = ms_render::test_support::gpu_lock();
    let gpu = Gpu::headless();
    assert!(gpu.is_ok(), "Gpu::headless() failed: {:?}", gpu.err());
    if let Ok(gpu) = gpu {
        let info = gpu.adapter.get_info();
        eprintln!(
            "adapter: {} ({:?}, {:?}, driver {} {})",
            info.name, info.backend, info.device_type, info.driver, info.driver_info
        );
        assert!(!gpu.lost.load(std::sync::atomic::Ordering::SeqCst));
    }
}

#[test]
fn clear_and_read() {
    let gpu = gpu();
    let tex = texture(&gpu, 64, 64, wgpu::TextureFormat::Rgba8Unorm);
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear red"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
    }
    gpu.queue.submit([encoder.finish()]);

    let img = read_texture(&gpu, &tex);
    assert_eq!(img.dimensions(), (64, 64));
    for (x, y, p) in img.enumerate_pixels() {
        assert_eq!(p.0, [255, 0, 0, 255], "pixel ({x},{y})");
    }
}

#[test]
fn odd_width_readback() {
    let gpu = gpu();
    let (w, h) = (37, 5);
    let tex = texture(&gpu, w, h, wgpu::TextureFormat::Rgba8Unorm);
    let data: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| pattern(x, y)))
        .collect();
    upload(&gpu, &tex, 4, &data);

    let img = read_texture(&gpu, &tex);
    assert_eq!(img.dimensions(), (w, h));
    for (x, y, p) in img.enumerate_pixels() {
        assert_eq!(p.0, pattern(x, y), "pixel ({x},{y})");
    }
}

#[test]
fn bgra_readback_is_swizzled_to_rgba() {
    let gpu = gpu();
    let tex = texture(&gpu, 3, 2, wgpu::TextureFormat::Bgra8Unorm);
    // Stored as B, G, R, A.
    let data: Vec<u8> = [10u8, 20, 30, 40].repeat(6);
    upload(&gpu, &tex, 4, &data);

    let img = read_texture(&gpu, &tex);
    for p in img.pixels() {
        assert_eq!(p.0, [30, 20, 10, 40]);
    }
}

#[test]
fn rgba16float_readback_clamps_and_scales() {
    let gpu = gpu();
    let tex = texture(&gpu, 5, 3, wgpu::TextureFormat::Rgba16Float);
    // (0.5, 1.0, 2.0 -> clamped to 1, -1.0 -> clamped to 0)
    let texel: Vec<u8> = [0.5f32, 1.0, 2.0, -1.0]
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let data = texel.repeat(5 * 3);
    upload(&gpu, &tex, 8, &data);

    let img = read_texture(&gpu, &tex);
    assert_eq!(img.dimensions(), (5, 3));
    for p in img.pixels() {
        assert_eq!(p.0, [128, 255, 255, 0]);
    }
}

#[test]
fn uncaptured_error_does_not_panic() {
    let gpu = gpu();
    // Zero-sized texture: a validation error with no error scope pushed.
    let _bad = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("invalid"),
        size: wgpu::Extent3d {
            width: 0,
            height: 0,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
}

#[test]
fn golden_roundtrip() {
    let dir = unique_temp_dir("golden-roundtrip");
    let dir_str = dir.to_string_lossy().into_owned();
    let (w, h) = (100, 100);
    let img = RgbaImage::from_fn(w, h, |x, y| Rgba(pattern(x, y)));

    // Bless writes the reference (creating directories).
    golden::check_with(&img, &dir_str, "roundtrip", true);
    let reference = dir.join("tests").join("golden").join("roundtrip.png");
    assert!(reference.is_file(), "bless did not write {reference:?}");

    // The same image passes.
    golden::check_with(&img, &dir_str, "roundtrip", false);

    // Small differences (<= 8 per channel) are ignored entirely.
    let mut nudged = img.clone();
    for p in nudged.pixels_mut() {
        p.0[0] = p.0[0].saturating_add(8);
    }
    golden::check_with(&nudged, &dir_str, "roundtrip", false);

    // 0.4% of pixels changed: within tolerance.
    let mut few = img.clone();
    for i in 0..40 {
        few.put_pixel(i, 0, Rgba([255, 255, 255, 0]));
    }
    golden::check_with(&few, &dir_str, "roundtrip", false);

    // 1% of pixels changed: fails and writes the actual image.
    let mut changed = img.clone();
    for i in 0..100 {
        changed.put_pixel(i, 1, Rgba([255, 255, 255, 0]));
    }
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        golden::check_with(&changed, &dir_str, "roundtrip", false)
    })));
    assert!(msg.contains("roundtrip"), "unhelpful message: {msg}");
    let actual = dir
        .join("tests")
        .join("golden")
        .join("roundtrip.actual.png");
    assert!(actual.is_file(), "failure did not write {actual:?}");

    // A size mismatch fails.
    let small = RgbaImage::from_fn(10, 10, |x, y| Rgba(pattern(x, y)));
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        golden::check_with(&small, &dir_str, "roundtrip", false)
    })));
    assert!(msg.contains("size"), "unhelpful message: {msg}");

    // The macro form compiles against this crate's manifest dir (not run, so
    // the repo is not touched).
    let _compiles = || ms_render::assert_golden!(img, "never_run");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn golden_missing_reference_fails_with_bless_hint() {
    let dir = unique_temp_dir("golden-missing");
    let dir_str = dir.to_string_lossy().into_owned();
    let img = RgbaImage::from_pixel(4, 4, Rgba([1, 2, 3, 255]));

    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        golden::check_with(&img, &dir_str, "does_not_exist", false)
    })));
    assert!(
        msg.contains("MS_BLESS=1"),
        "message lacks bless hint: {msg}"
    );
    assert!(msg.contains("does_not_exist"), "message lacks name: {msg}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn lost_device_reads_black_and_poll_wait_errs() {
    let gpu = gpu();
    let tex = texture(&gpu, 8, 4, wgpu::TextureFormat::Rgba8Unorm);
    upload(&gpu, &tex, 4, &[200u8; 8 * 4 * 4]);
    gpu.device.destroy();

    assert!(
        gpu.poll_wait().is_err(),
        "poll_wait on a destroyed device must fail"
    );
    assert!(gpu.lost.load(std::sync::atomic::Ordering::SeqCst));

    let img = read_texture(&gpu, &tex);
    assert_eq!(img.dimensions(), (8, 4));
    for p in img.pixels() {
        assert_eq!(p.0, [0, 0, 0, 255]);
    }
}
