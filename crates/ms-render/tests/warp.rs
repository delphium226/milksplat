//! Feedback chain and built-in warp pass tests.
//!
//! These run on the machine's real adapter, or on WARP with
//! `MS_FORCE_WARP=1`.

use half::f16;
use image::{Rgba, RgbaImage};
use ms_render::{
    read_texture, FeedbackFormat, Gpu, MeshData, RenderConfig, Renderer, Uniforms, WarpLayer,
    WarpShader,
};

fn gpu() -> Gpu {
    match Gpu::headless() {
        Ok(gpu) => gpu,
        Err(e) => panic!("Gpu::headless() failed (no hardware adapter and no WARP): {e}"),
    }
}

fn renderer(gpu: &Gpu, w: u32, h: u32, format: FeedbackFormat) -> Renderer {
    Renderer::new(
        gpu,
        RenderConfig {
            internal_size: [w, h],
            format,
        },
    )
}

/// Identity mesh: every vertex samples the texel under it, colour white.
fn identity_mesh(cols: u32, rows: u32) -> MeshData {
    let mut uv = Vec::new();
    for y in 0..=rows {
        for x in 0..=cols {
            uv.push([x as f32 / cols as f32, y as f32 / rows as f32]);
        }
    }
    let n = uv.len();
    MeshData {
        cols,
        rows,
        uv,
        color: vec![[1.0; 4]; n],
    }
}

fn vertex_count(mesh: &MeshData) -> usize {
    ((mesh.cols + 1) * (mesh.rows + 1)) as usize
}

fn uniforms_with_decay(decay: f32) -> Uniforms {
    Uniforms {
        decay,
        ..Uniforms::default()
    }
}

fn assert_all_near(img: &RgbaImage, rgb: u8, tol: u8) {
    for (x, y, p) in img.enumerate_pixels() {
        for c in 0..3 {
            assert!(
                p.0[c].abs_diff(rgb) <= tol,
                "pixel ({x},{y}) channel {c} = {} (want {rgb} +/- {tol}); pixel {:?}",
                p.0[c],
                p.0
            );
        }
        assert_eq!(p.0[3], 255, "pixel ({x},{y}) alpha");
    }
}

/// Deterministic 128x128 test pattern: coloured quadrants, a white cross
/// through the centre and a grid of white dots, softened by four 3x3 box
/// blurs.
///
/// The softening keeps the golden portable: hardware bilinear filtering
/// quantises sub-texel weights differently per device, and over 30 zoom
/// passes a hard 1-pixel edge drifts by up to 16 levels between NVIDIA and
/// WARP (2.2% of pixels over the golden tolerance). Band-limited edges keep
/// the drift at 6 or less.
fn test_pattern() -> RgbaImage {
    let sharp = RgbaImage::from_fn(128, 128, |x, y| {
        let cross = (63..=64).contains(&x) || (63..=64).contains(&y);
        let dot = (6..10).contains(&(x % 16)) && (6..10).contains(&(y % 16));
        if cross || dot {
            return Rgba([255, 255, 255, 255]);
        }
        match (x < 64, y < 64) {
            (true, true) => Rgba([220, 40, 40, 255]),
            (false, true) => Rgba([40, 200, 60, 255]),
            (true, false) => Rgba([40, 60, 220, 255]),
            (false, false) => Rgba([230, 210, 40, 255]),
        }
    });
    (0..4).fold(sharp, |img, _| box_blur(&img))
}

/// 3x3 box blur, clamped at the edges, rounded.
fn box_blur(img: &RgbaImage) -> RgbaImage {
    let (w, h) = img.dimensions();
    RgbaImage::from_fn(w, h, |x, y| {
        let (mut sum, mut n) = ([0u32; 4], 0u32);
        for sy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            for sx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                for (acc, v) in sum.iter_mut().zip(img.get_pixel(sx, sy).0) {
                    *acc += u32::from(v);
                }
                n += 1;
            }
        }
        Rgba(sum.map(|v| ((v + n / 2) / n) as u8))
    })
}

/// Reads an `Rgba16Float` texture back as raw f16 texels (no clamping), so
/// NaN and Inf are visible.
fn read_f16(gpu: &Gpu, tex: &wgpu::Texture) -> Vec<[f16; 4]> {
    assert_eq!(tex.format(), wgpu::TextureFormat::Rgba16Float);
    let (w, h) = (tex.width(), tex.height());
    let row = w * 8;
    let padded =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_f16"),
        size: u64::from(padded) * u64::from(h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
        },
        tex.size(),
    );
    gpu.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    gpu.poll_wait().unwrap();
    rx.recv().unwrap().unwrap();
    let view = buffer.get_mapped_range(..).unwrap();
    let mut out = Vec::new();
    for r in view.chunks_exact(padded as usize) {
        for t in r[..row as usize].as_chunks::<8>().0 {
            out.push([0, 1, 2, 3].map(|c| f16::from_le_bytes([t[2 * c], t[2 * c + 1]])));
        }
    }
    out
}

#[test]
fn identity_warp_decays() {
    let gpu = gpu();
    for format in [FeedbackFormat::Rgba8, FeedbackFormat::Fp16] {
        let mut r = renderer(&gpu, 64, 48, format);
        let builtin = WarpShader::builtin(&gpu, &r);
        r.write_feedback(
            &gpu,
            &RgbaImage::from_pixel(64, 48, Rgba([128, 128, 128, 255])),
        );
        let mesh = identity_mesh(8, 6);
        let weight = vec![1.0; vertex_count(&mesh)];
        let uniforms = uniforms_with_decay(0.5);
        r.warp(
            &gpu,
            &[WarpLayer {
                mesh: &mesh,
                weight: &weight,
                shader: &builtin,
                uniforms: &uniforms,
            }],
        );
        let img = read_texture(&gpu, r.feedback_texture());
        assert_eq!(img.dimensions(), (64, 48), "{format:?}");
        assert_all_near(&img, 64, 1);
    }
}

/// One identity warp reproduces an asymmetric image: catches a flipped or
/// mirrored mesh/UV mapping (which an even number of warps would hide).
#[test]
fn identity_warp_keeps_orientation() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 40, 30, FeedbackFormat::Rgba8);
    let builtin = WarpShader::builtin(&gpu, &r);
    let seed = RgbaImage::from_fn(40, 30, |x, y| Rgba([(x * 6) as u8, (y * 8) as u8, 77, 255]));
    r.write_feedback(&gpu, &seed);
    let mesh = identity_mesh(5, 3);
    let weight = vec![1.0; vertex_count(&mesh)];
    let uniforms = uniforms_with_decay(1.0);
    r.warp(
        &gpu,
        &[WarpLayer {
            mesh: &mesh,
            weight: &weight,
            shader: &builtin,
            uniforms: &uniforms,
        }],
    );
    let img = read_texture(&gpu, r.feedback_texture());
    for (x, y, p) in img.enumerate_pixels() {
        let want = seed.get_pixel(x, y);
        for c in 0..4 {
            assert!(
                p.0[c].abs_diff(want.0[c]) <= 1,
                "pixel ({x},{y}): {:?} want {:?}",
                p.0,
                want.0
            );
        }
    }
}

#[test]
fn zoom_mesh_golden() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 128, 128, FeedbackFormat::Rgba8);
    let builtin = WarpShader::builtin(&gpu, &r);
    r.write_feedback(&gpu, &test_pattern());
    let mut mesh = identity_mesh(32, 24);
    for uv in &mut mesh.uv {
        *uv = [0.5 + (uv[0] - 0.5) * 0.98, 0.5 + (uv[1] - 0.5) * 0.98];
    }
    let weight = vec![1.0; vertex_count(&mesh)];
    let uniforms = uniforms_with_decay(1.0);
    for _ in 0..30 {
        r.warp(
            &gpu,
            &[WarpLayer {
                mesh: &mesh,
                weight: &weight,
                shader: &builtin,
                uniforms: &uniforms,
            }],
        );
    }
    let img = read_texture(&gpu, r.feedback_texture());
    ms_render::assert_golden!(img, "warp_zoom");
}

#[test]
fn two_layers_weighted() {
    let gpu = gpu();
    for format in [FeedbackFormat::Rgba8, FeedbackFormat::Fp16] {
        let mut r = renderer(&gpu, 32, 32, format);
        let builtin = WarpShader::builtin(&gpu, &r);
        r.write_feedback(&gpu, &RgbaImage::from_pixel(32, 32, Rgba([255; 4])));
        let mesh = identity_mesh(4, 4);
        let n = vertex_count(&mesh);
        let (wa, wb) = (vec![0.25; n], vec![0.75; n]);
        let (ua, ub) = (uniforms_with_decay(1.0), uniforms_with_decay(0.0));
        r.warp(
            &gpu,
            &[
                WarpLayer {
                    mesh: &mesh,
                    weight: &wa,
                    shader: &builtin,
                    uniforms: &ua,
                },
                WarpLayer {
                    mesh: &mesh,
                    weight: &wb,
                    shader: &builtin,
                    uniforms: &ub,
                },
            ],
        );
        let img = read_texture(&gpu, r.feedback_texture());
        assert_all_near(&img, 64, 1);
    }
}

#[test]
fn fp16_nan_sanitised() {
    let gpu = gpu();
    let (w, h) = (32u32, 32u32);
    let mut r = renderer(&gpu, w, h, FeedbackFormat::Fp16);
    let builtin = WarpShader::builtin(&gpu, &r);
    // Left half NaN, top-right quarter +Inf, bottom-right quarter 100.0.
    let seed: Vec<[f16; 4]> = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let v = if x < w / 2 {
                    f16::NAN
                } else if y < h / 2 {
                    f16::INFINITY
                } else {
                    f16::from_f32(100.0)
                };
                [v, v, v, f16::ONE]
            })
        })
        .collect();
    r.write_feedback_f16(&gpu, &seed);
    assert!(
        read_f16(&gpu, r.feedback_texture())
            .iter()
            .any(|t| t[0].is_nan()),
        "the seed must contain NaN"
    );

    let mesh = identity_mesh(4, 4);
    let weight = vec![1.0; vertex_count(&mesh)];
    let uniforms = uniforms_with_decay(1.0);
    r.warp(
        &gpu,
        &[WarpLayer {
            mesh: &mesh,
            weight: &weight,
            shader: &builtin,
            uniforms: &uniforms,
        }],
    );
    let out = read_f16(&gpu, r.feedback_texture());
    assert_eq!(out.len(), (w * h) as usize);
    for (i, t) in out.iter().enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        for (c, v) in t.iter().enumerate() {
            assert!(v.is_finite(), "texel ({x},{y}) channel {c} = {v}");
            assert!(
                (0.0..=64.0).contains(&v.to_f32()),
                "texel ({x},{y}) channel {c} = {v} outside [0, 64]"
            );
        }
        // Away from the NaN/Inf neighbours, 100.0 clamps to exactly 64.
        if (18..30).contains(&x) && (18..30).contains(&y) {
            assert_eq!(t[0].to_f32(), 64.0, "texel ({x},{y})");
        }
        // NaN and Inf become 0.
        if x < 14 && (2..30).contains(&y) || (18..30).contains(&x) && (2..14).contains(&y) {
            assert_eq!(t[0].to_f32(), 0.0, "texel ({x},{y})");
        }
    }
}

#[test]
fn resize_clamps() {
    let gpu = gpu();
    let max_dim = gpu.device.limits().max_texture_dimension_2d;
    let mut r = renderer(&gpu, 64, 64, FeedbackFormat::Rgba8);
    r.resize(&gpu, [1, 1]);
    assert_eq!(r.feedback_texture().size().width, 16);
    assert_eq!(r.feedback_texture().size().height, 16);
    assert_eq!(r.internal_size(), [16, 16]);
    r.resize(&gpu, [100_000, 10]);
    assert_eq!(r.internal_size(), [max_dim, 16]);
    assert_eq!(r.feedback_texture().width(), max_dim);
    assert_eq!(r.feedback_texture().height(), 16);
}

/// Solid quadrants survive a resize (bilinear copy) and a format change.
#[test]
fn resize_and_set_format_keep_contents() {
    let gpu = gpu();
    let quadrants = |w: u32, h: u32| {
        RgbaImage::from_fn(w, h, |x, y| match (x < w / 2, y < h / 2) {
            (true, true) => Rgba([200, 0, 0, 255]),
            (false, true) => Rgba([0, 200, 0, 255]),
            (true, false) => Rgba([0, 0, 200, 255]),
            (false, false) => Rgba([100, 150, 50, 255]),
        })
    };
    let mut r = renderer(&gpu, 64, 64, FeedbackFormat::Rgba8);
    r.write_feedback(&gpu, &quadrants(64, 64));

    r.resize(&gpu, [128, 96]);
    let img = read_texture(&gpu, r.feedback_texture());
    assert_eq!(img.dimensions(), (128, 96));
    let want = quadrants(128, 96);
    // Away from the seams (which blend bilinearly).
    for (x, y) in [(10, 10), (100, 20), (30, 80), (120, 90), (60, 44), (68, 52)] {
        assert_eq!(
            img.get_pixel(x, y),
            want.get_pixel(x, y),
            "after resize, ({x},{y})"
        );
    }

    r.set_format(&gpu, FeedbackFormat::Fp16);
    assert_eq!(r.format(), FeedbackFormat::Fp16);
    assert_eq!(
        r.feedback_texture().format(),
        wgpu::TextureFormat::Rgba16Float
    );
    let fp = read_texture(&gpu, r.feedback_texture());
    for (a, b) in img.pixels().zip(fp.pixels()) {
        for c in 0..4 {
            assert!(a.0[c].abs_diff(b.0[c]) <= 1, "{a:?} vs {b:?}");
        }
    }

    r.set_format(&gpu, FeedbackFormat::Rgba8);
    assert_eq!(
        r.feedback_texture().format(),
        wgpu::TextureFormat::Rgba8Unorm
    );
    let back = read_texture(&gpu, r.feedback_texture());
    for (a, b) in img.pixels().zip(back.pixels()) {
        for c in 0..4 {
            assert!(a.0[c].abs_diff(b.0[c]) <= 1, "{a:?} vs {b:?}");
        }
    }
}

/// A layer whose arrays do not match `(cols+1)*(rows+1)` is skipped (logged),
/// not a panic; the target is still cleared.
#[test]
fn malformed_layer_is_skipped() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 32, 32, FeedbackFormat::Rgba8);
    let builtin = WarpShader::builtin(&gpu, &r);
    r.write_feedback(&gpu, &RgbaImage::from_pixel(32, 32, Rgba([255; 4])));
    let mut mesh = identity_mesh(4, 4);
    mesh.uv.pop();
    let weight = vec![1.0; vertex_count(&mesh)];
    let uniforms = uniforms_with_decay(1.0);
    r.warp(
        &gpu,
        &[WarpLayer {
            mesh: &mesh,
            weight: &weight,
            shader: &builtin,
            uniforms: &uniforms,
        }],
    );
    assert_all_near(&read_texture(&gpu, r.feedback_texture()), 0, 0);
}
