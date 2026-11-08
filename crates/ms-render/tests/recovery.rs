//! Device-loss recovery (spec §6.7): `Renderer::recreate` on a new `Gpu`,
//! stale shader objects, and recompiling a `ShaderSet` from its sources.
//!
//! These run on the machine's real adapter, or on WARP with
//! `MS_FORCE_WARP=1`.

use image::{Rgba, RgbaImage};
use ms_render::{
    read_texture, CompShader, CompositeLayer, DrawList, FeedbackFormat, Gpu, MeshData,
    RenderConfig, Renderer, ShapeDraw, Uniforms, WarpLayer, WarpShader,
};

const SIZE: u32 = 128;
const RGBA8: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn gpu() -> Gpu {
    match Gpu::headless() {
        Ok(gpu) => gpu,
        Err(e) => panic!("Gpu::headless() failed (no hardware adapter and no WARP): {e}"),
    }
}

fn renderer(gpu: &Gpu) -> Renderer {
    Renderer::new(
        gpu,
        RenderConfig {
            internal_size: [SIZE, SIZE],
            format: FeedbackFormat::Rgba8,
        },
    )
}

fn no_textures(_: &str) -> Option<RgbaImage> {
    None
}

/// Coloured quadrants with a white cross.
fn pattern() -> RgbaImage {
    RgbaImage::from_fn(SIZE, SIZE, |x, y| {
        if (60..68).contains(&x) || (60..68).contains(&y) {
            return Rgba([255, 255, 255, 255]);
        }
        match (x < 64, y < 64) {
            (true, true) => Rgba([220, 40, 40, 255]),
            (false, true) => Rgba([40, 200, 60, 255]),
            (true, false) => Rgba([40, 60, 220, 255]),
            (false, false) => Rgba([230, 210, 40, 255]),
        }
    })
}

/// A 32x24 mesh zooming in by 2% per tick (as in the `warp_zoom` golden).
fn zoom_mesh() -> MeshData {
    let (cols, rows) = (32, 24);
    let mut uv = Vec::new();
    for y in 0..=rows {
        for x in 0..=cols {
            let (u, v) = (x as f32 / cols as f32, y as f32 / rows as f32);
            uv.push([0.5 + (u - 0.5) * 0.98, 0.5 + (v - 0.5) * 0.98]);
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

fn target(gpu: &Gpu) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("composite target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RGBA8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// `ticks` warp passes with `warp` (decay 1), each followed by a composite
/// with `comp`; returns the last composite.
fn run_ticks(
    gpu: &Gpu,
    r: &mut Renderer,
    warp: &WarpShader,
    comp: &CompShader,
    ticks: usize,
) -> RgbaImage {
    let mesh = zoom_mesh();
    let weight = vec![1.0; mesh.uv.len()];
    let uniforms = Uniforms {
        decay: 1.0,
        ..Uniforms::default()
    };
    let tex = target(gpu);
    let view = tex.create_view(&Default::default());
    for _ in 0..ticks {
        r.warp(
            gpu,
            &[WarpLayer {
                mesh: &mesh,
                weight: &weight,
                shader: warp,
                uniforms: &uniforms,
            }],
        );
        r.composite(
            gpu,
            &view,
            RGBA8,
            &[CompositeLayer {
                shader: comp,
                uniforms: &uniforms,
                alpha: 1.0,
            }],
        );
    }
    read_texture(gpu, &tex)
}

/// Composites the current feedback once with `comp` and reads it back.
fn composite_once(gpu: &Gpu, r: &mut Renderer, comp: &CompShader) -> RgbaImage {
    let tex = target(gpu);
    let view = tex.create_view(&Default::default());
    r.composite(
        gpu,
        &view,
        RGBA8,
        &[CompositeLayer {
            shader: comp,
            uniforms: &Uniforms::default(),
            alpha: 1.0,
        }],
    );
    read_texture(gpu, &tex)
}

/// Runs `f` inside validation and out-of-memory error scopes on `gpu` and
/// returns the first error, if any.
fn gpu_error(gpu: &Gpu, f: impl FnOnce()) -> Option<wgpu::Error> {
    let validation = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let oom = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    f();
    let oom = pollster::block_on(oom.pop());
    let validation = pollster::block_on(validation.pop());
    validation.or(oom)
}

fn is_black(img: &RgbaImage) -> bool {
    img.pixels().all(|p| p.0 == [0, 0, 0, 255])
}

fn red_square() -> DrawList {
    DrawList {
        shapes: vec![ShapeDraw {
            x: 0.5,
            y: 0.5,
            rad: 0.3,
            sides: 4,
            inner: [1.0, 0.0, 0.0, 1.0],
            outer: [1.0, 0.0, 0.0, 1.0],
            ..ShapeDraw::default()
        }],
        ..DrawList::default()
    }
}

#[test]
fn recreate_on_new_device() {
    let old_gpu = gpu();
    let mut r = renderer(&old_gpu);
    r.write_feedback(&old_gpu, &pattern());
    let builtin = r.compile_shaders(&old_gpu, None, None, RGBA8, &no_textures);
    let before = run_ticks(&old_gpu, &mut r, &builtin.warp, &builtin.comp, 10);
    assert!(!is_black(&before));

    let new_gpu = gpu();
    let error = gpu_error(&new_gpu, || r.recreate(&new_gpu));
    assert!(error.is_none(), "recreate: {error:?}");
    drop(old_gpu);
    assert_eq!(r.internal_size(), [SIZE, SIZE], "config kept");
    assert_eq!(r.format(), FeedbackFormat::Rgba8, "config kept");
    assert!(
        is_black(&read_texture(&new_gpu, r.feedback_texture())),
        "the feedback is cleared to black"
    );

    // Callers fetch new shader objects after `recreate`.
    let set = r.compile_shaders(&new_gpu, None, None, RGBA8, &no_textures);
    let error = gpu_error(&new_gpu, || {
        r.draw(&new_gpu, &red_square(), 1.0);
        r.blur(&new_gpu, 3);
        let img = run_ticks(&new_gpu, &mut r, &set.warp, &set.comp, 10);
        assert!(!is_black(&img), "the shape survives 10 zoom ticks");
        // Centre is red (the zoom grows the square).
        let p = img.get_pixel(SIZE / 2, SIZE / 2).0;
        assert!(p[0] > 200 && p[1] < 40 && p[2] < 40, "centre {p:?}");
    });
    assert!(error.is_none(), "rendering after recreate: {error:?}");
}

#[test]
fn stale_shader_set_after_recreate_does_not_panic() {
    let old_gpu = gpu();
    let mut r = renderer(&old_gpu);
    let stale_shared = r.shared();
    // A user warp that blacks everything out and a red user composite: if
    // either were used after `recreate`, the output would show it.
    let stale = r.compile_shaders(
        &old_gpu,
        Some("return vec3f(0.0);"),
        Some("return vec3f(1.0, 0.0, 0.0);"),
        RGBA8,
        &no_textures,
    );
    assert!(stale.errors.is_empty(), "{:?}", stale.errors);
    let stale_builtin_warp = WarpShader::builtin(&old_gpu, &r);

    let new_gpu = gpu();
    r.recreate(&new_gpu);
    // The old device is lost, and a loader thread still holding the stale
    // `RendererShared` compiles with it afterwards.
    old_gpu.device.destroy();
    let late = stale_shared.compile_shaders(
        None,
        Some("return vec3f(0.0, 1.0, 0.0);"),
        RGBA8,
        &no_textures,
    );

    let fresh = r.compile_shaders(&new_gpu, None, None, RGBA8, &no_textures);
    r.write_feedback(&new_gpu, &pattern());
    let expected = {
        let mut reference = renderer(&new_gpu);
        reference.write_feedback(&new_gpu, &pattern());
        let set = reference.compile_shaders(&new_gpu, None, None, RGBA8, &no_textures);
        run_ticks(&new_gpu, &mut reference, &set.warp, &set.comp, 3)
    };
    let error = gpu_error(&new_gpu, || {
        // Stale objects fall back to the built-ins (with a logged error).
        let img = run_ticks(&new_gpu, &mut r, &stale.warp, &stale.comp, 1);
        let img2 = run_ticks(&new_gpu, &mut r, &stale_builtin_warp, &late.comp, 1);
        let img3 = run_ticks(&new_gpu, &mut r, &fresh.warp, &stale.comp, 1);
        // Not black (stale warp) and not red (stale comp): the white cross.
        let centre = img.get_pixel(SIZE / 2, SIZE / 2).0;
        assert!(centre[0] > 200 && centre[1] > 200, "centre {centre:?}");
        assert!(img2 != img, "the built-in warp kept zooming");
        assert!(img3 == expected, "3 built-in ticks match a fresh renderer");
    });
    assert!(error.is_none(), "stale objects reached the GPU: {error:?}");
}

#[test]
fn recompile_on_new_device_matches() {
    let old_gpu = gpu();
    let mut r = renderer(&old_gpu);
    r.write_feedback(&old_gpu, &pattern());
    let checker = |file: &str| {
        let (w, b) = (Rgba([255, 255, 255, 255]), Rgba([0, 0, 0, 255]));
        (file == "checker").then(|| RgbaImage::from_fn(2, 2, |x, y| if x == y { w } else { b }))
    };
    let warp = "return GetMain(uv) * 0.5;";
    let comp = "return mix(GetMain(uv), \
                textureSample(sampler_fc_checker, sampler_fc_checker_s, uv).rgb, 0.5);";
    let set = r.compile_shaders(&old_gpu, Some(warp), Some(comp), RGBA8, &checker);
    assert!(set.errors.is_empty(), "{:?}", set.errors);
    let before = composite_once(&old_gpu, &mut r, &set.comp);
    let before_warped = run_ticks(&old_gpu, &mut r, &set.warp, &set.comp, 1);

    let new_gpu = gpu();
    r.recreate(&new_gpu);
    old_gpu.device.destroy();
    r.write_feedback(&new_gpu, &pattern());
    // The cached images are used: `tex` is not asked for `checker` again.
    let again = r.recompile(&new_gpu, &set, RGBA8, &no_textures);
    assert!(again.errors.is_empty(), "{:?}", again.errors);
    assert_eq!(again.warp_source.as_deref(), Some(warp));
    assert_eq!(again.comp_source.as_deref(), Some(comp));
    assert_eq!(again.user_textures, set.user_textures);
    assert_eq!(again.blur_levels, set.blur_levels);
    assert!(again.texture_images.contains_key("checker"));

    let after = composite_once(&new_gpu, &mut r, &again.comp);
    assert!(after == before, "user comp output matches after recompile");
    let after_warped = run_ticks(&new_gpu, &mut r, &again.warp, &again.comp, 1);
    assert!(
        after_warped == before_warped,
        "user warp output matches after recompile"
    );

    // The loader-thread form, on `RendererShared`.
    let shared = r.shared();
    let again2 = std::thread::spawn(move || shared.recompile(&set, RGBA8, &no_textures))
        .join()
        .unwrap();
    assert!(again2.errors.is_empty(), "{:?}", again2.errors);
}
