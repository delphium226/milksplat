//! Shader prelude, user shader compilation and the composite pass.
//!
//! These run on the machine's real adapter, or on WARP with
//! `MS_FORCE_WARP=1`.

use ms_render::test_support::{test_gpu, TestGpu};
use std::sync::Mutex;

use image::{Rgba, RgbaImage};
use ms_render::{
    assert_golden, read_texture, CompShader, CompositeLayer, FeedbackFormat, Gpu, MeshData,
    RenderConfig, Renderer, RendererShared, ShaderSet, Uniforms, WarpLayer, PRELUDE_WGSL,
};

const SIZE: u32 = 128;
const RGBA8: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A headless GPU holding the GPU test lock (see `ms_render::test_support`).
fn gpu() -> TestGpu {
    test_gpu()
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

/// 128x128 pattern: coloured quadrants, a white cross through the centre and
/// a grid of white dots. A single composite samples texel centres exactly, so
/// hard edges are fine here (no iterated resampling).
fn pattern() -> RgbaImage {
    RgbaImage::from_fn(SIZE, SIZE, |x, y| {
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
    })
}

fn target(gpu: &Gpu, format: wgpu::TextureFormat) -> wgpu::Texture {
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
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Composites `layers` (shader, alpha) with `uniforms` into a fresh target
/// of `format` and reads it back.
fn composite_to(
    gpu: &Gpu,
    r: &mut Renderer,
    format: wgpu::TextureFormat,
    uniforms: &Uniforms,
    layers: &[(&CompShader, f32)],
) -> RgbaImage {
    let tex = target(gpu, format);
    let view = tex.create_view(&Default::default());
    let layers: Vec<CompositeLayer<'_>> = layers
        .iter()
        .map(|&(shader, alpha)| CompositeLayer {
            shader,
            uniforms,
            alpha,
        })
        .collect();
    r.composite(gpu, &view, format, &layers);
    read_texture(gpu, &tex)
}

fn composite(gpu: &Gpu, r: &mut Renderer, shader: &CompShader, uniforms: &Uniforms) -> RgbaImage {
    composite_to(
        gpu,
        r,
        wgpu::TextureFormat::Rgba8Unorm,
        uniforms,
        &[(shader, 1.0)],
    )
}

fn compile(gpu: &Gpu, r: &Renderer, warp: Option<&str>, comp: Option<&str>) -> ShaderSet {
    r.compile_shaders(gpu, warp, comp, RGBA8, &no_textures)
}

fn assert_all_near(img: &RgbaImage, rgb: [u8; 3], tol: u8) {
    for (x, y, p) in img.enumerate_pixels() {
        for c in 0..3 {
            assert!(
                p.0[c].abs_diff(rgb[c]) <= tol,
                "pixel ({x},{y}) = {:?}, want {rgb:?} +/- {tol}",
                p.0
            );
        }
        assert_eq!(p.0[3], 255, "pixel ({x},{y}) alpha");
    }
}

fn assert_images_near(actual: &RgbaImage, expected: &RgbaImage, tol: u8) {
    assert_eq!(actual.dimensions(), expected.dimensions());
    for ((x, y, a), e) in actual.enumerate_pixels().zip(expected.pixels()) {
        for c in 0..4 {
            assert!(
                a.0[c].abs_diff(e.0[c]) <= tol,
                "pixel ({x},{y}) = {:?}, want {:?} +/- {tol}",
                a.0,
                e.0
            );
        }
    }
}

/// Reads an `Rgba16Float` texture back as raw f16 texels (no clamping).
fn read_f16(gpu: &Gpu, tex: &wgpu::Texture) -> Vec<[half::f16; 4]> {
    let (w, h) = (tex.width(), tex.height());
    let padded =
        (w * 8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
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
    buffer.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    gpu.poll_wait().unwrap();
    let data = buffer.get_mapped_range(..).unwrap();
    data.chunks_exact(padded as usize)
        .flat_map(|row| row[..(w * 8) as usize].as_chunks::<8>().0)
        .map(|t| {
            let c = |i: usize| half::f16::from_le_bytes([t[2 * i], t[2 * i + 1]]);
            [c(0), c(1), c(2), c(3)]
        })
        .collect()
}

#[test]
fn uniforms_layout() {
    assert_eq!(std::mem::size_of::<Uniforms>() % 16, 0);

    let module = match naga::front::wgsl::parse_str(PRELUDE_WGSL) {
        Ok(m) => m,
        Err(e) => panic!(
            "prelude does not parse:\n{}",
            e.emit_to_string(PRELUDE_WGSL)
        ),
    };
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).unwrap();
    let (handle, members) = module
        .types
        .iter()
        .find_map(|(h, ty)| match (&ty.name, &ty.inner) {
            (Some(name), naga::TypeInner::Struct { members, .. }) if name == "Uniforms" => {
                Some((h, members))
            }
            _ => None,
        })
        .expect("prelude declares struct Uniforms");

    macro_rules! rust_offsets {
        ($($field:ident),* $(,)?) => {
            vec![$((stringify!($field), std::mem::offset_of!(Uniforms, $field))),*]
        };
    }
    let rust = rust_offsets!(
        time,
        fps,
        frame,
        progress,
        bass,
        mid,
        treb,
        beat,
        bass_att,
        mid_att,
        treb_att,
        beat_strength,
        decay,
        gamma,
        echo_zoom,
        echo_alpha,
        echo_orient,
        invert,
        solarize,
        brighten,
        darken,
        darken_center,
        aspect_x,
        aspect_y,
        texsize,
        rand_frame,
        rand_preset,
        roam_cos,
        roam_sin,
        blur_min,
        blur_max,
        q,
    );
    assert_eq!(members.len(), rust.len(), "member count");

    // Recompute every offset with naga's layouter (WGSL uniform layout rules)
    // and compare it with the IR's offset and the Rust struct.
    let mut offset = 0u32;
    for (member, (name, rust_offset)) in members.iter().zip(&rust) {
        assert_eq!(member.name.as_deref(), Some(*name), "member order");
        let layout = layouter[member.ty];
        offset = layout.alignment.round_up(offset);
        assert_eq!(member.offset, offset, "naga IR offset of {name}");
        assert_eq!(offset as usize, *rust_offset, "offset of {name}");
        offset += layout.size;
    }
    assert_eq!(
        layouter[handle].size as usize,
        std::mem::size_of::<Uniforms>(),
        "struct size"
    );
}

#[test]
fn builtin_comp_identity_golden() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    r.write_feedback(&gpu, &pattern());
    let set = compile(&gpu, &r, None, None);
    assert!(set.errors.is_empty(), "{:?}", set.errors);
    let img = composite(&gpu, &mut r, &set.comp, &Uniforms::default());
    assert_images_near(&img, &pattern(), 1);
    assert_golden!(img, "comp_identity");
}

#[test]
fn user_comp_applies() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    r.write_feedback(&gpu, &pattern());
    let set = compile(&gpu, &r, None, Some("return vec3f(1.0, 0.0, 0.0);"));
    assert!(set.errors.is_empty(), "{:?}", set.errors);
    let img = composite(&gpu, &mut r, &set.comp, &Uniforms::default());
    assert_all_near(&img, [255, 0, 0], 0);
    // A second target format gets its own pipeline.
    let bgra = composite_to(
        &gpu,
        &mut r,
        wgpu::TextureFormat::Bgra8Unorm,
        &Uniforms::default(),
        &[(&set.comp, 1.0)],
    );
    assert_all_near(&bgra, [255, 0, 0], 0);
}

#[test]
fn syntax_error_falls_back() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    r.write_feedback(&gpu, &pattern());
    let set = compile(&gpu, &r, None, Some("return vec3f(;"));
    assert!(!set.errors.is_empty());
    let img = composite(&gpu, &mut r, &set.comp, &Uniforms::default());
    assert_golden!(img, "comp_identity");
}

#[test]
fn validation_error_falls_back() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    r.write_feedback(&gpu, &pattern());
    let set = compile(&gpu, &r, None, Some("return 1.0;"));
    assert!(!set.errors.is_empty());
    let img = composite(&gpu, &mut r, &set.comp, &Uniforms::default());
    assert_golden!(img, "comp_identity");
}

#[test]
fn pipeline_error_falls_back() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    r.write_feedback(&gpu, &pattern());
    // A valid module (naga accepts it) whose pipeline wgpu rejects: the body
    // closes `user_comp` early and uses a group 2 binding that the pipeline
    // layout does not have.
    let body = "return extra.xyz;\n}\n\
                @group(2) @binding(0) var<uniform> extra: vec4f;\n\
                fn ms_unused() {";
    let set = compile(&gpu, &r, None, Some(body));
    assert_eq!(set.errors.len(), 1, "{:?}", set.errors);
    assert!(
        set.errors[0].contains("pipeline creation failed"),
        "{}",
        set.errors[0]
    );
    let img = composite(&gpu, &mut r, &set.comp, &Uniforms::default());
    assert_golden!(img, "comp_identity");
}

#[test]
fn pipeline_error_off_thread_falls_back() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    r.write_feedback(&gpu, &pattern());
    let shared = r.shared();
    // The pipeline error is raised on the spawned thread and must be caught
    // by that thread's error scopes.
    let set = std::thread::spawn(move || {
        let body = "return extra.xyz;\n}\n\
                    @group(2) @binding(0) var<uniform> extra: vec4f;\n\
                    fn ms_unused() {";
        shared.compile_shaders(None, Some(body), RGBA8, &no_textures)
    })
    .join()
    .unwrap();
    assert_eq!(set.errors.len(), 1, "{:?}", set.errors);
    assert!(
        set.errors[0].contains("pipeline creation failed"),
        "{}",
        set.errors[0]
    );
    let img = composite(&gpu, &mut r, &set.comp, &Uniforms::default());
    assert_golden!(img, "comp_identity");
}

#[test]
fn comp_pipeline_prepared_for_target_format() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let bgra = wgpu::TextureFormat::Bgra8Unorm;
    let user = r.compile_shaders(
        &gpu,
        None,
        Some("return vec3f(1.0, 0.0, 0.0);"),
        bgra,
        &no_textures,
    );
    let builtin = r.compile_shaders(&gpu, None, None, bgra, &no_textures);
    assert!(user.errors.is_empty(), "{:?}", user.errors);
    assert!(builtin.errors.is_empty(), "{:?}", builtin.errors);

    // Compositing into the format the set was compiled for creates nothing.
    let u = Uniforms::default();
    let img = composite_to(&gpu, &mut r, bgra, &u, &[(&user.comp, 1.0)]);
    assert_all_near(&img, [255, 0, 0], 0);
    composite_to(&gpu, &mut r, bgra, &u, &[(&builtin.comp, 1.0)]);
    assert_eq!(user.comp.render_thread_pipeline_creations(), 0);
    assert_eq!(builtin.comp.render_thread_pipeline_creations(), 0);

    // Another format still works, through the counted safety net.
    let img = composite_to(&gpu, &mut r, RGBA8, &u, &[(&user.comp, 1.0)]);
    assert_all_near(&img, [255, 0, 0], 0);
    assert_eq!(user.comp.render_thread_pipeline_creations(), 1);
}

#[test]
fn float_target_is_not_clamped() {
    let gpu = gpu();
    let mut r = Renderer::new(
        &gpu,
        RenderConfig {
            internal_size: [SIZE, SIZE],
            format: FeedbackFormat::Fp16,
        },
    );
    let two = half::f16::from_f32(2.0);
    let one = half::f16::from_f32(1.0);
    r.write_feedback_f16(&gpu, &vec![[two, two, two, one]; (SIZE * SIZE) as usize]);
    let float = wgpu::TextureFormat::Rgba16Float;
    let u = Uniforms::default();
    let user = r.compile_shaders(
        &gpu,
        None,
        Some("return vec3f(2.0, 0.5, 70.0);"),
        float,
        &no_textures,
    );
    let builtin = r.compile_shaders(&gpu, None, None, float, &no_textures);
    assert!(user.errors.is_empty(), "{:?}", user.errors);

    let tex = target(&gpu, float);
    let view = tex.create_view(&Default::default());
    for (set, want) in [(&user, [2.0, 0.5, 64.0]), (&builtin, [2.0, 2.0, 2.0])] {
        let layer = CompositeLayer {
            shader: &set.comp,
            uniforms: &u,
            alpha: 1.0,
        };
        r.composite(&gpu, &view, float, &[layer]);
        let texel = read_f16(&gpu, &tex)[0];
        // Sanitised to [0, 64], but not clamped to 1.
        assert_eq!(texel.map(f32::from)[..3], want);
    }

    // The same built-in clamps on a unorm target.
    let img = composite_to(&gpu, &mut r, RGBA8, &u, &[(&builtin.comp, 1.0)]);
    assert_all_near(&img, [255, 255, 255], 0);
}

#[test]
fn fallen_back_stage_does_not_drive_blur_or_textures() {
    let gpu = gpu();
    let r = renderer(&gpu);
    let requested = Mutex::new(Vec::new());
    let tex = |file: &str| {
        requested.lock().unwrap().push(file.to_string());
        Some(RgbaImage::from_pixel(1, 1, Rgba([255; 4])))
    };
    // The warp fails naga: its blur level and texture must not count.
    let set = r.compile_shaders(
        &gpu,
        Some("return textureSample(sampler_rock, sampler_rock_s, uv).rgb + GetBlur3(uv) +;"),
        Some("return GetBlur1(uv) + textureSample(sampler_pc_sky, sampler_pc_sky_s, uv).rgb;"),
        RGBA8,
        &tex,
    );
    assert_eq!(set.errors.len(), 1, "{:?}", set.errors);
    assert!(
        set.errors[0].starts_with("warp shader"),
        "{}",
        set.errors[0]
    );
    assert_eq!(set.blur_levels, 1);
    assert_eq!(set.user_textures, ["pc_sky"]);
    assert_eq!(*requested.lock().unwrap(), ["sky"]);

    // A pipeline-stage failure (after naga) does not count for blur levels.
    let set = r.compile_shaders(
        &gpu,
        None,
        Some(
            "return GetBlur2(uv) + extra.xyz;\n}\n\
             @group(2) @binding(0) var<uniform> extra: vec3f;\n\
             fn ms_unused() {",
        ),
        RGBA8,
        &no_textures,
    );
    assert_eq!(set.errors.len(), 1, "{:?}", set.errors);
    assert!(
        set.errors[0].contains("pipeline creation failed"),
        "{}",
        set.errors[0]
    );
    assert_eq!(set.blur_levels, 0);
}

#[test]
fn empty_texture_name_is_an_error() {
    let gpu = gpu();
    let r = renderer(&gpu);
    let set = compile(
        &gpu,
        &r,
        None,
        Some("return textureSample(sampler_fc_, sampler_fc__s, uv).rgb;"),
    );
    assert!(set.user_textures.is_empty());
    assert!(
        set.errors[0].contains("empty texture name"),
        "{:?}",
        set.errors
    );
    // The identifier stays undeclared, so the stage falls back too.
    assert!(
        set.errors
            .iter()
            .any(|e| e.contains("no definition in scope")),
        "{:?}",
        set.errors
    );
}

#[test]
fn errors_report_user_line_numbers() {
    let gpu = gpu();
    let r = renderer(&gpu);
    let set = compile(
        &gpu,
        &r,
        None,
        Some("let a = 1.0;\nlet b = a * 2.0;\nreturn vec3f(;"),
    );
    assert_eq!(set.errors.len(), 1, "{:?}", set.errors);
    assert!(set.errors[0].contains("comp"), "{}", set.errors[0]);
    assert!(set.errors[0].contains("line 3"), "{}", set.errors[0]);

    // Wrong return type: naga points at the returned expression.
    let set = compile(
        &gpu,
        &r,
        Some("let a = vec3f(1.0);\nreturn 2.0 * a.x;"),
        None,
    );
    assert_eq!(set.errors.len(), 1, "{:?}", set.errors);
    assert!(set.errors[0].contains("warp"), "{}", set.errors[0]);
    assert!(set.errors[0].contains("line 2"), "{}", set.errors[0]);
}

#[test]
fn user_texture_bound() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let requested = Mutex::new(Vec::new());
    let tex = |file: &str| {
        requested.lock().unwrap().push(file.to_string());
        let (w, b) = (Rgba([255, 255, 255, 255]), Rgba([0, 0, 0, 255]));
        (file == "checker").then(|| RgbaImage::from_fn(2, 2, |x, y| if x == y { w } else { b }))
    };
    let set = r.compile_shaders(
        &gpu,
        None,
        Some("return textureSample(sampler_fc_checker, sampler_fc_checker_s, uv).rgb;"),
        RGBA8,
        &tex,
    );
    assert!(set.errors.is_empty(), "{:?}", set.errors);
    assert_eq!(set.user_textures, ["fc_checker"]);
    assert_eq!(*requested.lock().unwrap(), ["checker"]);

    let img = composite(&gpu, &mut r, &set.comp, &Uniforms::default());
    // Clamp addressing: each quadrant's outer part is exactly one texel.
    for (x, y, want) in [(8, 8, 255), (119, 8, 0), (8, 119, 0), (119, 119, 255)] {
        let p = img.get_pixel(x, y).0;
        assert_eq!(p, [want, want, want, 255], "pixel ({x},{y})");
    }
}

#[test]
fn missing_user_texture_records_error_but_renders() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let set = compile(
        &gpu,
        &r,
        None,
        Some("return textureSample(sampler_nothere, sampler_nothere_s, uv).rgb * 0.5;"),
    );
    assert_eq!(set.user_textures, ["nothere"]);
    assert_eq!(set.errors.len(), 1, "{:?}", set.errors);
    assert!(set.errors[0].contains("nothere"), "{}", set.errors[0]);
    // The user shader runs against a white placeholder.
    let img = composite(&gpu, &mut r, &set.comp, &Uniforms::default());
    assert_all_near(&img, [128, 128, 128], 1);
}

#[test]
fn too_many_user_textures_falls_back() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    r.write_feedback(&gpu, &pattern());
    let body = |n: usize| {
        let mut body = String::from(
            "var c = vec3f(0.0);
",
        );
        for i in 0..n {
            body += &format!(
                "c += textureSample(sampler_t{i}, sampler_t{i}_s, uv).rgb;
"
            );
        }
        body + "return c;"
    };
    let tex = |_: &str| Some(RgbaImage::from_pixel(1, 1, Rgba([10, 10, 10, 255])));

    // Exactly 16 fit (and stay within the device's per-stage limits).
    let set = r.compile_shaders(&gpu, None, Some(&body(16)), RGBA8, &tex);
    assert!(set.errors.is_empty(), "{:?}", set.errors);
    assert_eq!(set.user_textures.len(), 16);
    let img = composite(&gpu, &mut r, &set.comp, &Uniforms::default());
    assert_all_near(&img, [160, 160, 160], 2);

    // A 17th has no slot: the stage falls back, so nothing is bound.
    let set = r.compile_shaders(&gpu, None, Some(&body(17)), RGBA8, &tex);
    assert!(set.user_textures.is_empty(), "{:?}", set.user_textures);
    assert!(
        set.errors.iter().any(|e| e.contains("16")),
        "{:?}",
        set.errors
    );
    let img = composite(&gpu, &mut r, &set.comp, &Uniforms::default());
    assert_golden!(img, "comp_identity");
}

#[test]
fn blur_levels_detected() {
    let gpu = gpu();
    let r = renderer(&gpu);
    let set = compile(&gpu, &r, None, Some("return GetBlur2(uv);"));
    assert!(set.errors.is_empty(), "{:?}", set.errors);
    assert_eq!(set.blur_levels, 2);

    let set = compile(
        &gpu,
        &r,
        Some("return textureSample(sampler_blur3, s_fc, uv).rgb + GetBlur1(uv);"),
        Some("// GetBlur2 only in a comment\nreturn GetMain(uv);"),
    );
    assert!(set.errors.is_empty(), "{:?}", set.errors);
    assert_eq!(set.blur_levels, 3);

    assert_eq!(compile(&gpu, &r, None, None).blur_levels, 0);
}

#[test]
fn compile_off_thread() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let shared: RendererShared = r.shared();
    let set = std::thread::spawn(move || {
        shared.compile_shaders(
            None,
            Some("return vec3f(0.0, 1.0, 0.0);"),
            RGBA8,
            &no_textures,
        )
    })
    .join()
    .unwrap();
    assert!(set.errors.is_empty(), "{:?}", set.errors);
    let img = composite(&gpu, &mut r, &set.comp, &Uniforms::default());
    assert_all_near(&img, [0, 255, 0], 0);
}

#[test]
fn user_warp_applies() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    r.write_feedback(
        &gpu,
        &RgbaImage::from_pixel(SIZE, SIZE, Rgba([200, 100, 40, 255])),
    );
    let set = compile(&gpu, &r, Some("return GetMain(uv).rgb * 0.5;"), None);
    assert!(set.errors.is_empty(), "{:?}", set.errors);

    let (cols, rows) = (8, 6);
    let mut uv = Vec::new();
    for y in 0..=rows {
        for x in 0..=cols {
            uv.push([x as f32 / cols as f32, y as f32 / rows as f32]);
        }
    }
    let n = uv.len();
    let mesh = MeshData {
        cols,
        rows,
        uv,
        color: vec![[1.0; 4]; n],
    };
    let weight = vec![1.0; n];
    r.warp(
        &gpu,
        &[WarpLayer {
            mesh: &mesh,
            weight: &weight,
            shader: &set.warp,
            uniforms: &Uniforms::default(),
        }],
    );
    let img = read_texture(&gpu, r.feedback_texture());
    assert_all_near(&img, [100, 50, 20], 1);
}

#[test]
fn composite_layers_blend() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let red = compile(&gpu, &r, None, Some("return vec3f(1.0, 0.0, 0.0);"));
    let green = compile(&gpu, &r, None, Some("return vec3f(0.0, 1.0, 0.0);"));
    // The first layer is drawn opaque whatever its alpha; later layers blend
    // over it with theirs.
    let img = composite_to(
        &gpu,
        &mut r,
        wgpu::TextureFormat::Rgba8Unorm,
        &Uniforms::default(),
        &[(&red.comp, 0.25), (&green.comp, 0.5)],
    );
    assert_all_near(&img, [128, 128, 0], 1);
}

#[test]
fn builtin_comp_effects() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let set = compile(&gpu, &r, None, None);
    let grey = RgbaImage::from_pixel(SIZE, SIZE, Rgba([64, 64, 64, 255]));
    r.write_feedback(&gpu, &grey);
    let run = |r: &mut Renderer, u: Uniforms| composite(&gpu, r, &set.comp, &u);

    // invert: 1 - c.
    let img = run(
        &mut r,
        Uniforms {
            invert: 1.0,
            ..Uniforms::default()
        },
    );
    assert_all_near(&img, [191, 191, 191], 1);
    // gamma 2: pow(c, 1/2) = sqrt(64/255) = 0.501.
    let img = run(
        &mut r,
        Uniforms {
            gamma: 2.0,
            ..Uniforms::default()
        },
    );
    assert_all_near(&img, [128, 128, 128], 1);
    // darken: c^2 = 0.063.
    let img = run(
        &mut r,
        Uniforms {
            darken: 1.0,
            ..Uniforms::default()
        },
    );
    assert_all_near(&img, [16, 16, 16], 1);
    // solarize: 4c(1-c) = 0.751.
    let img = run(
        &mut r,
        Uniforms {
            solarize: 1.0,
            ..Uniforms::default()
        },
    );
    assert_all_near(&img, [192, 192, 192], 1);
    // brighten: sqrt(c) = 0.501.
    let img = run(
        &mut r,
        Uniforms {
            brighten: 1.0,
            ..Uniforms::default()
        },
    );
    assert_all_near(&img, [128, 128, 128], 1);

    // darken_center on white: the centre pixel (offset 1/128 per axis in
    // clip units, r = 0.01105) is scaled by 1 - 3/32 * (1 - r/0.05) = 0.927
    // -> 236; outside radius 0.05 nothing changes.
    r.write_feedback(&gpu, &RgbaImage::from_pixel(SIZE, SIZE, Rgba([255; 4])));
    let img = run(
        &mut r,
        Uniforms {
            darken_center: 1.0,
            ..Uniforms::default()
        },
    );
    for (x, y, want) in [(64, 64, 236), (63, 63, 236), (0, 0, 255), (80, 64, 255)] {
        let p = img.get_pixel(x, y).0;
        for c in 0..3 {
            assert!(
                p[c].abs_diff(want) <= 1,
                "pixel ({x},{y}) = {p:?}, want {want}"
            );
        }
    }

    // Full echo zoomed 2x about the centre (no flip): pixel 16 samples texel
    // 40.25 (inside a dot, white) and pixel 24 samples texel 44.25 (red);
    // unzoomed they are red and a dot respectively.
    r.write_feedback(&gpu, &pattern());
    let img = run(
        &mut r,
        Uniforms {
            echo_alpha: 1.0,
            echo_zoom: 2.0,
            ..Uniforms::default()
        },
    );
    assert_eq!(img.get_pixel(16, 16).0, [255, 255, 255, 255]);
    assert_eq!(img.get_pixel(24, 24).0, [220, 40, 40, 255]);
    assert_eq!(pattern().get_pixel(16, 16).0, [220, 40, 40, 255]);
    assert_eq!(pattern().get_pixel(24, 24).0, [255, 255, 255, 255]);

    // Full echo flipped left-right: the image is mirrored.
    r.write_feedback(&gpu, &pattern());
    let img = run(
        &mut r,
        Uniforms {
            echo_alpha: 1.0,
            echo_orient: 1.0,
            ..Uniforms::default()
        },
    );
    let mirrored = image::imageops::flip_horizontal(&pattern());
    assert_images_near(&img, &mirrored, 1);
}

#[test]
fn warm_up_draws_off_thread_and_rejects_stale_sets() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let shared = r.shared();
    let (set, warmed) = std::thread::spawn(move || {
        let tex = |_: &str| Some(RgbaImage::from_pixel(2, 2, Rgba([10, 200, 30, 255])));
        let set = shared.compile_shaders(
            Some("return GetMain(uv).rgb * 0.9;"),
            Some("return textureSample(sampler_fc_checker, sampler_fc_checker_s, uv).rgb;"),
            RGBA8,
            &tex,
        );
        let warmed =
            [FeedbackFormat::Rgba8, FeedbackFormat::Fp16].map(|f| shared.warm_up(&set, f, RGBA8));
        (set, warmed)
    })
    .join()
    .unwrap();
    assert!(set.errors.is_empty(), "{:?}", set.errors);
    for w in warmed {
        assert_eq!(w, Ok(()));
    }
    // The built-in set warms up too.
    let builtin = r.compile_shaders(&gpu, None, None, RGBA8, &no_textures);
    assert_eq!(
        r.shared().warm_up(&builtin, FeedbackFormat::Rgba8, RGBA8),
        Ok(())
    );

    // After `recreate`, the old set belongs to another generation.
    r.recreate(&gpu);
    let e = r
        .shared()
        .warm_up(&set, FeedbackFormat::Rgba8, RGBA8)
        .unwrap_err();
    assert!(e.contains("generation"), "{e}");
}
