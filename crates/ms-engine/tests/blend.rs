//! Preset blending (spec §7.1): blend patterns and per-vertex weights, mesh
//! mixing, and the warp/draw/composite orchestration of a transition.
//!
//! The GPU tests run on the machine's real adapter, or on WARP with
//! `MS_FORCE_WARP=1`.

use image::RgbaImage;
use ms_analysis::AudioFrame;
use ms_engine::{
    blend_weights, composite_layers, mix_meshes, render_blend, BlendPattern, BlendScratch, Globals,
    PresetInstance, TickInput,
};
use ms_expr::Rng;
use ms_preset::{compile_preset, parse};
use ms_render::{
    assert_golden, golden, read_texture, FeedbackFormat, Gpu, MeshData, RenderConfig, Renderer,
    ShaderSet, Uniforms, WarpLayer,
};
use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;

// ---------------------------------------------------------------- patterns

fn patterns() -> Vec<BlendPattern> {
    vec![
        BlendPattern::Uniform,
        BlendPattern::RadialIn,
        BlendPattern::RadialOut,
        BlendPattern::Sweep { angle: 0.7 },
        BlendPattern::Sweep { angle: 4.0 },
        BlendPattern::Plasma { seed: 12345 },
    ]
}

/// Weight of vertex `(col, row)` in a `cols x rows` grid.
fn at(w: &[f32], cols: u32, col: u32, row: u32) -> f32 {
    w[(row * (cols + 1) + col) as usize]
}

#[test]
fn weights_endpoints() {
    for pat in patterns() {
        let w0 = blend_weights(&pat, 16, 12, 0.0);
        let w1 = blend_weights(&pat, 16, 12, 1.0);
        assert_eq!(w0.len(), 17 * 13, "{pat:?}");
        assert_eq!(w1.len(), 17 * 13, "{pat:?}");
        assert!(w0.iter().all(|&w| w == 0.0), "{pat:?} at p = 0: {w0:?}");
        assert!(w1.iter().all(|&w| w == 1.0), "{pat:?} at p = 1: {w1:?}");
    }
}

#[test]
fn weights_monotonic_in_p() {
    for pat in patterns() {
        let steps: Vec<Vec<f32>> = (0..=10)
            .map(|i| blend_weights(&pat, 16, 12, i as f32 / 10.0))
            .collect();
        for pair in steps.windows(2) {
            for (k, (a, b)) in pair[0].iter().zip(&pair[1]).enumerate() {
                assert!(b >= a, "{pat:?}: vertex {k} went from {a} to {b}");
                assert!((0.0..=1.0).contains(b), "{pat:?}: vertex {k} = {b}");
            }
        }
    }
}

#[test]
fn uniform_equals_p() {
    for p in [0.0, 0.1, 0.25, 0.5, 0.77, 1.0] {
        let w = blend_weights(&BlendPattern::Uniform, 8, 6, p);
        assert!(w.iter().all(|&w| w == p), "p = {p}: {w:?}");
        assert_eq!(BlendPattern::Uniform.weight(0.1, 0.9, p), p);
    }
    // Progress is clamped to [0, 1].
    assert!(blend_weights(&BlendPattern::Uniform, 8, 6, 1.5)
        .iter()
        .all(|&w| w == 1.0));
    assert!(blend_weights(&BlendPattern::Uniform, 8, 6, -0.5)
        .iter()
        .all(|&w| w == 0.0));
}

#[test]
fn radial_out_centre_first() {
    let w = blend_weights(&BlendPattern::RadialOut, 16, 16, 0.5);
    let (centre, corner) = (at(&w, 16, 8, 8), at(&w, 16, 0, 0));
    assert!(centre > corner, "centre {centre}, corner {corner}");
    assert_eq!(centre, 1.0);
    assert_eq!(corner, 0.0);
    // RadialIn is the opposite: the corners go first.
    let w = blend_weights(&BlendPattern::RadialIn, 16, 16, 0.5);
    assert!(at(&w, 16, 0, 0) > at(&w, 16, 8, 8));
    // `weight` and `blend_weights` agree.
    let p = BlendPattern::RadialOut;
    assert_eq!(p.weight(0.5, 0.5, 0.5), 1.0);
    assert_eq!(p.weight(0.0, 0.0, 0.5), 0.0);
}

#[test]
fn weights_match_weight_fn_and_grid_convention() {
    // Vertex (col, row) is at preset x = col / cols, y = 1 - row / rows.
    let pat = BlendPattern::Sweep { angle: 1.1 };
    let (cols, rows) = (10u32, 7u32);
    let w = blend_weights(&pat, cols, rows, 0.4);
    for row in 0..=rows {
        for col in 0..=cols {
            let (x, y) = (col as f32 / cols as f32, 1.0 - row as f32 / rows as f32);
            assert_eq!(at(&w, cols, col, row), pat.weight(x, y, 0.4));
        }
    }
}

#[test]
fn sweep_moves_along_angle() {
    // The sweep starts at the far side of `angle` and moves along it.
    let east = BlendPattern::Sweep { angle: 0.0 };
    assert!(east.weight(0.0, 0.5, 0.5) > east.weight(1.0, 0.5, 0.5));
    let north = BlendPattern::Sweep { angle: FRAC_PI_2 };
    assert!(north.weight(0.5, 0.0, 0.5) > north.weight(0.5, 1.0, 0.5));
}

#[test]
fn plasma_seeded_and_varied() {
    let a = blend_weights(&BlendPattern::Plasma { seed: 1 }, 32, 24, 0.5);
    let a2 = blend_weights(&BlendPattern::Plasma { seed: 1 }, 32, 24, 0.5);
    let b = blend_weights(&BlendPattern::Plasma { seed: 2 }, 32, 24, 0.5);
    assert_eq!(a, a2);
    assert_ne!(a, b);
    let (lo, hi) = a
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &w| (lo.min(w), hi.max(w)));
    assert!(hi - lo > 0.3, "plasma too flat: [{lo}, {hi}]");
}

#[test]
fn random_covers_all_variants() {
    let mut rng = Rng::seeded(99);
    let mut counts = [0usize; 5];
    let mut seeds = std::collections::HashSet::new();
    for _ in 0..1000 {
        let i = match BlendPattern::random(&mut rng) {
            BlendPattern::Uniform => 0,
            BlendPattern::RadialIn => 1,
            BlendPattern::RadialOut => 2,
            BlendPattern::Sweep { angle } => {
                assert!((0.0..TAU).contains(&angle), "angle {angle}");
                3
            }
            BlendPattern::Plasma { seed } => {
                seeds.insert(seed);
                4
            }
        };
        counts[i] += 1;
    }
    assert!(counts.iter().all(|&c| c > 120), "{counts:?}");
    assert!(
        seeds.len() > 100,
        "plasma seeds not varied: {}",
        seeds.len()
    );
}

#[test]
fn non_finite_progress_all_zero() {
    for pat in patterns() {
        for p in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let w = blend_weights(&pat, 8, 6, p);
            assert!(w.iter().all(|&w| w == 0.0), "{pat:?} at p = {p}: {w:?}");
        }
        // Non-finite coordinates give a finite weight in [0, 1].
        let w = pat.weight(f32::NAN, f32::INFINITY, 0.5);
        assert!((0.0..=1.0).contains(&w), "{pat:?}: {w}");
    }
}

// ------------------------------------------------------------ mesh mixing

/// A `cols x rows` mesh whose UV is `scale * identity + offset` and whose
/// colour varies linearly over the grid.
fn mesh(cols: u32, rows: u32, scale: f32, offset: f32) -> MeshData {
    let mut uv = Vec::new();
    let mut color = Vec::new();
    for row in 0..=rows {
        for col in 0..=cols {
            let (u, v) = (col as f32 / cols as f32, row as f32 / rows as f32);
            uv.push([u * scale + offset, v * scale + offset]);
            color.push([u, v, 1.0 - u, 0.5 + offset]);
        }
    }
    MeshData {
        cols,
        rows,
        uv,
        color,
    }
}

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-5
}

#[test]
fn mix_meshes_halfway() {
    let a = mesh(8, 6, 1.0, 0.0);
    let b = mesh(8, 6, 0.5, 0.25);
    let w = vec![0.5; 9 * 7];
    let m = mix_meshes(&a, &b, &w);
    assert_eq!((m.cols, m.rows), (8, 6));
    for k in 0..m.uv.len() {
        for c in 0..2 {
            assert!(near(m.uv[k][c], (a.uv[k][c] + b.uv[k][c]) / 2.0), "{k}");
        }
        for c in 0..4 {
            assert!(near(m.color[k][c], (a.color[k][c] + b.color[k][c]) / 2.0));
        }
    }
    // w = 0 is exactly `a`, w = 1 exactly `b`.
    assert_eq!(mix_meshes(&a, &b, &vec![0.0; 63]), a);
    assert_eq!(mix_meshes(&a, &b, &vec![1.0; 63]), b);
}

#[test]
fn mix_uses_per_vertex_weight() {
    let a = mesh(4, 4, 1.0, 0.0);
    let b = mesh(4, 4, 1.0, 1.0);
    let w: Vec<f32> = (0..25).map(|k| k as f32 / 24.0).collect();
    let m = mix_meshes(&a, &b, &w);
    for (k, wk) in w.iter().enumerate() {
        assert!(near(m.uv[k][0], a.uv[k][0] + wk), "{k}");
    }
}

#[test]
fn mix_different_sizes() {
    let small = mesh(32, 24, 0.5, 0.25);
    let big = mesh(64, 48, 1.0, 0.0);
    let n = 65 * 49;
    for (a, b, w, expect) in [
        (&small, &big, 0.0, &small),
        (&big, &small, 1.0, &small),
        (&small, &big, 1.0, &big),
    ] {
        let m = mix_meshes(a, b, &vec![w; n]);
        assert_eq!((m.cols, m.rows), (64, 48));
        assert_eq!(m.uv.len(), n);
        assert_eq!(m.color.len(), n);
        // The fields are linear, so bilinear resampling reproduces them.
        let want = mesh(64, 48, 1.0, 0.0);
        let scale = expect.uv[expect.uv.len() - 1][0] - expect.uv[0][0];
        let offset = expect.uv[0][0];
        for k in 0..n {
            for c in 0..2 {
                let e = want.uv[k][c] * scale + offset;
                assert!(near(m.uv[k][c], e), "vertex {k}: {:?} vs {e}", m.uv[k]);
            }
            for c in 0..3 {
                assert!(near(m.color[k][c], want.color[k][c]), "vertex {k}");
            }
        }
    }
}

#[test]
fn mix_tolerates_bad_input() {
    let a = mesh(8, 6, 1.0, 0.0);
    let b = mesh(8, 6, 0.5, 0.25);
    // Missing weights count as 0 (outgoing); non-finite weights count as 0.
    assert_eq!(mix_meshes(&a, &b, &[]), a);
    assert_eq!(mix_meshes(&a, &b, &vec![f32::NAN; 63]), a);
    let m = mix_meshes(&a, &b, &[1.0; 5]);
    assert_eq!(m.uv[..5], b.uv[..5]);
    assert_eq!(m.uv[5..], a.uv[5..]);
    // Short arrays and empty meshes do not panic.
    let broken = MeshData {
        cols: 8,
        rows: 6,
        uv: vec![[0.5; 2]; 3],
        color: vec![],
    };
    let m = mix_meshes(&broken, &b, &vec![0.5; 63]);
    assert_eq!((m.uv.len(), m.color.len()), (63, 63));
    let m = mix_meshes(&MeshData::default(), &MeshData::default(), &[]);
    assert_eq!(m.uv.len(), (m.cols as usize + 1) * (m.rows as usize + 1));
}

// ------------------------------------------------------------- GPU blend

const SIZE: u32 = 128;
const RGBA8: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Zooms in by 4% per tick around a red square.
const RED_ZOOM_IN: &str = r#"
[params]
mesh = [32, 24]
warp = 0.0
wave_a = 0.0
[code]
frame = "zoom = 1.04"
[[shape]]
sides = 4
rad = 0.12
r = 1.0
g = 0.0
b = 0.0
a = 1.0
r2 = 1.0
g2 = 0.0
b2 = 0.0
a2 = 0.6
"#;

/// Rotates by 0.05 rad per tick around an off-centre blue hexagon, on a
/// different mesh size (so mixing resamples) and with its own warp shader,
/// which fades red and green faster than blue (so the two warp layers are
/// distinguishable).
const BLUE_ROTATE: &str = r#"
[params]
mesh = [48, 36]
warp = 0.0
wave_a = 0.0
[code]
frame = "rot = 0.05"
[[shape]]
sides = 6
x = 0.72
rad = 0.1
r = 0.0
g = 0.0
b = 1.0
a = 1.0
r2 = 0.0
g2 = 0.2
b2 = 1.0
a2 = 0.6
[shader]
warp = "return GetMain(uv).rgb * vec3f(0.9, 0.93, 0.99);"
"#;

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

/// A preset instance and its compiled shaders.
struct Side {
    inst: PresetInstance,
    set: ShaderSet,
}

fn side(gpu: &Gpu, r: &Renderer, toml: &str, seed: u64) -> Side {
    let (p, warnings) = parse(toml.as_bytes()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let p = compile_preset(p);
    assert!(p.errors.is_empty(), "{:?}", p.errors);
    let shader = &p.source.shader;
    let set = r.compile_shaders(
        gpu,
        shader.warp.as_deref(),
        shader.comp.as_deref(),
        RGBA8,
        &no_textures,
    );
    assert!(set.errors.is_empty(), "{:?}", set.errors);
    Side {
        inst: PresetInstance::new(Arc::new(p), seed, [192, 144]),
        set,
    }
}

fn input(tick: u64, audio: &AudioFrame) -> TickInput<'_> {
    TickInput {
        time: tick as f64 / 60.0,
        fps: 60.0,
        frame: tick,
        progress: 0.0,
        audio,
        aspect: [1.0, 1.0],
        pixels: [SIZE, SIZE],
    }
}

/// One tick of `s` alone: a single full-weight warp layer, then its
/// overlays at full alpha. Returns its uniforms.
fn solo_tick(gpu: &Gpu, r: &mut Renderer, s: &mut Side, g: &mut Globals, tick: u64) -> Uniforms {
    let audio = AudioFrame::default();
    let out = s.inst.tick(&input(tick, &audio), g);
    let weight = vec![1.0; out.mesh.uv.len()];
    r.warp(
        gpu,
        &[WarpLayer {
            mesh: &out.mesh,
            weight: &weight,
            shader: &s.set.warp,
            uniforms: &out.uniforms,
        }],
    );
    r.draw(gpu, &out.draw, 1.0);
    out.uniforms
}

/// One blend tick: both presets tick, then `render_blend`. Returns both
/// uniforms.
#[allow(clippy::too_many_arguments)]
fn blend_tick(
    gpu: &Gpu,
    r: &mut Renderer,
    from: &mut Side,
    to: &mut Side,
    g: &mut Globals,
    tick: u64,
    pattern: &BlendPattern,
    p: f32,
    scratch: &mut BlendScratch,
) -> (Uniforms, Uniforms) {
    let audio = AudioFrame::default();
    let a = from.inst.tick(&input(tick, &audio), g);
    let b = to.inst.tick(&input(tick, &audio), g);
    render_blend(
        r,
        gpu,
        (a, &from.set.warp),
        (b, &to.set.warp),
        pattern,
        p,
        scratch,
    );
    (a.uniforms, b.uniforms)
}

fn target(gpu: &Gpu) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("blend target"),
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

fn composite(gpu: &Gpu, r: &mut Renderer, layers: &[ms_render::CompositeLayer<'_>]) -> RgbaImage {
    let tex = target(gpu);
    let view = tex.create_view(&Default::default());
    r.composite(gpu, &view, RGBA8, layers);
    read_texture(gpu, &tex)
}

/// Fraction of pixels differing by more than the golden channel tolerance.
fn diff_fraction(a: &RgbaImage, b: &RgbaImage) -> f64 {
    assert_eq!(a.dimensions(), b.dimensions());
    let differing = a
        .pixels()
        .zip(b.pixels())
        .filter(|(p, q)| {
            p.0.iter()
                .zip(q.0.iter())
                .any(|(x, y)| x.abs_diff(*y) > golden::MAX_CHANNEL_DELTA)
        })
        .count();
    differing as f64 / (a.width() * a.height()) as f64
}

fn mean_rgb(img: &RgbaImage) -> [f64; 3] {
    let mut s = [0.0; 3];
    for p in img.pixels() {
        for (sum, v) in s.iter_mut().zip(p.0) {
            *sum += f64::from(v);
        }
    }
    let n = f64::from(img.width() * img.height());
    s.map(|v| v / n)
}

#[test]
fn composite_layers_math() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let a = r.compile_shaders(
        &gpu,
        None,
        Some("return vec3f(0.8, 0.2, 0.4);"),
        RGBA8,
        &no_textures,
    );
    let b = r.compile_shaders(
        &gpu,
        None,
        Some("return vec3f(0.0, 1.0, 0.8);"),
        RGBA8,
        &no_textures,
    );
    assert!(a.errors.is_empty() && b.errors.is_empty());
    let u = Uniforms::default();
    let layers = composite_layers((&a.comp, &u), (&b.comp, &u), 0.25);
    assert_eq!(layers[0].alpha, 0.75);
    assert_eq!(layers[1].alpha, 0.25);
    let img = composite(&gpu, &mut r, &layers);
    // 0.75 * A + 0.25 * B = (0.6, 0.4, 0.5).
    let want = [153.0, 102.0, 127.5];
    let px = img.get_pixel(SIZE / 2, SIZE / 2).0;
    for c in 0..3 {
        assert!(
            (f64::from(px[c]) - want[c]).abs() <= 2.0,
            "pixel {px:?}, want {want:?}"
        );
    }
    // Non-finite progress counts as 0: the outgoing layer alone.
    let layers = composite_layers((&a.comp, &u), (&b.comp, &u), f32::NAN);
    assert_eq!((layers[0].alpha, layers[1].alpha), (1.0, 0.0));
}

#[test]
fn blend_golden() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let mut red = side(&gpu, &r, RED_ZOOM_IN, 11);
    let mut blue = side(&gpu, &r, BLUE_ROTATE, 22);
    let mut g = Globals::default();
    let mut scratch = BlendScratch::default();
    // The outgoing preset builds an image first.
    for tick in 0..20 {
        solo_tick(&gpu, &mut r, &mut red, &mut g, tick);
    }
    // A 60-tick blend with RadialOut, stopped at tick 30 (p = 0.5).
    let pattern = BlendPattern::RadialOut;
    let mut last = None;
    for t in 0..=30u64 {
        let p = t as f32 / 60.0;
        let u = blend_tick(
            &gpu,
            &mut r,
            &mut red,
            &mut blue,
            &mut g,
            20 + t,
            &pattern,
            p,
            &mut scratch,
        );
        last = Some((u, p));
    }
    let ((ua, ub), p) = last.unwrap();
    let layers = composite_layers((&red.set.comp, &ua), (&blue.set.comp, &ub), p);
    let img = composite(&gpu, &mut r, &layers);
    let [mr, _, mb] = mean_rgb(&img);
    assert!(
        mr > 2.0 && mb > 2.0,
        "both presets visible: mean {mr} red, {mb} blue"
    );
    assert_golden!(img, "blend_radial_mid");
}

#[test]
fn blend_continuity() {
    let gpu = gpu();
    let mut solo_r = renderer(&gpu);
    let mut blend_r = renderer(&gpu);
    let mut solo = side(&gpu, &solo_r, RED_ZOOM_IN, 11);
    let mut red = side(&gpu, &blend_r, RED_ZOOM_IN, 11);
    let mut blue = side(&gpu, &blend_r, BLUE_ROTATE, 22);
    let (mut g1, mut g2) = (Globals::default(), Globals::default());
    let mut scratch = BlendScratch::default();
    for tick in 0..20 {
        solo_tick(&gpu, &mut solo_r, &mut solo, &mut g1, tick);
        solo_tick(&gpu, &mut blend_r, &mut red, &mut g2, tick);
    }
    // The first frame of a blend (p = 0) equals the outgoing preset's frame
    // alone; so do the frames after it while p stays 0.
    for tick in 20..30 {
        let u = solo_tick(&gpu, &mut solo_r, &mut solo, &mut g1, tick);
        let (ua, ub) = blend_tick(
            &gpu,
            &mut blend_r,
            &mut red,
            &mut blue,
            &mut g2,
            tick,
            &BlendPattern::RadialOut,
            0.0,
            &mut scratch,
        );
        let alone = composite(
            &gpu,
            &mut solo_r,
            &[ms_render::CompositeLayer {
                shader: &solo.set.comp,
                uniforms: &u,
                alpha: 1.0,
            }],
        );
        let layers = composite_layers((&red.set.comp, &ua), (&blue.set.comp, &ub), 0.0);
        let blended = composite(&gpu, &mut blend_r, &layers);
        assert!(mean_rgb(&alone)[0] > 2.0, "the outgoing image is not black");
        let f = diff_fraction(&alone, &blended);
        assert!(
            f <= golden::MAX_DIFF_FRACTION,
            "tick {tick}: {:.3}% of pixels differ",
            f * 100.0
        );
        // The feedback buffers match too.
        let f = diff_fraction(
            &read_texture(&gpu, solo_r.feedback_texture()),
            &read_texture(&gpu, blend_r.feedback_texture()),
        );
        assert!(
            f <= golden::MAX_DIFF_FRACTION,
            "tick {tick}: feedback: {:.3}% differ",
            f * 100.0
        );
    }
}
