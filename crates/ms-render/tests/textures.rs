//! Blur chain, noise textures, the texture resolver and the spectrum
//! texture.
//!
//! GPU tests run on the machine's real adapter, or on WARP with
//! `MS_FORCE_WARP=1`.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use image::{Rgba, RgbaImage};
use ms_expr::Rng;
use ms_render::{
    assert_golden, noise_textures, read_texture, CompositeLayer, FeedbackFormat, Gpu, NoiseSet,
    RenderConfig, Renderer, TextureResolver, Uniforms,
};

const RGBA8: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn gpu() -> Gpu {
    match Gpu::headless() {
        Ok(gpu) => gpu,
        Err(e) => panic!("Gpu::headless() failed (no hardware adapter and no WARP): {e}"),
    }
}

fn renderer(gpu: &Gpu, w: u32, h: u32) -> Renderer {
    Renderer::new(
        gpu,
        RenderConfig {
            internal_size: [w, h],
            format: FeedbackFormat::Rgba8,
        },
    )
}

fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbaImage {
    RgbaImage::from_pixel(w, h, Rgba([rgb[0], rgb[1], rgb[2], 255]))
}

fn assert_all_near(img: &RgbaImage, rgb: [u8; 3], tol: u8, what: &str) {
    for (x, y, p) in img.enumerate_pixels() {
        for c in 0..3 {
            assert!(
                p.0[c].abs_diff(rgb[c]) <= tol,
                "{what}: pixel ({x},{y}) = {:?}, want {rgb:?} +/- {tol}",
                p.0
            );
        }
    }
}

fn blur_image(gpu: &Gpu, r: &Renderer, level: u8) -> RgbaImage {
    let tex = r
        .blur_texture(level)
        .unwrap_or_else(|| panic!("no blur texture for level {level}"));
    read_texture(gpu, tex)
}

/// Compiles `comp` and composites it into a fresh `w`x`h` `Rgba8Unorm`
/// target, returning the image.
fn run_comp(gpu: &Gpu, r: &mut Renderer, comp: &str, w: u32, h: u32) -> RgbaImage {
    let set = r.compile_shaders(gpu, None, Some(comp), RGBA8, &|_| None);
    assert!(set.errors.is_empty(), "compile errors: {:?}", set.errors);
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("textures test target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RGBA8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let uniforms = Uniforms::default();
    r.composite(
        gpu,
        &view,
        RGBA8,
        &[CompositeLayer {
            shader: &set.comp,
            uniforms: &uniforms,
            alpha: 1.0,
        }],
    );
    read_texture(gpu, &tex)
}

/// Reads every texel of a 2D or 3D `Rgba8Unorm` texture, row-major, slice
/// by slice.
fn read_rgba8_bytes(gpu: &Gpu, tex: &wgpu::Texture) -> Vec<u8> {
    let (w, h, d) = (tex.width(), tex.height(), tex.depth_or_array_layers());
    let row = w * 4;
    let padded =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise readback"),
        size: u64::from(padded) * u64::from(h) * u64::from(d),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
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
    let (tx, rx) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    gpu.poll_wait().unwrap();
    rx.recv().unwrap().unwrap();
    let view = buffer.get_mapped_range(..).unwrap();
    let mut out = Vec::with_capacity((row * h * d) as usize);
    for padded_row in view.chunks_exact(padded as usize) {
        out.extend_from_slice(&padded_row[..row as usize]);
    }
    out
}

// ---------------------------------------------------------------------------
// Blur chain

#[test]
fn blur_levels_sizes() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 512, 512);
    let sizes = |r: &Renderer| {
        (1..=3)
            .map(|l| {
                let t = r.blur_texture(l).unwrap();
                (t.width(), t.height(), t.format())
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        sizes(&r),
        [(256, 256, RGBA8), (128, 128, RGBA8), (64, 64, RGBA8)]
    );
    assert!(r.blur_texture(0).is_none());
    assert!(r.blur_texture(4).is_none());

    // Recreated on resize (odd sizes round down) and on a format change.
    r.resize(&gpu, [100, 60]);
    assert_eq!(
        sizes(&r),
        [(50, 30, RGBA8), (25, 15, RGBA8), (12, 7, RGBA8)]
    );
    r.set_format(&gpu, FeedbackFormat::Fp16);
    let f16 = wgpu::TextureFormat::Rgba16Float;
    assert_eq!(sizes(&r), [(50, 30, f16), (25, 15, f16), (12, 7, f16)]);
    // Fp16 blur works and reads back.
    r.write_feedback(&gpu, &solid(100, 60, [10, 200, 90]));
    r.blur(&gpu, 3);
    for level in 1..=3 {
        assert_all_near(&blur_image(&gpu, &r, level), [10, 200, 90], 1, "fp16 blur");
    }
}

#[test]
fn blur_only_requested() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 128, 128);
    let first = [200, 100, 50];
    r.write_feedback(&gpu, &solid(128, 128, first));
    r.blur(&gpu, 3);
    // A uniform image stays uniform under a normalised, clamped Gaussian.
    for level in 1..=3 {
        assert_all_near(&blur_image(&gpu, &r, level), first, 1, "blur(3)");
    }

    let second = [20, 220, 120];
    r.write_feedback(&gpu, &solid(128, 128, second));
    r.blur(&gpu, 1);
    assert_all_near(&blur_image(&gpu, &r, 1), second, 1, "level 1 after blur(1)");
    // Levels 2 and 3 keep the sentinel from the first blur.
    assert_all_near(&blur_image(&gpu, &r, 2), first, 1, "level 2 after blur(1)");
    assert_all_near(&blur_image(&gpu, &r, 3), first, 1, "level 3 after blur(1)");

    // 0 computes nothing; more than 3 is clamped to 3.
    r.write_feedback(&gpu, &solid(128, 128, [0, 0, 0]));
    r.blur(&gpu, 0);
    assert_all_near(&blur_image(&gpu, &r, 1), second, 1, "level 1 after blur(0)");
    r.blur(&gpu, 9);
    for level in 1..=3 {
        assert_all_near(&blur_image(&gpu, &r, level), [0, 0, 0], 1, "blur(9)");
    }
}

#[test]
fn blur_golden() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 512, 512);
    // A white disc of radius 16 centred at (144, 336): off-centre, so a
    // flipped or transposed chain would show.
    let img = RgbaImage::from_fn(512, 512, |x, y| {
        let (dx, dy) = (x as f32 + 0.5 - 144.0, y as f32 + 0.5 - 336.0);
        if dx * dx + dy * dy <= 16.0 * 16.0 {
            Rgba([255, 255, 255, 255])
        } else {
            Rgba([0, 0, 0, 255])
        }
    });
    r.write_feedback(&gpu, &img);
    r.blur(&gpu, 3);
    let blur3 = blur_image(&gpu, &r, 3);
    assert_eq!(blur3.dimensions(), (64, 64));
    // The blob's peak is where the disc was: (144, 336) / 8 = (18, 42), the
    // corner shared by texels 17..18 and 41..42.
    let (px, py, peak) = blur3
        .enumerate_pixels()
        .max_by_key(|(_, _, p)| p.0[0])
        .map(|(x, y, p)| (x, y, p.0[0]))
        .unwrap();
    assert!(
        (17..=18).contains(&px) && (41..=42).contains(&py),
        "peak at ({px},{py})"
    );
    assert!((40..255).contains(&peak), "peak value {peak}");
    // Far from the disc it is black.
    assert_eq!(blur3.get_pixel(60, 4).0[..3], [0, 0, 0]);
    assert_golden!(blur3, "blur3");
}

#[test]
fn comp_get_blur1_sees_blurred_feedback() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 128, 128);
    // Left half white, right half black.
    let img = RgbaImage::from_fn(128, 128, |x, _| {
        let v = if x < 64 { 255 } else { 0 };
        Rgba([v, v, v, 255])
    });
    r.write_feedback(&gpu, &img);
    r.blur(&gpu, 1);
    let out = run_comp(&gpu, &mut r, "return GetBlur1(uv);", 128, 128);
    let at = |x: u32| out.get_pixel(x, 64).0[0];
    assert!(at(4) >= 254, "far left {}", at(4));
    assert!(at(123) <= 1, "far right {}", at(123));
    // The edge is softened (the placeholder was black everywhere, and the
    // unblurred feedback would be a hard step).
    for x in [60, 63, 64, 67] {
        assert!((20..=235).contains(&at(x)), "edge pixel {x} = {}", at(x));
    }
    assert!(at(60) > at(63) && at(63) > at(64) && at(64) > at(67));

    // After a resize the recreated levels are the ones bound.
    r.resize(&gpu, [96, 96]);
    r.write_feedback(&gpu, &solid(96, 96, [100, 100, 100]));
    r.blur(&gpu, 1);
    let out = run_comp(&gpu, &mut r, "return GetBlur1(uv);", 32, 32);
    assert_all_near(&out, [100, 100, 100], 1, "GetBlur1 after resize");
}

// ---------------------------------------------------------------------------
// Noise

fn noise_bytes(gpu: &Gpu, set: &NoiseSet) -> Vec<Vec<u8>> {
    [&set.lq, &set.mq, &set.hq, &set.vol_lq, &set.vol_hq]
        .into_iter()
        .map(|t| read_rgba8_bytes(gpu, t))
        .collect()
}

#[test]
fn noise_deterministic() {
    let gpu = gpu();
    let a = noise_textures(&gpu);
    let b = noise_textures(&gpu);

    let shapes: Vec<_> = [&a.lq, &a.mq, &a.hq, &a.vol_lq, &a.vol_hq]
        .into_iter()
        .map(|t| {
            (
                t.width(),
                t.height(),
                t.depth_or_array_layers(),
                t.dimension(),
                t.format(),
            )
        })
        .collect();
    use wgpu::TextureDimension::{D2, D3};
    assert_eq!(
        shapes,
        [
            (32, 32, 1, D2, RGBA8),
            (64, 64, 1, D2, RGBA8),
            (256, 256, 1, D2, RGBA8),
            (32, 32, 32, D3, RGBA8),
            (64, 64, 64, D3, RGBA8),
        ]
    );

    let bytes_a = noise_bytes(&gpu, &a);
    let bytes_b = noise_bytes(&gpu, &b);
    assert_eq!(bytes_a, bytes_b, "two calls differ");

    // It is noise: about uniform, channels independent, textures distinct.
    for bytes in &bytes_a {
        let mean = bytes.iter().map(|&v| f64::from(v)).sum::<f64>() / bytes.len() as f64;
        assert!((120.0..135.0).contains(&mean), "mean {mean}");
        assert!(bytes.iter().any(|&v| v < 8) && bytes.iter().any(|&v| v > 247));
        assert!(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .any(|t| t[0] != t[1] || t[1] != t[2] || t[2] != t[3]));
    }
    assert_ne!(bytes_a[0][..64], bytes_a[1][..64]);
}

#[test]
fn noise_bound_in_shaders() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 64, 64);
    let set = noise_textures(&gpu);
    let lq = read_rgba8_bytes(&gpu, &set.lq);
    let vol = read_rgba8_bytes(&gpu, &set.vol_lq);

    // 128x128 target over the 32x32 texture: each 4x4 block is one texel.
    let out = run_comp(
        &gpu,
        &mut r,
        "return textureSampleLevel(sampler_noise_lq, s_pc, uv, 0.0).rgb;",
        128,
        128,
    );
    for (x, y, p) in out.enumerate_pixels() {
        let i = (((y / 4) * 32 + x / 4) * 4) as usize;
        assert_eq!(p.0[..3], lq[i..i + 3], "noise_lq at ({x},{y})");
    }

    // Slice z = 3 of the 32^3 volume.
    let out = run_comp(
        &gpu,
        &mut r,
        "return textureSampleLevel(sampler_noisevol_lq, s_pc, vec3f(uv, 3.5 / 32.0), 0.0).rgb;",
        32,
        32,
    );
    for (x, y, p) in out.enumerate_pixels() {
        let i = (((3 * 32 + y) * 32 + x) * 4) as usize;
        assert_eq!(p.0[..3], vol[i..i + 3], "noisevol_lq at ({x},{y},3)");
    }
}

// ---------------------------------------------------------------------------
// Spectrum

#[test]
fn spectrum_uploaded() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 64, 64);
    // Point, linear and textureLoad reads of texel 10 (x = 10/512 is its
    // left edge; sample its centre).
    let comp = "let x = vec2f(10.5 / 512.0, 0.5);\n\
                return vec3f(textureSample(sampler_spectrum, s_pc, x).r,\n\
                             textureSample(sampler_spectrum, s_fc, x).r,\n\
                             textureLoad(sampler_spectrum, vec2i(10, 0), 0).r);";

    let before = run_comp(&gpu, &mut r, comp, 8, 8);
    assert_all_near(&before, [0, 0, 0], 0, "spectrum before upload");

    let mut spectrum = [0.25f32; 512];
    spectrum[10] = 0.75;
    r.set_spectrum(&gpu, &spectrum);
    let out = run_comp(&gpu, &mut r, comp, 8, 8);
    assert_all_near(&out, [191, 191, 191], 1, "spectrum[10] = 0.75");

    // Updated in place; non-finite values read as 0.
    spectrum[10] = 0.5;
    spectrum[11] = f32::NAN;
    r.set_spectrum(&gpu, &spectrum);
    let out = run_comp(&gpu, &mut r, comp, 8, 8);
    assert_all_near(&out, [128, 128, 128], 1, "spectrum[10] = 0.5");
    let out = run_comp(
        &gpu,
        &mut r,
        "return vec3f(textureLoad(sampler_spectrum, vec2i(11, 0), 0).r);",
        8,
        8,
    );
    assert_all_near(&out, [0, 0, 0], 0, "NaN spectrum");
}

// ---------------------------------------------------------------------------
// Texture resolver

/// A unique directory under the temp dir, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "ms-render-textures-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    fn sub(&self, name: &str) -> PathBuf {
        let p = self.0.join(name);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Saves `img` (as RGB for `.jpg`, which has no alpha).
fn save(dir: &Path, file: &str, img: &RgbaImage) {
    let path = dir.join(file);
    if file.ends_with(".jpg") {
        image::DynamicImage::ImageRgba8(img.clone())
            .to_rgb8()
            .save(path)
            .unwrap();
    } else {
        img.save(path).unwrap();
    }
}

/// Saves a solid 4x4 image whose red channel tags it.
fn save_tag(dir: &Path, file: &str, tag: u8) {
    save(dir, file, &solid(4, 4, [tag, 0, 0]));
}

fn tag_of(img: &RgbaImage) -> u8 {
    img.get_pixel(0, 0).0[0]
}

#[test]
fn resolver_order() {
    let tmp = TempDir::new("order");
    let preset = tmp.sub("preset");
    let bundled = tmp.sub("bundled");
    save_tag(&preset, "rock.png", 10);
    save_tag(&bundled, "rock.png", 20);
    // .png before .jpg in a root, but every extension of a root before the
    // next root.
    save_tag(&bundled, "sky.png", 30);
    save(&preset, "sky.jpg", &solid(4, 4, [200, 0, 0]));
    save_tag(&bundled, "both.jpg", 40);
    save_tag(&bundled, "both.png", 50);
    save_tag(&bundled, "only_bundled.png", 60);

    let resolver = TextureResolver::new(vec![preset, bundled]);
    assert_eq!(tag_of(&resolver.resolve("rock").unwrap()), 10);
    // JPEG is lossy; a solid colour comes back close.
    assert!(tag_of(&resolver.resolve("sky").unwrap()).abs_diff(200) <= 4);
    assert_eq!(tag_of(&resolver.resolve("both").unwrap()), 50);
    assert_eq!(tag_of(&resolver.resolve("only_bundled").unwrap()), 60);
}

#[test]
fn resolver_missing_none() {
    let tmp = TempDir::new("missing");
    let root = tmp.sub("root");
    save_tag(&root, "present.png", 1);
    // A file that is not an image: logged, then the next candidate is tried.
    std::fs::write(root.join("broken.png"), b"not a png").unwrap();
    let other = tmp.sub("other");
    save_tag(&other, "broken.png", 7);

    let resolver = TextureResolver::new(vec![root.clone(), tmp.0.join("does-not-exist")]);
    assert!(resolver.resolve("absent").is_none());
    assert!(resolver.resolve("broken").is_none());
    assert_eq!(tag_of(&resolver.resolve("present").unwrap()), 1);
    let resolver = TextureResolver::new(vec![root, other]);
    assert_eq!(tag_of(&resolver.resolve("broken").unwrap()), 7);
    assert!(TextureResolver::new(Vec::new())
        .resolve("present")
        .is_none());
}

#[test]
fn resolver_rejects_path_traversal() {
    let tmp = TempDir::new("traversal");
    let root = tmp.sub("root");
    // These would resolve if the name were joined onto the root unchecked.
    save_tag(&tmp.0, "secret.png", 9);
    save_tag(&tmp.sub("root/sub"), "inner.png", 8);
    let resolver = TextureResolver::new(vec![root.clone()]);
    let secret = tmp.0.join("secret");
    let absolute = secret.to_string_lossy().into_owned();
    for name in [
        "../secret",
        "..\\secret",
        "..",
        "sub/inner",
        "sub\\inner",
        "C:secret",
        absolute.as_str(),
        "",
    ] {
        assert!(resolver.resolve(name).is_none(), "{name:?} resolved");
        let mut rng = Rng::seeded(1);
        assert!(
            resolver.resolve_with(name, &mut rng).is_none(),
            "{name:?} resolved with rng"
        );
    }
}

#[test]
fn rand_texture_from_folder() {
    let tmp = TempDir::new("rand");
    let empty_root = tmp.sub("empty");
    tmp.sub("empty/random"); // empty: skipped
    let root = tmp.sub("root");
    let random = tmp.sub("root/random");
    save_tag(&random, "a.png", 101);
    save_tag(&random, "b.png", 102);
    save(&random, "c.jpg", &solid(4, 4, [150, 0, 0]));
    std::fs::write(random.join("notes.txt"), b"ignored").unwrap();
    let late = tmp.sub("late/random");
    save_tag(&late, "z.png", 250);

    let roots = vec![empty_root, root, tmp.0.join("late")];
    let is_one_of_three = |t: u8| t == 101 || t == 102 || t.abs_diff(150) <= 4;
    let mut seen = std::collections::BTreeSet::new();
    for seed in 0..32 {
        let resolver = TextureResolver::new(roots.clone());
        let mut rng = Rng::seeded(seed);
        let t = tag_of(&resolver.resolve_with("rand03", &mut rng).unwrap());
        assert!(is_one_of_three(t), "seed {seed}: tag {t}");
        // The same name keeps resolving to the same file for this resolver,
        // however the rng moves on.
        for _ in 0..4 {
            assert_eq!(
                tag_of(&resolver.resolve_with("rand03", &mut rng).unwrap()),
                t
            );
        }
        seen.insert(if t.abs_diff(150) <= 4 { 150 } else { t });
    }
    assert_eq!(seen.len(), 3, "picks over 32 seeds: {seen:?}");

    // Plain `resolve` does not pick random textures; names outside
    // rand00..rand15 are ordinary files.
    let resolver = TextureResolver::new(roots.clone());
    assert!(resolver.resolve("rand03").is_none());
    let mut rng = Rng::seeded(5);
    assert!(resolver.resolve_with("rand16", &mut rng).is_none());
    save_tag(&roots[1], "rand16.png", 77);
    assert_eq!(
        tag_of(&resolver.resolve_with("rand16", &mut rng).unwrap()),
        77
    );
    // Ordinary names work through `resolve_with` too.
    save_tag(&roots[1], "plain.png", 66);
    assert_eq!(
        tag_of(&resolver.resolve_with("plain", &mut rng).unwrap()),
        66
    );
    // No random folder anywhere: None.
    let resolver = TextureResolver::new(vec![tmp.sub("bare")]);
    assert!(resolver.resolve_with("rand00", &mut rng).is_none());
}

#[test]
fn large_images_downscaled_or_rejected() {
    let tmp = TempDir::new("large");
    let root = tmp.sub("root");
    save(&root, "wide.png", &solid(8192, 16, [10, 20, 30]));
    save(&root, "fits.png", &solid(4096, 3, [1, 2, 3]));
    let resolver = TextureResolver::new(vec![root.clone()]);

    let wide = resolver.resolve("wide").unwrap();
    assert_eq!(wide.dimensions(), (4096, 8));
    assert_eq!(wide.get_pixel(100, 3).0, [10, 20, 30, 255]);
    assert_eq!(resolver.resolve("fits").unwrap().dimensions(), (4096, 3));

    // Over 64 MB: rejected without decoding (and without reading it all).
    let huge = std::fs::File::create(root.join("huge.png")).unwrap();
    huge.set_len(64 * 1024 * 1024 + 1).unwrap();
    drop(huge);
    assert!(resolver.resolve("huge").is_none());
}
