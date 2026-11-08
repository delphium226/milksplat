//! On-screen text overlay (spec §6.3): `font8x8` glyphs drawn onto a display
//! target with a one-pixel black shadow.
//!
//! These run on the machine's real adapter, or on WARP with
//! `MS_FORCE_WARP=1`.

use image::RgbaImage;
use ms_render::{read_texture, FeedbackFormat, Gpu, RenderConfig, Renderer, TextItem};

const RGBA8: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// The clear colour of the test targets, as bytes.
const BACKGROUND: [u8; 4] = [51, 76, 115, 255];

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
            internal_size: [64, 64],
            format: FeedbackFormat::Rgba8,
        },
    )
}

/// A `w`x`h` target of `format`, cleared to [`BACKGROUND`].
fn target(gpu: &Gpu, w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("text target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let [r, g, b, a] = BACKGROUND.map(|v| f64::from(v) / 255.0);
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("text target clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
    }
    gpu.queue.submit([encoder.finish()]);
    tex
}

/// Draws `items` onto a fresh `w`x`h` Rgba8 target and reads it back.
fn draw(gpu: &Gpu, r: &mut Renderer, w: u32, h: u32, items: &[TextItem]) -> RgbaImage {
    let tex = target(gpu, w, h, RGBA8);
    let view = tex.create_view(&Default::default());
    r.draw_text(gpu, &view, [w, h], items);
    read_texture(gpu, &tex)
}

fn item(text: &str, pos: [f32; 2], scale: u32) -> TextItem {
    TextItem {
        text: text.to_string(),
        pos,
        scale,
        color: [1.0, 1.0, 1.0, 1.0],
    }
}

/// Runs `f` inside a validation error scope and returns the error, if any.
fn validation_error(gpu: &Gpu, f: impl FnOnce()) -> Option<wgpu::Error> {
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    f();
    pollster::block_on(scope.pop())
}

#[test]
fn text_golden() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let golden = TextItem {
        text: "Ink Bloom — 60 fps".to_string(),
        pos: [8.0, 12.0],
        scale: 2,
        color: [1.0, 0.85, 0.3, 1.0],
    };
    let img = draw(&gpu, &mut r, 320, 40, std::slice::from_ref(&golden));

    // The em dash is not basic Latin, so it is drawn exactly like `?`.
    let question = TextItem {
        text: "Ink Bloom ? 60 fps".to_string(),
        ..golden.clone()
    };
    let with_question = draw(&gpu, &mut r, 320, 40, &[question]);
    assert!(
        img == with_question,
        "the em dash must render exactly like `?`"
    );
    ms_render::assert_golden!(img, "text");
}

/// `!` is rows `18 3C 3C 18 18 00 18 00` (bit 0 = leftmost column): checks
/// glyph pixels, background and the shadow at (+1, +1) exactly.
#[test]
fn glyph_and_shadow_pixels() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let img = draw(&gpu, &mut r, 16, 16, &[item("!", [2.0, 3.0], 1)]);
    let px = |x: u32, y: u32| img.get_pixel(2 + x, 3 + y).0;
    const WHITE: [u8; 4] = [255, 255, 255, 255];
    const BLACK: [u8; 4] = [0, 0, 0, 255];
    // Row 0: columns 3 and 4 lit.
    assert_eq!(px(2, 0), BACKGROUND);
    assert_eq!(px(3, 0), WHITE);
    assert_eq!(px(4, 0), WHITE);
    assert_eq!(px(5, 0), BACKGROUND);
    // Row 5 is empty: only the shadow of row 4 (columns 3-4 lit) shows,
    // shifted to columns 4-5.
    assert_eq!(px(3, 5), BACKGROUND);
    assert_eq!(px(4, 5), BLACK);
    assert_eq!(px(5, 5), BLACK);
    assert_eq!(px(6, 5), BACKGROUND);
    // Row 6 lit at columns 3-4 (column 5 would hold the shadow of the
    // empty row 5); its own shadow is at row 7, columns 4-5.
    assert_eq!(px(3, 6), WHITE);
    assert_eq!(px(5, 6), BACKGROUND);
    assert_eq!(px(4, 7), BLACK);
    assert_eq!(px(5, 7), BLACK);
    // Nothing outside the glyph cell plus its shadow.
    assert_eq!(img.get_pixel(0, 0).0, BACKGROUND);
    assert_eq!(img.get_pixel(15, 15).0, BACKGROUND);
}

#[test]
fn colour_and_alpha_blend_over_the_target() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let half_red = TextItem {
        color: [1.0, 0.0, 0.0, 0.5],
        ..item("!", [0.0, 0.0], 1)
    };
    let img = draw(&gpu, &mut r, 16, 16, &[half_red]);
    // Lit pixel (3, 0): 0.5 * red + 0.5 * background.
    let p = img.get_pixel(3, 0).0;
    let want = [
        (255.0 + f32::from(BACKGROUND[0])) / 2.0,
        f32::from(BACKGROUND[1]) / 2.0,
        f32::from(BACKGROUND[2]) / 2.0,
    ];
    for c in 0..3 {
        assert!(
            (f32::from(p[c]) - want[c]).abs() <= 1.5,
            "lit pixel {p:?}, want about {want:?}"
        );
    }
    // Shadow-only pixel (4, 5): half black over the background.
    let s = img.get_pixel(4, 5).0;
    for c in 0..3 {
        let want = f32::from(BACKGROUND[c]) / 2.0;
        assert!(
            (f32::from(s[c]) - want).abs() <= 1.5,
            "shadow pixel {s:?}, want about {want}"
        );
    }
}

#[test]
fn scale_multiplies_glyph_and_shadow() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let img = draw(&gpu, &mut r, 32, 32, &[item("!", [0.0, 0.0], 3)]);
    // Row 0 column 3 becomes the 3x3 block at (9..12, 0..3).
    for y in 0..3 {
        for x in 9..15 {
            assert_eq!(img.get_pixel(x, y).0, [255; 4], "({x},{y})");
        }
        assert_eq!(img.get_pixel(8, y).0, BACKGROUND, "(8,{y})");
    }
    // Row 5 (empty) shows the shadow of row 4, offset by 3 px: columns
    // 3-4 -> pixels 12..18, rows 15..18.
    for y in 15..18 {
        assert_eq!(img.get_pixel(11, y).0, BACKGROUND, "(11,{y})");
        for x in 12..18 {
            assert_eq!(img.get_pixel(x, y).0, [0, 0, 0, 255], "({x},{y})");
        }
        assert_eq!(img.get_pixel(18, y).0, BACKGROUND, "(18,{y})");
    }
}

#[test]
fn scale_zero_is_treated_as_one() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let zero = draw(&gpu, &mut r, 48, 16, &[item("Hi!", [1.0, 1.0], 0)]);
    let one = draw(&gpu, &mut r, 48, 16, &[item("Hi!", [1.0, 1.0], 1)]);
    let two = draw(&gpu, &mut r, 48, 16, &[item("Hi!", [1.0, 1.0], 2)]);
    assert!(zero == one, "scale 0 must draw exactly like scale 1");
    assert!(one != two, "scale 2 must differ from scale 1");
    assert!(
        one.pixels().any(|p| p.0 == [255; 4]),
        "scale 1 draws something"
    );
}

#[test]
fn zero_size_target_skipped() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let tex = target(&gpu, 16, 16, RGBA8);
    let view = tex.create_view(&Default::default());
    let items = [item("Hello", [0.0, 0.0], 1)];
    for size in [[0, 0], [0, 16], [16, 0]] {
        let error = validation_error(&gpu, || r.draw_text(&gpu, &view, size, &items));
        assert!(error.is_none(), "{size:?}: {error:?}");
    }
    // Empty item lists and empty strings draw nothing either.
    let error = validation_error(&gpu, || {
        r.draw_text(&gpu, &view, [16, 16], &[]);
        r.draw_text(&gpu, &view, [16, 16], &[item("", [0.0, 0.0], 1)]);
    });
    assert!(error.is_none(), "{error:?}");
    let img = read_texture(&gpu, &tex);
    assert!(img.pixels().all(|p| p.0 == BACKGROUND), "target unchanged");
}

#[test]
fn bad_items_do_not_panic() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let items = [
        TextItem {
            pos: [f32::NAN, 0.0],
            ..item("nan", [0.0, 0.0], 1)
        },
        TextItem {
            color: [f32::INFINITY, f32::NAN, -1.0, 2.0],
            ..item("colour", [0.0, 0.0], 1)
        },
        item("far away", [1e30, -1e30], u32::MAX),
        item("\u{0}\n\t\u{7f}\u{1F600}", [0.0, 8.0], 1),
    ];
    let error = validation_error(&gpu, || {
        draw(&gpu, &mut r, 32, 32, &items);
    });
    assert!(error.is_none(), "{error:?}");
}

#[test]
fn pipelines_are_created_once_per_target_format() {
    let gpu = gpu();
    let mut r = renderer(&gpu);
    let items = [item("fps", [0.0, 0.0], 1)];
    assert_eq!(r.text_pipeline_creations(), 0, "created lazily");
    for _ in 0..3 {
        draw(&gpu, &mut r, 32, 16, &items);
    }
    assert_eq!(r.text_pipeline_creations(), 1);
    let bgra = wgpu::TextureFormat::Bgra8Unorm;
    let tex = target(&gpu, 32, 16, bgra);
    let view = tex.create_view(&Default::default());
    r.draw_text(&gpu, &view, [32, 16], &items);
    r.draw_text(&gpu, &view, [32, 16], &items);
    assert_eq!(r.text_pipeline_creations(), 2);
    // Same glyphs on BGRA (read back swizzled to RGBA).
    let rgba = draw(&gpu, &mut r, 32, 16, &items);
    assert!(read_texture(&gpu, &tex) == rgba);
}
