//! Overlay drawing tests: the main waveform, custom waves, shapes, borders
//! and motion vectors drawn into the feedback buffer (spec §6.2, §6.4).
//!
//! These run on the machine's real adapter, or on WARP with
//! `MS_FORCE_WARP=1`.

use ms_render::test_support::{test_gpu, TestGpu};
use std::f32::consts::PI;

use half::f16;
use image::{Rgba, RgbaImage};
use ms_render::{
    main_wave_points, read_texture, Borders, DrawList, FeedbackFormat, Gpu, MainWave,
    MotionVectors, RenderConfig, Renderer, ShapeDraw, WaveDraw,
};

/// A headless GPU holding the GPU test lock (see `ms_render::test_support`).
fn gpu() -> TestGpu {
    test_gpu()
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

/// Draws `list` into a fresh black `w`x`h` Rgba8 feedback buffer and reads
/// it back.
fn draw_on_black(gpu: &Gpu, w: u32, h: u32, list: &DrawList, alpha: f32) -> RgbaImage {
    let mut r = renderer(gpu, w, h, FeedbackFormat::Rgba8);
    r.draw(gpu, list, alpha);
    read_texture(gpu, r.feedback_texture())
}

/// The pixel under `[x, y]` (MilkDrop coordinates: [0, 1], origin at the
/// bottom left).
fn at(img: &RgbaImage, x: f32, y: f32) -> [u8; 4] {
    let px = ((x * img.width() as f32) as u32).min(img.width() - 1);
    let py = (((1.0 - y) * img.height() as f32) as u32).min(img.height() - 1);
    img.get_pixel(px, py).0
}

fn assert_near(got: [u8; 4], want: [u8; 3], tol: u8, what: &str) {
    for c in 0..3 {
        assert!(
            got[c].abs_diff(want[c]) <= tol,
            "{what}: got {got:?}, want {want:?} +/- {tol}"
        );
    }
}

fn is_black(img: &RgbaImage) -> bool {
    img.pixels().all(|p| p.0 == [0, 0, 0, 255])
}

fn sine_wave(cycles: f32, amp: f32) -> [[f32; 576]; 2] {
    let t = |i: usize| i as f32 / 512.0 * cycles * 2.0 * PI;
    [
        std::array::from_fn(|i| amp * t(i).sin()),
        std::array::from_fn(|i| amp * t(i).cos()),
    ]
}

fn main_wave(mode: u8) -> MainWave {
    MainWave {
        mode,
        wave: sine_wave(3.0, 0.8),
        color: [1.0; 4],
        x: 0.5,
        y: 0.5,
        ..MainWave::default()
    }
}

fn opaque_shape(x: f32, y: f32, rad: f32, rgb: [f32; 3]) -> ShapeDraw {
    let c = [rgb[0], rgb[1], rgb[2], 1.0];
    ShapeDraw {
        x,
        y,
        rad,
        sides: 4,
        inner: c,
        outer: c,
        ..ShapeDraw::default()
    }
}

fn line_wave(points: Vec<[f32; 2]>, rgb: [f32; 3], thick: bool) -> WaveDraw {
    let n = points.len();
    WaveDraw {
        points,
        colors: vec![[rgb[0], rgb[1], rgb[2], 1.0]; n],
        dots: false,
        thick,
        additive: false,
    }
}

/// Reads an `Rgba16Float` texture back as raw f16 texels.
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

/// Makes the previous feedback buffer `previous` and the current one
/// `current` (an empty warp clears the next buffer and flips).
fn seed_previous_and_current(
    gpu: &Gpu,
    r: &mut Renderer,
    previous: &RgbaImage,
    current: &RgbaImage,
) {
    r.write_feedback(gpu, previous);
    r.warp(gpu, &[]);
    r.write_feedback(gpu, current);
}

// ---------------------------------------------------------------- geometry

#[test]
fn main_wave_modes_in_bounds() {
    for mode in 0..=7u8 {
        for mystery in [-1.0, 0.0, 0.37, 1.0] {
            for aspect in [[1.0, 1.0], [16.0 / 9.0, 1.0], [1.0, 2.0]] {
                for wave in [sine_wave(3.0, 1.0), sine_wave(40.0, 1.0)] {
                    let w = MainWave {
                        mystery,
                        wave,
                        ..main_wave(mode)
                    };
                    let points = main_wave_points(&w, aspect);
                    assert!(points.len() >= 2, "mode {mode}: {} points", points.len());
                    for p in &points {
                        assert!(
                            p.iter().all(|v| v.is_finite() && (-0.5..=1.5).contains(v)),
                            "mode {mode}, mystery {mystery}, aspect {aspect:?}: point {p:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn main_wave_modes_differ() {
    let shapes: Vec<Vec<[f32; 2]>> = (0..=7)
        .map(|m| main_wave_points(&main_wave(m), [1.0, 1.0]))
        .collect();
    for a in 0..shapes.len() {
        for b in a + 1..shapes.len() {
            // Mode 3 is mode 2's line with a volume-driven alpha.
            if (a, b) == (2, 3) {
                assert_eq!(shapes[a], shapes[b]);
                continue;
            }
            assert_ne!(
                shapes[a], shapes[b],
                "modes {a} and {b} give the same points"
            );
        }
    }
}

#[test]
fn mode0_is_closed() {
    let points = main_wave_points(&main_wave(0), [1.0, 1.0]);
    assert_eq!(points.first(), points.last());
    // A ring around the centre: every point is roughly the same distance
    // away, and it goes all the way round.
    for p in &points {
        let d = ((p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2)).sqrt();
        assert!((0.1..=0.45).contains(&d), "distance {d} at {p:?}");
    }
    // On a wide target the ring is squashed horizontally in [0, 1] units, so
    // it is round on screen.
    let flat = MainWave {
        wave: [[0.0; 576]; 2],
        ..main_wave(0)
    };
    let wide = main_wave_points(&flat, [2.0, 1.0]);
    let (mut max_dx, mut max_dy) = (0.0f32, 0.0f32);
    for p in &wide {
        max_dx = max_dx.max((p[0] - 0.5).abs());
        max_dy = max_dy.max((p[1] - 0.5).abs());
    }
    assert!(
        (max_dx * 2.0 - max_dy).abs() < 1e-3,
        "x extent {max_dx}, y extent {max_dy}"
    );
}

#[test]
fn main_wave_follows_position() {
    let centred = main_wave_points(&main_wave(1), [1.0, 1.0]);
    let moved = main_wave_points(
        &MainWave {
            x: 0.3,
            y: 0.6,
            ..main_wave(1)
        },
        [1.0, 1.0],
    );
    for (a, b) in centred.iter().zip(&moved) {
        assert!((b[0] - a[0] + 0.2).abs() < 1e-5 && (b[1] - a[1] - 0.1).abs() < 1e-5);
    }
}

// ----------------------------------------------------------------- goldens

#[test]
fn wave_golden() {
    let gpu = gpu();
    let list = DrawList {
        main_wave: Some(MainWave {
            thick: true,
            ..main_wave(2)
        }),
        ..DrawList::default()
    };
    let img = draw_on_black(&gpu, 256, 256, &list, 1.0);
    ms_render::assert_golden!(&img, "wave_mode2");
}

#[test]
fn shape_hexagon_golden() {
    let gpu = gpu();
    let list = DrawList {
        shapes: vec![ShapeDraw {
            x: 0.5,
            y: 0.5,
            rad: 0.4,
            sides: 6,
            inner: [1.0, 0.0, 0.0, 1.0],
            outer: [0.0, 0.0, 1.0, 1.0],
            border: [1.0, 1.0, 1.0, 1.0],
            thick_outline: true,
            ..ShapeDraw::default()
        }],
        ..DrawList::default()
    };
    let img = draw_on_black(&gpu, 256, 256, &list, 1.0);
    // Centre red, near the rim mostly blue, outside black.
    assert_near(at(&img, 0.5, 0.5), [255, 0, 0], 8, "centre");
    let near_rim = at(&img, 0.5 + 0.3, 0.5);
    assert!(
        near_rim[2] > 150 && near_rim[0] < 100,
        "near rim {near_rim:?}"
    );
    assert_eq!(at(&img, 0.02, 0.02), [0, 0, 0, 255]);
    ms_render::assert_golden!(&img, "shape_hex");
}

#[test]
fn borders_golden() {
    let gpu = gpu();
    let list = DrawList {
        borders: Borders {
            outer: (0.05, [1.0, 0.0, 0.0, 1.0]),
            inner: (0.1, [0.0, 0.0, 1.0, 0.5]),
        },
        ..DrawList::default()
    };
    // 160x120: the outer border is 8 px at the sides, 6 px at top and
    // bottom; the inner one 16 and 12 px, just inside it.
    let img = draw_on_black(&gpu, 160, 120, &list, 1.0);
    let px = |x: u32, y: u32| img.get_pixel(x, y).0;
    assert_eq!(px(0, 60), [255, 0, 0, 255]);
    assert_eq!(px(7, 60), [255, 0, 0, 255]);
    assert_near(px(8, 60), [0, 0, 128], 1, "inner border, left");
    assert_near(px(23, 60), [0, 0, 128], 1, "inner border, left edge");
    assert_eq!(px(24, 60), [0, 0, 0, 255]);
    assert_eq!(px(80, 5), [255, 0, 0, 255]);
    assert_near(px(80, 6), [0, 0, 128], 1, "inner border, top");
    assert_near(px(80, 17), [0, 0, 128], 1, "inner border, top edge");
    assert_eq!(px(80, 18), [0, 0, 0, 255]);
    assert_eq!(px(159, 119), [255, 0, 0, 255]);
    assert_eq!(px(80, 60), [0, 0, 0, 255]);
    ms_render::assert_golden!(&img, "borders");
}

#[test]
fn borders_zero_draw_nothing() {
    let gpu = gpu();
    let list = DrawList {
        borders: Borders {
            outer: (0.0, [1.0; 4]),
            inner: (0.2, [1.0, 1.0, 1.0, 0.0]),
        },
        ..DrawList::default()
    };
    assert!(is_black(&draw_on_black(&gpu, 64, 64, &list, 1.0)));
}

// ------------------------------------------------------------------ shapes

#[test]
fn shape_textured() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 64, 64, FeedbackFormat::Rgba8);
    let green = RgbaImage::from_pixel(64, 64, Rgba([0, 255, 0, 255]));
    let black = RgbaImage::from_pixel(64, 64, Rgba([0, 0, 0, 255]));
    seed_previous_and_current(&gpu, &mut r, &green, &black);
    let list = DrawList {
        shapes: vec![ShapeDraw {
            textured: true,
            ..opaque_shape(0.5, 0.5, 0.3, [1.0, 1.0, 1.0])
        }],
        ..DrawList::default()
    };
    r.draw(&gpu, &list, 1.0);
    let img = read_texture(&gpu, r.feedback_texture());
    assert_eq!(at(&img, 0.5, 0.5), [0, 255, 0, 255], "shape centre");
    assert_eq!(at(&img, 0.05, 0.05), [0, 0, 0, 255], "outside the shape");
}

#[test]
fn textured_shape_reads_previous_not_its_target() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 64, 64, FeedbackFormat::Rgba8);
    let green = RgbaImage::from_pixel(64, 64, Rgba([0, 255, 0, 255]));
    let red = RgbaImage::from_pixel(64, 64, Rgba([255, 0, 0, 255]));
    seed_previous_and_current(&gpu, &mut r, &green, &red);
    let list = DrawList {
        shapes: vec![ShapeDraw {
            textured: true,
            ..opaque_shape(0.5, 0.5, 0.3, [1.0, 1.0, 1.0])
        }],
        ..DrawList::default()
    };
    r.draw(&gpu, &list, 1.0);
    let img = read_texture(&gpu, r.feedback_texture());
    assert_eq!(
        at(&img, 0.5, 0.5),
        [0, 255, 0, 255],
        "shape shows the previous frame"
    );
    assert_eq!(
        at(&img, 0.05, 0.05),
        [255, 0, 0, 255],
        "target kept outside"
    );
}

#[test]
fn textured_shape_uv_mapping() {
    let gpu = gpu();
    // Previous frame: red top-left, green top-right, blue bottom-left,
    // white bottom-right.
    let quadrants = RgbaImage::from_fn(64, 64, |x, y| match (x < 32, y < 32) {
        (true, true) => Rgba([255, 0, 0, 255]),
        (false, true) => Rgba([0, 255, 0, 255]),
        (true, false) => Rgba([0, 0, 255, 255]),
        (false, false) => Rgba([255, 255, 255, 255]),
    });
    let black = RgbaImage::from_pixel(64, 64, Rgba([0, 0, 0, 255]));
    let sample = |tex_ang: f32, tint: [f32; 3]| {
        let mut r = renderer(&gpu, 64, 64, FeedbackFormat::Rgba8);
        seed_previous_and_current(&gpu, &mut r, &quadrants, &black);
        let list = DrawList {
            shapes: vec![ShapeDraw {
                textured: true,
                tex_ang,
                tex_zoom: 1.0,
                ..opaque_shape(0.5, 0.5, 0.45, tint)
            }],
            ..DrawList::default()
        };
        r.draw(&gpu, &list, 1.0);
        read_texture(&gpu, r.feedback_texture())
    };
    // Unrotated: the shape reproduces what is under it (no flip).
    let img = sample(0.0, [1.0, 1.0, 1.0]);
    assert_eq!(at(&img, 0.35, 0.65), [255, 0, 0, 255], "top-left");
    assert_eq!(at(&img, 0.65, 0.65), [0, 255, 0, 255], "top-right");
    assert_eq!(at(&img, 0.35, 0.35), [0, 0, 255, 255], "bottom-left");
    assert_eq!(at(&img, 0.65, 0.35), [255, 255, 255, 255], "bottom-right");
    // tex_ang rotates the sampling offset counter-clockwise: at pi/2 the
    // top-left of the shape samples the bottom-left of the image.
    let img = sample(PI / 2.0, [1.0, 1.0, 1.0]);
    assert_eq!(at(&img, 0.35, 0.65), [0, 0, 255, 255], "pi/2, top-left");
    let img = sample(PI, [1.0, 1.0, 1.0]);
    assert_eq!(at(&img, 0.35, 0.65), [255, 255, 255, 255], "pi, top-left");
    // The colour multiplies the texel.
    let img = sample(0.0, [0.5, 1.0, 0.0]);
    assert_near(at(&img, 0.65, 0.35), [128, 255, 0], 1, "tinted white");
}

#[test]
fn instances_1024() {
    let gpu = gpu();
    let mut seed = 0x2545_f491_u32;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };
    let shapes: Vec<ShapeDraw> = (0..1024)
        .map(|i| ShapeDraw {
            x: rand(),
            y: rand(),
            rad: 0.01,
            ang: rand() * 6.0,
            sides: 3 + i % 98,
            inner: [rand(), rand(), rand(), 1.0],
            outer: [rand(), rand(), rand(), 0.5],
            border: [1.0, 1.0, 1.0, rand()],
            additive: i % 3 == 0,
            thick_outline: i % 2 == 0,
            textured: i % 5 == 0,
            tex_zoom: 1.0 + rand(),
            tex_ang: rand(),
        })
        .collect();
    let list = DrawList {
        shapes,
        ..DrawList::default()
    };
    let mut r = renderer(&gpu, 256, 256, FeedbackFormat::Rgba8);
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    r.draw(&gpu, &list, 1.0);
    // Twice, so the reused buffers are exercised too.
    r.draw(&gpu, &list, 1.0);
    let error = pollster::block_on(scope.pop());
    assert!(error.is_none(), "{error:?}");
    gpu.poll_wait().unwrap();
    let img = read_texture(&gpu, r.feedback_texture());
    let lit = img.pixels().filter(|p| p.0[..3] != [0, 0, 0]).count();
    assert!(lit > 1000, "only {lit} pixels lit");
}

#[test]
fn additive_shapes_add() {
    let gpu = gpu();
    let half_red = |additive| ShapeDraw {
        additive,
        ..opaque_shape(0.5, 0.5, 0.3, [0.5, 0.0, 0.0])
    };
    let list = DrawList {
        shapes: vec![half_red(true), half_red(true)],
        ..DrawList::default()
    };
    let img = draw_on_black(&gpu, 32, 32, &list, 1.0);
    assert_near(at(&img, 0.5, 0.5), [255, 0, 0], 2, "two additive halves");
    let list = DrawList {
        shapes: vec![half_red(false), half_red(false)],
        ..DrawList::default()
    };
    let img = draw_on_black(&gpu, 32, 32, &list, 1.0);
    assert_near(at(&img, 0.5, 0.5), [128, 0, 0], 2, "opaque over");
}

#[test]
fn alpha_scales() {
    let gpu = gpu();
    let list = DrawList {
        shapes: vec![opaque_shape(0.5, 0.5, 0.3, [1.0, 1.0, 1.0])],
        main_wave: Some(MainWave {
            wave: [[0.0; 576]; 2],
            y: 0.2,
            thick: true,
            ..main_wave(2)
        }),
        ..DrawList::default()
    };
    let full = draw_on_black(&gpu, 64, 64, &list, 1.0);
    let half = draw_on_black(&gpu, 64, 64, &list, 0.5);
    let mut lit = 0;
    for (a, b) in full.pixels().zip(half.pixels()) {
        for c in 0..3 {
            let want = f32::from(a.0[c]) * 0.5;
            assert!(
                (f32::from(b.0[c]) - want).abs() <= 2.0,
                "full {:?}, half {:?}",
                a.0,
                b.0
            );
        }
        if a.0[0] == 255 {
            lit += 1;
        }
    }
    assert!(lit > 100, "only {lit} pixels lit");
}

// ------------------------------------------------------------- draw order

#[test]
fn order_shapes_before_waves() {
    let gpu = gpu();
    let list = DrawList {
        shapes: vec![opaque_shape(0.5, 0.5, 0.3, [1.0, 0.0, 0.0])],
        waves: vec![line_wave(
            vec![[0.0, 0.51], [1.0, 0.51]],
            [0.0, 1.0, 0.0],
            true,
        )],
        ..DrawList::default()
    };
    let img = draw_on_black(&gpu, 64, 64, &list, 1.0);
    let column: Vec<[u8; 4]> = (0..64).map(|y| img.get_pixel(32, y).0).collect();
    assert!(column.contains(&[0, 255, 0, 255]), "no green in {column:?}");
    assert!(column.contains(&[255, 0, 0, 255]), "no red in {column:?}");
}

#[test]
fn draw_order_follows_spec() {
    let gpu = gpu();
    let centre_column = |list: &DrawList| -> Vec<[u8; 4]> {
        let img = draw_on_black(&gpu, 64, 64, list, 1.0);
        (0..64).map(|y| img.get_pixel(32, y).0).collect()
    };
    let flat = MainWave {
        wave: [[0.0; 576]; 2],
        y: 0.04,
        color: [0.0, 0.0, 1.0, 1.0],
        thick: true,
        ..main_wave(2)
    };
    // Motion vectors, then shapes: the shape covers them.
    let list = DrawList {
        motion: Some(MotionVectors {
            nx: 8.0,
            ny: 8.0,
            dx: 0.0,
            dy: 0.0,
            len: 10.0,
            color: [0.0, 1.0, 0.0, 1.0],
        }),
        shapes: vec![opaque_shape(0.5, 0.5, 1.0, [1.0, 0.0, 0.0])],
        ..DrawList::default()
    };
    let col = centre_column(&list);
    assert!(col.iter().all(|p| *p == [255, 0, 0, 255]), "{col:?}");
    // Custom waves, then the main wave.
    let list = DrawList {
        waves: vec![line_wave(
            vec![[0.0, 0.04], [1.0, 0.04]],
            [0.0, 1.0, 0.0],
            true,
        )],
        main_wave: Some(flat.clone()),
        ..DrawList::default()
    };
    let col = centre_column(&list);
    assert!(
        col.contains(&[0, 0, 255, 255]) && !col.contains(&[0, 255, 0, 255]),
        "{col:?}"
    );
    // The main wave, then the outer border, then the inner border.
    let list = DrawList {
        main_wave: Some(flat),
        borders: Borders {
            outer: (0.1, [1.0, 0.0, 0.0, 1.0]),
            inner: (0.1, [0.0, 1.0, 0.0, 1.0]),
        },
        ..DrawList::default()
    };
    let col = centre_column(&list);
    assert!(!col.contains(&[0, 0, 255, 255]), "{col:?}");
    // 64 rows: the outer border covers y < 0.1 (rows 58-63), the inner
    // one 0.1 <= y < 0.2 (rows 51-57).
    assert_eq!(col[63], [255, 0, 0, 255]);
    assert_eq!(col[58], [255, 0, 0, 255]);
    assert_eq!(col[57], [0, 255, 0, 255]);
    assert_eq!(col[51], [0, 255, 0, 255]);
    assert_eq!(col[50], [0, 0, 0, 255]);
}

// ------------------------------------------------------------- lines, dots

#[test]
fn thick_lines_and_dots_scale_with_height() {
    let gpu = gpu();
    // 1080 rows: thin lines are 1 px, thick ones 2 px, dots 2x2 px. The
    // line's centre sits a quarter pixel below a pixel centre, so no edge
    // falls on a pixel centre.
    let y = 1.0 - 536.25 / 1080.0;
    let lit_rows = |thick| {
        let list = DrawList {
            waves: vec![line_wave(vec![[0.0, y], [1.0, y]], [1.0; 3], thick)],
            ..DrawList::default()
        };
        let img = draw_on_black(&gpu, 64, 1080, &list, 1.0);
        (0..1080)
            .filter(|&r| img.get_pixel(32, r).0[0] > 0)
            .collect::<Vec<u32>>()
    };
    assert_eq!(lit_rows(false), [536]);
    assert_eq!(lit_rows(true), [535, 536]);

    let list = DrawList {
        waves: vec![WaveDraw {
            dots: true,
            ..line_wave(vec![[32.25 / 64.0, y]], [1.0; 3], false)
        }],
        ..DrawList::default()
    };
    let img = draw_on_black(&gpu, 64, 1080, &list, 1.0);
    let lit: Vec<(u32, u32)> = img
        .enumerate_pixels()
        .filter(|(_, _, p)| p.0[0] > 0)
        .map(|(x, y, _)| (x, y))
        .collect();
    assert_eq!(lit, [(31, 535), (32, 535), (31, 536), (32, 536)]);

    // At 64 rows everything is at least one pixel wide.
    let list = DrawList {
        waves: vec![line_wave(vec![[0.0, 0.51], [1.0, 0.51]], [1.0; 3], false)],
        ..DrawList::default()
    };
    let img = draw_on_black(&gpu, 64, 64, &list, 1.0);
    for x in 0..64 {
        assert!(
            (0..64).any(|r| img.get_pixel(x, r).0[0] > 0),
            "column {x} has no line"
        );
    }
}

#[test]
fn main_wave_brighten_and_volume_alpha() {
    let gpu = gpu();
    let flat = MainWave {
        wave: [[0.0; 576]; 2],
        color: [0.5, 0.25, 0.0, 1.0],
        thick: true,
        ..main_wave(2)
    };
    let brightest = |w: MainWave| {
        let list = DrawList {
            main_wave: Some(w),
            ..DrawList::default()
        };
        let img = draw_on_black(&gpu, 64, 64, &list, 1.0);
        img.pixels().map(|p| p.0).max_by_key(|p| p[0]).unwrap()
    };
    assert_near(brightest(flat.clone()), [128, 64, 0], 1, "plain");
    let bright = brightest(MainWave {
        brighten: true,
        ..flat.clone()
    });
    assert_near(bright, [255, 128, 0], 1, "brightened");
    // Mode 3: alpha follows the volume, so silence draws nothing and a
    // full-scale square wave draws at full alpha.
    assert_eq!(
        brightest(MainWave {
            mode: 3,
            ..flat.clone()
        }),
        [0, 0, 0, 255]
    );
    let mut square = [[0.0; 576]; 2];
    for (i, s) in square.iter_mut().flatten().enumerate() {
        *s = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let full = brightest(MainWave {
        mode: 3,
        wave: square,
        dots: true,
        ..flat
    });
    assert_near(full, [128, 64, 0], 1, "mode 3 at full volume");
}

// ---------------------------------------------------------- motion vectors

fn motion(alpha: f32) -> MotionVectors {
    MotionVectors {
        nx: 4.9,
        ny: 3.0,
        dx: 0.0,
        dy: 0.0,
        len: 5.0,
        color: [1.0, 1.0, 1.0, alpha],
    }
}

#[test]
fn motion_vectors_alpha_zero_draw_nothing() {
    let gpu = gpu();
    let list = DrawList {
        motion: Some(motion(0.0)),
        ..DrawList::default()
    };
    assert!(is_black(&draw_on_black(&gpu, 128, 128, &list, 1.0)));
    let list = DrawList {
        motion: Some(MotionVectors {
            nx: 0.5,
            ..motion(1.0)
        }),
        ..DrawList::default()
    };
    assert!(is_black(&draw_on_black(&gpu, 128, 128, &list, 1.0)));
}

#[test]
fn motion_vectors_point_to_the_centre() {
    let gpu = gpu();
    let list = DrawList {
        motion: Some(motion(1.0)),
        ..DrawList::default()
    };
    let img = draw_on_black(&gpu, 128, 128, &list, 1.0);
    // nx 4.9 rounds down to 4 columns, 3 rows: 12 short segments.
    let lit_near = |x: f32, y: f32| {
        let (cx, cy) = (x * 128.0, (1.0 - y) * 128.0);
        img.enumerate_pixels().any(|(px, py, p)| {
            p.0[0] > 0 && (px as f32 + 0.5 - cx).abs() <= 1.5 && (py as f32 + 0.5 - cy).abs() <= 1.5
        })
    };
    for i in 0..4 {
        for j in 0..3 {
            let g = [(i as f32 + 0.5) / 4.0, (j as f32 + 0.5) / 3.0];
            let to_centre = [0.5 - g[0], 0.5 - g[1]];
            let d = (to_centre[0].powi(2) + to_centre[1].powi(2)).sqrt();
            if d < 1e-6 {
                continue;
            }
            // Halfway along the segment (5 steps of 0.01 = 0.05 long).
            let mid = [
                g[0] + to_centre[0] / d * 0.025,
                g[1] + to_centre[1] / d * 0.025,
            ];
            assert!(
                lit_near(mid[0], mid[1]),
                "vector {i},{j}: nothing at {mid:?}"
            );
            // Nothing on the far side of the grid point.
            let behind = [
                g[0] - to_centre[0] / d * 0.03,
                g[1] - to_centre[1] / d * 0.03,
            ];
            assert!(
                !lit_near(behind[0], behind[1]),
                "vector {i},{j}: lit at {behind:?}"
            );
        }
    }
    let lit = img.pixels().filter(|p| p.0[0] > 0).count();
    assert!((24..400).contains(&lit), "{lit} pixels lit");
}

// --------------------------------------------------------------- robustness

#[test]
fn non_finite_inputs_draw_finite_output() {
    let gpu = gpu();
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let mut bad_wave = [[nan; 576]; 2];
    bad_wave[1] = [inf; 576];
    let list = DrawList {
        motion: Some(MotionVectors {
            nx: inf,
            ny: 3.0,
            dx: nan,
            dy: -inf,
            len: inf,
            color: [nan, inf, 1.0, 1.0],
        }),
        shapes: vec![
            ShapeDraw {
                x: nan,
                y: inf,
                rad: inf,
                ang: nan,
                sides: 0,
                inner: [nan, inf, -inf, 1.0],
                outer: [1.0, 1.0, 1.0, nan],
                border: [inf; 4],
                additive: false,
                thick_outline: true,
                textured: true,
                tex_zoom: nan,
                tex_ang: inf,
            },
            ShapeDraw {
                sides: u32::MAX,
                tex_zoom: 0.0,
                textured: true,
                ..opaque_shape(0.5, 0.5, 0.2, [1.0, 1.0, 1.0])
            },
        ],
        waves: vec![
            WaveDraw {
                points: vec![[nan, inf], [-inf, 0.5], [0.5, 0.5], [1e30, -1e30]],
                colors: vec![[inf, nan, 1.0, 1.0]; 4],
                dots: false,
                thick: true,
                additive: false,
            },
            WaveDraw {
                points: vec![[0.5, 0.5]; 3],
                colors: vec![],
                dots: true,
                thick: false,
                additive: false,
            },
        ],
        main_wave: Some(MainWave {
            mode: 200,
            wave: bad_wave,
            spectrum: [nan; 512],
            color: [inf, inf, nan, inf],
            x: nan,
            y: inf,
            mystery: nan,
            dots: false,
            thick: true,
            additive: false,
            brighten: true,
        }),
        borders: Borders {
            outer: (nan, [1.0; 4]),
            inner: (inf, [inf, nan, 1.0, 1.0]),
        },
    };
    for mode in 0..=7 {
        let w = MainWave {
            mode,
            ..list.main_wave.clone().unwrap()
        };
        for p in main_wave_points(&w, [nan, inf]) {
            assert!(p.iter().all(|v| v.is_finite()), "mode {mode}: {p:?}");
        }
    }
    let mut r = renderer(&gpu, 64, 64, FeedbackFormat::Fp16);
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    r.draw(&gpu, &list, 1.0);
    r.draw(&gpu, &list, nan);
    r.draw(&gpu, &list, inf);
    let error = pollster::block_on(scope.pop());
    assert!(error.is_none(), "{error:?}");
    for (i, t) in read_f16(&gpu, r.feedback_texture()).iter().enumerate() {
        let v = t.map(f16::to_f32);
        assert!(
            v.iter().all(|c| c.is_finite() && (0.0..=1.0).contains(c)),
            "texel {i}: {v:?}"
        );
        assert_eq!(v[3], 1.0, "texel {i} alpha");
    }
}

#[test]
fn fp16_feedback_keeps_alpha_one() {
    let gpu = gpu();
    let mut r = renderer(&gpu, 32, 32, FeedbackFormat::Fp16);
    let list = DrawList {
        shapes: vec![ShapeDraw {
            inner: [1.0, 0.0, 0.0, 0.5],
            outer: [1.0, 0.0, 0.0, 0.5],
            additive: true,
            ..opaque_shape(0.5, 0.5, 0.5, [1.0, 0.0, 0.0])
        }],
        ..DrawList::default()
    };
    r.draw(&gpu, &list, 1.0);
    let texels = read_f16(&gpu, r.feedback_texture());
    let centre = texels[16 * 32 + 16].map(f16::to_f32);
    assert!((centre[0] - 0.5).abs() < 0.01, "{centre:?}");
    assert!(texels.iter().all(|t| t[3].to_f32() == 1.0));
}
