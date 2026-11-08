//! Per-tick execution of a preset instance (spec §5.2, §5.3, §6.2, §6.4).
//!
//! Mesh vertex `k = row * (cols + 1) + col` is at preset `x = col / cols`,
//! `y = 1 - row / rows` (bottom-up); its UV is in texture space (v = 0 at the
//! top), so the identity warp gives `uv = (col / cols, row / rows)`.

use ms_analysis::AudioFrame;
use ms_engine::{Globals, PresetInstance, TickInput, TickOutput, VertexPath};
use ms_preset::{compile_preset, parse, CompiledPreset};
use std::sync::Arc;

fn compiled(toml: &str) -> Arc<CompiledPreset> {
    let (p, warnings) = parse(toml.as_bytes()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    Arc::new(compile_preset(p))
}

fn instance(toml: &str) -> PresetInstance {
    PresetInstance::new(compiled(toml), 7, [192, 144])
}

fn input(audio: &AudioFrame) -> TickInput<'_> {
    TickInput {
        time: 1.0,
        fps: 60.0,
        frame: 60,
        progress: 0.25,
        audio,
        aspect: [1.0, 1.0],
        pixels: [640, 480],
    }
}

/// Ticks `n` times with silent audio and returns a copy of the last output.
fn run(inst: &mut PresetInstance, n: usize) -> TickOutput {
    let audio = AudioFrame::default();
    let mut g = Globals::default();
    let mut out = None;
    for _ in 0..n {
        out = Some(inst.tick(&input(&audio), &mut g).clone());
    }
    out.unwrap()
}

fn close(a: f32, b: f64, tol: f64) -> bool {
    (f64::from(a) - b).abs() <= tol
}

/// Preset coordinates of vertex `k`.
fn xy(out: &TickOutput, k: usize) -> (f64, f64) {
    let cols = out.mesh.cols as usize;
    let rows = out.mesh.rows as usize;
    let (row, col) = (k / (cols + 1), k % (cols + 1));
    (col as f64 / cols as f64, 1.0 - row as f64 / rows as f64)
}

const SMALL: &str = "[params]\nmesh = [8, 6]\nwarp = 0.0\n";

#[test]
fn identity_preset_identity_mesh() {
    let out = run(&mut instance(SMALL), 1);
    assert_eq!((out.mesh.cols, out.mesh.rows), (8, 6));
    assert_eq!(out.mesh.uv.len(), 9 * 7);
    assert_eq!(out.mesh.color.len(), 9 * 7);
    for (k, uv) in out.mesh.uv.iter().enumerate() {
        let (x, y) = xy(&out, k);
        assert!(
            close(uv[0], x, 1e-6) && close(uv[1], 1.0 - y, 1e-6),
            "vertex {k}: {uv:?}"
        );
    }
    assert!(out.mesh.color.iter().all(|c| *c == [1.0; 4]));
}

#[test]
fn zoom_param_shrinks() {
    let out = run(&mut instance(&format!("{SMALL}zoom = 1.1\n")), 1);
    let id = run(&mut instance(SMALL), 1);
    for k in [0, 8, 9 * 6, 9 * 7 - 1] {
        let d = |uv: [f32; 2]| ((uv[0] - 0.5).powi(2) + (uv[1] - 0.5).powi(2)).sqrt();
        assert!(d(out.mesh.uv[k]) < d(id.mesh.uv[k]) - 0.01, "corner {k}");
    }
    // Top-left corner: 0.5 - 0.5 / 1.1 on both axes.
    let e = 0.5 - 0.5 / 1.1;
    assert!(close(out.mesh.uv[0][0], e, 1e-6) && close(out.mesh.uv[0][1], e, 1e-6));
}

#[test]
fn frame_code_sees_bass() {
    let mut inst = instance(&format!("{SMALL}[code]\nframe = \"zoom = 1 + bass\"\n"));
    let audio = AudioFrame {
        bass: 0.5,
        ..AudioFrame::default()
    };
    let out = inst.tick(&input(&audio), &mut Globals::default());
    assert_eq!(out.uniforms.bass, 0.5);
    let e = 0.5 - 0.5 / 1.5;
    assert!(close(out.mesh.uv[0][0], e, 1e-6), "{:?}", out.mesh.uv[0]);
    assert!(close(out.mesh.uv[0][1], e, 1e-6), "{:?}", out.mesh.uv[0]);
}

/// Positive `dx` makes every vertex sample 0.25 further left (the image
/// moves right); `v` is unchanged.
#[test]
fn q_passes_to_vertex() {
    let toml = format!("{SMALL}[code]\nframe = \"q1 = 0.25\"\nvertex = \"dx = q1\"\n");
    let out = run(&mut instance(&toml), 1);
    for (k, uv) in out.mesh.uv.iter().enumerate() {
        let (x, y) = xy(&out, k);
        assert!(close(uv[0], x - 0.25, 1e-6), "vertex {k}: {uv:?}");
        assert!(close(uv[1], 1.0 - y, 1e-6), "vertex {k}: {uv:?}");
    }
    assert_eq!(out.uniforms.q[0][0], 0.25);
}

#[test]
fn vertex_starts_from_frame_values() {
    // Every vertex must see rot = 1 from the frame code, although the
    // previous vertex set rot = 0.
    let toml =
        format!("{SMALL}[code]\nframe = \"rot = 1\"\nvertex = \"zoom = rot + 1; rot = 0\"\n");
    let out = run(&mut instance(&toml), 1);
    for (k, uv) in out.mesh.uv.iter().enumerate() {
        let (x, y) = xy(&out, k);
        let (u, v) = (0.5 + (x - 0.5) / 2.0, 0.5 + (y - 0.5) / 2.0);
        assert!(
            close(uv[0], u, 1e-6) && close(uv[1], 1.0 - v, 1e-6),
            "vertex {k}: {uv:?}"
        );
    }
}

#[test]
fn user_var_carries_rowmajor() {
    let toml = format!("{SMALL}[code]\nvertex = \"n = n + 1; dx = n*0.0001\"\n");
    let mut inst = instance(&toml);
    let out = run(&mut inst, 1);
    assert_eq!(inst.vertex_path(), VertexPath::Scalar);
    for (k, uv) in out.mesh.uv.iter().enumerate() {
        let (x, _) = xy(&out, k);
        assert!(
            close(uv[0], x - (k + 1) as f64 * 0.0001, 1e-6),
            "vertex {k}"
        );
    }
    let last = out.mesh.uv.len() - 1;
    assert!(close(
        out.mesh.uv[last][0],
        1.0 - (9.0 * 7.0) * 0.0001,
        1e-6
    ));
}

/// A vertex program that is batchable (no carried variable is read before
/// it is written), and the same program forced onto the scalar path by a
/// carried read (`n = n + 0`).
fn batch_pair() -> (String, String) {
    let vertex = "zoom = 1 + x*0.1 + q1*rad; rot = if(y > 0.5, 0.1, ang*0.01); \
                  m = sin(x); dy = m*0.01; if(x < 0.5, k = x, 0); reg07 = y + 2";
    let frame = "q1 = 0.05; q2 = m; q3 = k; q4 = reg07";
    let mk =
        |v: &str| format!("[params]\nwarp = 0.5\n[code]\nframe = \"{frame}\"\nvertex = \"{v}\"\n");
    (mk(vertex), mk(&format!("{vertex}; n = n + 0")))
}

#[test]
fn batched_used_when_safe() {
    let (batched_src, scalar_src) = batch_pair();
    let (mut b, mut s) = (instance(&batched_src), instance(&scalar_src));
    let audio = AudioFrame::default();
    let (mut gb, mut gs) = (Globals::default(), Globals::default());
    for tick in 0..2 {
        let ob = b.tick(&input(&audio), &mut gb).clone();
        let os = s.tick(&input(&audio), &mut gs).clone();
        assert_eq!(b.vertex_path(), VertexPath::Batched);
        assert_eq!(s.vertex_path(), VertexPath::Scalar);
        assert_eq!((ob.mesh.cols, ob.mesh.rows), (64, 48));
        assert_eq!(ob.mesh.uv.len(), 65 * 49);
        for (k, (a, c)) in ob.mesh.uv.iter().zip(&os.mesh.uv).enumerate() {
            assert_eq!(a[0].to_bits(), c[0].to_bits(), "tick {tick} vertex {k}");
            assert_eq!(a[1].to_bits(), c[1].to_bits(), "tick {tick} vertex {k}");
        }
        assert_eq!(ob.uniforms.q, os.uniforms.q, "tick {tick}");
        if tick == 1 {
            // Carried variables end with the last writer's value, in
            // row-major order: m from the last vertex (x = 1), k from the
            // last vertex with x < 0.5 (col 31 of 64), reg07 = y of the last
            // vertex (0) plus 2.
            let q = ob.uniforms.q[0];
            assert_eq!(q[1], 1f64.sin() as f32);
            assert_eq!(q[2], (31.0f64 / 64.0) as f32);
            assert_eq!(q[3], 2.0);
            assert_eq!(gb.regs[7], 2.0);
        }
    }
    // The mesh actually moved.
    let out = run(&mut instance(&batched_src), 1);
    assert!(out.mesh.uv[0] != [0.0, 0.0]);
}

#[test]
fn user_vars_persist_ticks() {
    let toml = format!("{SMALL}[code]\nframe = \"t = t + 1; q1 = t\"\n");
    let out = run(&mut instance(&toml), 3);
    assert_eq!(out.uniforms.q[0][0], 3.0);
}

#[test]
fn writables_reset_each_tick() {
    let toml = format!("{SMALL}[code]\nframe = \"zoom = zoom * 2; q1 = zoom\"\n");
    let mut inst = instance(&toml);
    let audio = AudioFrame::default();
    let mut g = Globals::default();
    for _ in 0..3 {
        let out = inst.tick(&input(&audio), &mut g);
        assert_eq!(out.uniforms.q[0][0], 2.0);
        assert!(close(out.mesh.uv[0][0], 0.25, 1e-6));
    }
}

#[test]
fn regs_global() {
    let mut a = instance(&format!("{SMALL}[code]\nframe = \"reg05 = 7\"\n"));
    let mut b = instance(&format!("{SMALL}[code]\nframe = \"q1 = reg05\"\n"));
    let audio = AudioFrame::default();
    let mut g = Globals::default();
    a.tick(&input(&audio), &mut g);
    assert_eq!(g.regs[5], 7.0);
    let out = b.tick(&input(&audio), &mut g);
    assert_eq!(out.uniforms.q[0][0], 7.0);
}

#[test]
fn nan_sanitised() {
    let toml = format!(
        "{SMALL}[code]\nframe = \"rot = 1e300; q1 = 0/0\"\n\
         vertex = \"dx = sqrt(-1); zoom = 1/0.0000001; sx = if(x > 0.5, 0, 1e-300)\"\n"
    );
    let out = run(&mut instance(&toml), 2);
    for (k, uv) in out.mesh.uv.iter().enumerate() {
        for c in uv {
            assert!(
                c.is_finite() && (-10.0..=10.0).contains(c),
                "vertex {k}: {uv:?}"
            );
        }
    }
    assert!(out.uniforms.q.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn wave_point_code() {
    let toml = "[[wave]]\nsamples = 16\npoint = \"x = sample; y = 0.5\"\n";
    let out = run(&mut instance(toml), 1);
    assert_eq!(out.draw.waves.len(), 1);
    let w = &out.draw.waves[0];
    assert_eq!(w.points.len(), 16);
    assert_eq!(w.colors.len(), 16);
    for (i, p) in w.points.iter().enumerate() {
        assert!(
            close(p[0], i as f64 / 15.0, 1e-6) && p[1] == 0.5,
            "point {i}: {p:?}"
        );
    }
    assert!(w.colors.iter().all(|c| *c == [1.0; 4]));
    assert!(!w.dots && !w.thick && !w.additive);
}

#[test]
fn wave_values_from_waveform() {
    let toml =
        "[[wave]]\nsamples = 3\nusedots = 1\npoint = \"x = sample; y = value1; r = value2\"\n";
    let mut audio = AudioFrame::default();
    for i in 0..576 {
        audio.wave[0][i] = i as f32 / 575.0;
        audio.wave[1][i] = 1.0 - i as f32 / 575.0;
    }
    let mut inst = instance(toml);
    let out = inst.tick(&input(&audio), &mut Globals::default());
    let w = &out.draw.waves[0];
    assert!(w.dots);
    let ys: Vec<f32> = w.points.iter().map(|p| p[1]).collect();
    let rs: Vec<f32> = w.colors.iter().map(|c| c[0]).collect();
    assert_eq!(ys, [0.0, 0.5, 1.0]);
    assert_eq!(rs, [1.0, 0.5, 0.0]);
}

#[test]
fn shape_instances() {
    let toml = "[[shape]]\ninstances = 4\nframe = \"x = instance/4\"\n";
    let out = run(&mut instance(toml), 1);
    let xs: Vec<f32> = out.draw.shapes.iter().map(|s| s.x).collect();
    assert_eq!(xs, [0.0, 0.25, 0.5, 0.75]);
    let s = &out.draw.shapes[0];
    assert_eq!((s.y, s.rad, s.ang, s.sides), (0.5, 0.1, 0.0, 4));
    assert_eq!(s.inner, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(s.outer, [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(s.border, [1.0, 1.0, 1.0, 0.1]);
    assert_eq!(s.tex_zoom, 1.0);
}

#[test]
fn code_error_block_skipped() {
    let toml = format!("{SMALL}[code]\nframe = \"zoom = 2\"\nvertex = \"zoom = nosuchfn(1)\"\n");
    let p = compiled(&toml);
    assert!(p.vertex.is_none() && !p.errors.is_empty());
    let mut inst = PresetInstance::new(p, 1, [192, 144]);
    let out = run(&mut inst, 2);
    assert_eq!(inst.vertex_path(), VertexPath::None);
    assert!(close(out.mesh.uv[0][0], 0.25, 1e-6), "{:?}", out.mesh.uv[0]);
}

#[test]
fn aspect_affects_rad() {
    let toml = format!("{SMALL}[code]\nvertex = \"dx = rad * 0.1\"\n");
    let mut inst = instance(&toml);
    let audio = AudioFrame::default();
    let mut g = Globals::default();
    // Top-left corner: offsets (-0.5, 0.5) from the centre.
    let square = inst.tick(&input(&audio), &mut g).mesh.uv[0][0];
    assert!(close(square, -0.1, 1e-6), "{square}");
    let wide = TickInput {
        aspect: [0.5, 1.0],
        ..input(&audio)
    };
    let u = inst.tick(&wide, &mut g).mesh.uv[0][0];
    let rad = (0.25f64 * 0.25 + 0.5 * 0.5).sqrt() / 0.5f64.sqrt();
    assert!(close(u, -0.1 * rad, 1e-6), "{u}");
}

#[test]
fn disabled_wave_and_shape_emit_nothing() {
    let toml = "[[wave]]\nenabled = false\n[[shape]]\nenabled = false\n";
    let out = run(&mut instance(toml), 1);
    assert!(out.draw.waves.is_empty() && out.draw.shapes.is_empty());
}

#[test]
fn wave_t1_persists() {
    let toml = "[[wave]]\nsamples = 2\nframe = \"t1 = t1 + 1\"\npoint = \"y = t1 * 0.1\"\n";
    let out = run(&mut instance(toml), 3);
    assert!(out.draw.waves[0]
        .points
        .iter()
        .all(|p| close(p[1], 0.3, 1e-6)));
}

#[test]
fn loop_cap_warns_once() {
    let toml = format!("{SMALL}[code]\nframe = \"while(1)\"\n");
    let mut inst = instance(&toml);
    let audio = AudioFrame::default();
    let mut g = Globals::default();
    let first = inst.tick(&input(&audio), &mut g).warnings.clone();
    assert_eq!(first.len(), 1, "{first:?}");
    assert!(first[0].contains("frame"), "{first:?}");
    assert!(inst.tick(&input(&audio), &mut g).warnings.is_empty());
}

#[test]
fn draw_list_from_writables() {
    let out = run(&mut instance(SMALL), 1);
    let mw = out.draw.main_wave.as_ref().unwrap();
    assert_eq!(
        (mw.mode, mw.color, mw.x, mw.y),
        (0, [1.0, 1.0, 1.0, 0.8], 0.5, 0.5)
    );
    assert!(out.draw.motion.is_none());
    assert_eq!(out.draw.borders.outer, (0.01, [0.0; 4]));
    assert_eq!(out.draw.borders.inner, (0.01, [0.25, 0.25, 0.25, 0.0]));

    let toml =
        format!("{SMALL}wave_a = 0.0\nmv_a = 0.5\n[code]\nframe = \"ob_a = 2; ob_r = -1\"\n");
    let out = run(&mut instance(&toml), 1);
    assert!(out.draw.main_wave.is_none());
    let mv = out.draw.motion.unwrap();
    assert_eq!(
        (mv.nx, mv.ny, mv.len, mv.color),
        (12.0, 9.0, 0.9, [1.0, 1.0, 1.0, 0.5])
    );
    assert_eq!(out.draw.borders.outer.1, [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn uniforms_filled() {
    let toml = format!("{SMALL}decay = 0.9\nb2n = 0.25\n[code]\nframe = \"q32 = 3\"\n");
    let mut inst = instance(&toml);
    let audio = AudioFrame {
        treb_att: 0.75,
        ..AudioFrame::default()
    };
    let mut g = Globals::default();
    let a = inst.tick(&input(&audio), &mut g).uniforms;
    let b = inst.tick(&input(&audio), &mut g).uniforms;
    assert_eq!(
        (a.time, a.fps, a.frame, a.progress),
        (1.0, 60.0, 60.0, 0.25)
    );
    assert_eq!((a.treb_att, a.decay), (0.75, 0.9));
    assert_eq!(a.texsize, [640.0, 480.0, 1.0 / 640.0, 1.0 / 480.0]);
    assert_eq!((a.aspect_x, a.aspect_y), (1.0, 1.0));
    assert_eq!(a.blur_min[..3], [0.0, 0.25, 0.0]);
    assert_eq!(a.blur_max[..3], [1.0, 1.0, 1.0]);
    assert_eq!(a.q[7][3], 3.0);
    assert_eq!(a.roam_cos[0], 0.3f64.cos() as f32);
    assert_eq!(a.roam_sin[3], 20f64.sin() as f32);
    assert_ne!(a.rand_frame, b.rand_frame);
    assert_eq!(a.rand_preset, b.rand_preset);
    assert!(a.rand_frame.iter().all(|r| (0.0..1.0).contains(r)));
}

#[test]
fn mesh_capped() {
    let mut inst = PresetInstance::new(compiled(SMALL), 1, [4, 3]);
    let out = run(&mut inst, 1);
    assert_eq!((out.mesh.cols, out.mesh.rows), (4, 3));
    assert_eq!(out.mesh.uv.len(), 5 * 4);
}
