//! A running preset: its variables, the per-tick execution order (§5.2) and
//! the plain render data each tick produces for `ms-render`.
//!
//! # Vertex conventions
//!
//! The warp mesh has `cols x rows` cells, so `(cols + 1) * (rows + 1)`
//! vertices, stored row-major from the top row. Vertex `(row, col)` runs the
//! `vertex` code with:
//! - `x = col / cols` and `y = 1 - row / rows` (preset space: y from the
//!   bottom);
//! - `rad = sqrt(ax^2 + ay^2) / sqrt(0.5)` and `ang = atan2(ay, ax)`, where
//!   `ax = (x - 0.5) * aspect[0]` and `ay = (y - 0.5) * aspect[1]`, so `rad`
//!   is 1 in the corners of a square view.
//!
//! # Warp UV
//!
//! Each vertex samples the previous frame at a position computed from the
//! motion variables it ended with (an original approximation of MilkDrop's
//! warp, to be tuned in sub-project 4), in preset space:
//! 1. `u = (x - cx) / sx`, `v = (y - cy) / sy`;
//! 2. zoom: `z = zoom ^ (zoomexp ^ (rad * 2 - 1))`, then `u /= z`, `v /= z`;
//! 3. warp wobble, with `w = warp * 0.0035`, `t = time` and `u0`, `v0` the
//!    values from step 2:
//!    `u = u0 + w * (sin(t * 0.333 + v0 * 12) + cos(t * 0.753 - u0 * 8))`,
//!    `v = v0 + w * (cos(t * 0.375 - u0 * 12) + sin(t * 0.825 + v0 * 8))`;
//! 4. rotate by `rot` radians, counter-clockwise in preset space;
//! 5. translate: `su = u + cx - dx`, `sv = v + cy - dy`. A positive `dx`
//!    makes every vertex sample further left, so the image moves right
//!    (and a positive `dy` moves it up);
//! 6. texture space: `uv = (su, 1 - sv)`, with NaN/Inf mapped to 0 and both
//!    coordinates clamped to [-10, 10].
//!
//! With `zoom = zoomexp = sx = sy = 1`, `rot = warp = dx = dy = 0` this is
//! the identity: `uv = (x, 1 - y)`.
//!
//! # Batching
//!
//! The `vertex` code runs on the batched VM, with mesh rows split into
//! chunks across rayon, when it is batchable and reads no carried variable
//! (user variable, `regN` or `tN`; see [`crate::env::is_carried`]) before
//! writing it. Otherwise it runs on the scalar VM in row-major order, user
//! variables carrying from vertex to vertex.
//!
//! # Allocation
//!
//! [`PresetInstance::tick`] returns a reference to output buffers owned by
//! the instance. After the first tick, ticking reuses every buffer (mesh,
//! draw list, batching lanes and VM scratch) and does not allocate.

use std::sync::Arc;

use ms_expr::{BatchScratch, BatchShared, Lanes, Megabuf, Program, Rng};
use ms_preset::CompiledPreset;
use ms_render::{Borders, DrawList, MainWave, MeshData, MotionVectors, Uniforms};
use rayon::prelude::*;

use crate::env::*;
use crate::objects::{ShapeState, WaveState};

/// The render data of one tick.
#[derive(Debug, Clone, Default)]
pub struct TickOutput {
    pub mesh: MeshData,
    pub draw: DrawList,
    pub uniforms: Uniforms,
    /// New problems found this tick (each loop-cap warning appears once per
    /// block per instance lifetime).
    pub warnings: Vec<String>,
}

/// How the `vertex` code ran on the latest tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexPath {
    /// No vertex code (empty or failed to compile).
    None,
    /// Scalar VM, row-major, user variables carried between vertices.
    Scalar,
    /// Batched VM over row chunks.
    Batched,
}

/// Fewest lanes in a batching chunk (chunks hold whole rows).
const MIN_CHUNK_LANES: usize = 256;

/// A value no program can produce or observe: a signalling NaN. Carried
/// columns start with it in the batched path, so a lane still holding it
/// did not write that variable.
const UNWRITTEN: u64 = 0x7FF4_D15C_A4D0_0001;

/// Frequencies of `roam_cos`/`roam_sin`: `cos/sin(time * ROAM[i])`, an
/// approximation of MilkDrop's slowly roaming values.
const ROAM: [f64; 4] = [0.3, 1.3, 5.0, 20.0];

/// Per-vertex inputs.
#[derive(Debug, Clone, Copy, Default)]
struct Geo {
    x: f64,
    y: f64,
    rad: f64,
    ang: f64,
}

/// The motion variables, in [`MOTION`] order.
type Motion = [f64; 10];
const MOTION: [usize; 10] = [ZOOM, ZOOMEXP, ROT, WARP, CX, CY, DX, DY, SX, SY];

fn motion_of(vars: &[f64]) -> Motion {
    MOTION.map(|id| get(vars, id))
}

/// The warp formula from the module docs.
fn warp_uv(m: &Motion, g: &Geo, t: f64) -> [f32; 2] {
    let [zoom, zoomexp, rot, warp, cx, cy, dx, dy, sx, sy] = *m;
    let z = zoom.powf(zoomexp.powf(g.rad * 2.0 - 1.0));
    let u0 = (g.x - cx) / sx / z;
    let v0 = (g.y - cy) / sy / z;
    let w = warp * 0.0035;
    let u = u0 + w * ((t * 0.333 + v0 * 12.0).sin() + (t * 0.753 - u0 * 8.0).cos());
    let v = v0 + w * ((t * 0.375 - u0 * 12.0).cos() + (t * 0.825 + v0 * 8.0).sin());
    let (s, c) = rot.sin_cos();
    let su = u * c - v * s + cx - dx;
    let sv = u * s + v * c + cy - dy;
    [uv_coord(su), uv_coord(1.0 - sv)]
}

fn uv_coord(v: f64) -> f32 {
    gpu(v).clamp(-10.0, 10.0)
}

/// What a batched column holds before the program runs.
#[derive(Debug, Clone, Copy)]
enum Fill {
    X,
    Y,
    Rad,
    Ang,
    /// The post-frame value of this main-table variable.
    Frame(usize),
    /// A carried variable (written before read): starts as [`UNWRITTEN`].
    Carried(usize),
}

/// One batching chunk's lanes and scratch, reused every tick.
struct Chunk {
    lanes: Lanes,
    scratch: BatchScratch,
    // Batchable programs never touch these; `run_batched_with` needs them.
    megabuf: Megabuf,
    gmegabuf: Megabuf,
    rng: Rng,
    capped: bool,
}

struct Batched {
    /// The vertex program with its variables renumbered to lane columns.
    prog: Program,
    fills: Vec<Fill>,
    /// Lane column of each motion variable the program uses.
    motion_cols: [Option<usize>; 10],
    chunks: Vec<Chunk>,
}

enum Plan {
    None,
    Scalar {
        /// Built-ins restored to their post-frame value before each vertex.
        restore: Vec<usize>,
        /// Each vertex's motion values.
        motion: Vec<Motion>,
    },
    Batched(Box<Batched>),
}

#[derive(Default)]
struct Warned {
    init: bool,
    frame: bool,
    vertex: bool,
}

/// One preset running: its variables, which persist across ticks, and the
/// output buffers each tick refills.
pub struct PresetInstance {
    p: Arc<CompiledPreset>,
    vars: Vec<f64>,
    /// Built-in values after the frame code (scalar vertex path).
    frame_vals: Vec<f64>,
    /// Writable variables and their `[params]` values.
    resets: Vec<(usize, f64)>,
    q: [f64; 32],
    megabuf: Megabuf,
    rng: Rng,
    rand_preset: [f32; 4],
    mesh: [u32; 2],
    /// Lanes per batching chunk (whole rows).
    chunk_len: usize,
    geo: Vec<Geo>,
    /// The aspect `geo` was computed for (bit patterns).
    geo_aspect: Option<[u64; 2]>,
    plan: Plan,
    path: VertexPath,
    waves: Vec<WaveState>,
    shapes: Vec<ShapeState>,
    warned: Warned,
    /// Warnings from `new`, reported by the first tick.
    pending: Vec<String>,
    out: TickOutput,
}

impl PresetInstance {
    /// Creates the instance and runs the preset's `init` code (§5.2): after
    /// the `[params]` values are applied, with time and audio inputs 0. Its
    /// `q1`..`q32` become the initial `q` snapshot and the user variables it
    /// sets persist. `init` sees `reg00`..`reg99` as 0 and a private
    /// `gmegabuf` (there are no [`Globals`] yet); the waves' and shapes'
    /// `init` code runs next with that `q`.
    ///
    /// The mesh is `[params] mesh`, capped per axis by `mesh_cap` (at least
    /// 1x1).
    pub fn new(p: Arc<CompiledPreset>, seed: u64, mesh_cap: [u32; 2]) -> Self {
        let params = &p.source.params;
        let mesh = [
            params.mesh[0].min(mesh_cap[0]).max(1),
            params.mesh[1].min(mesh_cap[1]).max(1),
        ];
        let resets: Vec<(usize, f64)> = ms_preset::PARAM_DEFAULTS
            .iter()
            .filter_map(|&(name, _)| {
                let id = p.vars.lookup(name)?.0 as usize;
                Some((id, params.param(name)))
            })
            .collect();
        let mut vars = vec![0.0; p.vars.len().max(BUILTIN_LEN)];
        set(&mut vars, MESHX, f64::from(mesh[0]));
        set(&mut vars, MESHY, f64::from(mesh[1]));
        for &(id, v) in &resets {
            set(&mut vars, id, v);
        }
        let mut rng = Rng::seeded(seed);
        let mut megabuf = Megabuf::new();
        let mut init_gmegabuf = Megabuf::new();
        let mut warned = Warned::default();
        let mut pending = Vec::new();
        let capped = run(
            p.init.as_ref(),
            &mut vars,
            &mut megabuf,
            &mut init_gmegabuf,
            &mut rng,
        );
        note_cap(capped, &mut warned.init, "init", &mut pending);
        let mut q = [0.0; 32];
        q.copy_from_slice(&vars[Q1..Q1 + 32]);
        let rand_preset = [0; 4].map(|_| rng.next_f64() as f32);

        let mut waves = Vec::new();
        for (i, w) in p.waves.iter().enumerate().filter(|(_, w)| w.enabled) {
            let mut ws = WaveState::new(i, w);
            ws.init(
                w,
                &q,
                &mut megabuf,
                &mut init_gmegabuf,
                &mut rng,
                &mut pending,
            );
            waves.push(ws);
        }
        let mut shapes = Vec::new();
        for (i, s) in p.shapes.iter().enumerate().filter(|(_, s)| s.enabled) {
            let mut ss = ShapeState::new(i, s);
            ss.init(
                s,
                &q,
                &mut megabuf,
                &mut init_gmegabuf,
                &mut rng,
                &mut pending,
            );
            shapes.push(ss);
        }

        let stride = mesh[0] as usize + 1;
        let nrows = mesh[1] as usize + 1;
        let n = stride * nrows;
        let threads = rayon::current_num_threads().max(1);
        let rows_per_chunk = MIN_CHUNK_LANES
            .div_ceil(stride)
            .max(nrows.div_ceil(threads * 2))
            .min(nrows);
        let chunk_len = rows_per_chunk * stride;
        let plan = plan_vertex(&p, n, chunk_len);

        let out = TickOutput {
            mesh: MeshData {
                cols: mesh[0],
                rows: mesh[1],
                uv: vec![[0.0; 2]; n],
                color: vec![[1.0; 4]; n],
            },
            draw: DrawList {
                waves: vec![Default::default(); waves.len()],
                ..Default::default()
            },
            ..Default::default()
        };
        PresetInstance {
            frame_vals: vec![0.0; BUILTIN_LEN],
            p,
            vars,
            resets,
            q,
            megabuf,
            rng,
            rand_preset,
            mesh,
            chunk_len,
            geo: vec![Geo::default(); n],
            geo_aspect: None,
            plan,
            path: VertexPath::None,
            waves,
            shapes,
            warned,
            pending,
            out,
        }
    }

    /// How the `vertex` code ran on the latest tick.
    pub fn vertex_path(&self) -> VertexPath {
        self.path
    }

    /// Runs one tick (§5.2) and returns its render data, which stays valid
    /// until the next call.
    pub fn tick(&mut self, input: &TickInput, g: &mut Globals) -> &TickOutput {
        self.out.warnings.clear();
        self.out.warnings.append(&mut self.pending);

        // 1-5: inputs, params, regs, frame code, q snapshot.
        write_inputs(&mut self.vars, input, self.mesh);
        for &(id, v) in &self.resets {
            set(&mut self.vars, id, v);
        }
        regs_in(&mut self.vars, &g.regs);
        let capped = run(
            self.p.frame.as_ref(),
            &mut self.vars,
            &mut self.megabuf,
            &mut g.gmegabuf,
            &mut self.rng,
        );
        note_cap(
            capped,
            &mut self.warned.frame,
            "frame",
            &mut self.out.warnings,
        );
        self.q.copy_from_slice(&self.vars[Q1..Q1 + 32]);

        // 6: vertex code and the warp mesh.
        self.update_geo(input.aspect);
        let capped = self.vertex_pass(input.time, g);
        note_cap(
            capped,
            &mut self.warned.vertex,
            "vertex",
            &mut self.out.warnings,
        );
        regs_out(&self.vars, &mut g.regs);

        // 7: waves and shapes, then 8: their regs go back to `g` as they run.
        let audio = input.audio;
        for (w, draw) in self.waves.iter_mut().zip(self.out.draw.waves.iter_mut()) {
            if let Some(cw) = self.p.waves.get(w.index()) {
                w.tick(
                    cw,
                    input,
                    self.mesh,
                    &self.q,
                    &mut self.megabuf,
                    g,
                    &mut self.rng,
                    draw,
                    &mut self.out.warnings,
                );
            }
        }
        self.out.draw.shapes.clear();
        for s in &mut self.shapes {
            if let Some(cs) = self.p.shapes.get(s.index()) {
                s.tick(
                    cs,
                    input,
                    self.mesh,
                    &self.q,
                    &mut self.megabuf,
                    g,
                    &mut self.rng,
                    &mut self.out.draw.shapes,
                    &mut self.out.warnings,
                );
            }
        }

        self.fill_draw(audio);
        self.fill_uniforms(input);
        &self.out
    }

    /// Recomputes the per-vertex inputs when the aspect changes.
    fn update_geo(&mut self, aspect: [f64; 2]) {
        let key = aspect.map(f64::to_bits);
        if self.geo_aspect == Some(key) {
            return;
        }
        self.geo_aspect = Some(key);
        let (cols, rows) = (self.mesh[0] as usize, self.mesh[1] as usize);
        let norm = 0.5f64.sqrt();
        for (k, geo) in self.geo.iter_mut().enumerate() {
            let (row, col) = (k / (cols + 1), k % (cols + 1));
            let x = col as f64 / cols as f64;
            let y = 1.0 - row as f64 / rows as f64;
            let ax = (x - 0.5) * aspect[0];
            let ay = (y - 0.5) * aspect[1];
            *geo = Geo {
                x,
                y,
                rad: (ax * ax + ay * ay).sqrt() / norm,
                ang: ay.atan2(ax),
            };
        }
    }

    /// Runs the vertex code (if any) and fills the mesh UVs. Returns true if
    /// the loop cap was hit.
    fn vertex_pass(&mut self, t: f64, g: &mut Globals) -> bool {
        let chunk_len = self.chunk_len;
        let geo = &self.geo[..];
        let uv = &mut self.out.mesh.uv[..];
        match &mut self.plan {
            Plan::None => {
                self.path = VertexPath::None;
                let m = motion_of(&self.vars);
                fill_uv(uv, geo, chunk_len, t, |_| m);
                false
            }
            Plan::Scalar { restore, motion } => {
                self.path = VertexPath::Scalar;
                let vars = &mut self.vars[..];
                self.frame_vals.copy_from_slice(&vars[..BUILTIN_LEN]);
                let prog = self.p.vertex.as_ref();
                let mut capped = false;
                for (geo, m) in geo.iter().zip(motion.iter_mut()) {
                    for &id in restore.iter() {
                        vars[id] = self.frame_vals[id];
                    }
                    vars[X] = geo.x;
                    vars[Y] = geo.y;
                    vars[RAD] = geo.rad;
                    vars[ANG] = geo.ang;
                    capped |= run(
                        prog,
                        vars,
                        &mut self.megabuf,
                        &mut g.gmegabuf,
                        &mut self.rng,
                    );
                    *m = motion_of(vars);
                }
                // Leave the post-frame values for the waves, shapes,
                // uniforms and draw list.
                for &id in restore.iter().chain(&[X, Y, RAD, ANG]) {
                    vars[id] = self.frame_vals[id];
                }
                let motion = &motion[..];
                fill_uv(uv, geo, chunk_len, t, |k| {
                    motion.get(k).copied().unwrap_or_default()
                });
                capped
            }
            Plan::Batched(b) => {
                self.path = VertexPath::Batched;
                let vars = &self.vars[..];
                let frame_motion = motion_of(vars);
                let Batched {
                    prog,
                    fills,
                    motion_cols,
                    chunks,
                } = &mut **b;
                let job = |c: &mut Chunk, base: usize, uv: &mut [[f32; 2]]| {
                    let geo = geo.get(base..base + uv.len()).unwrap_or_default();
                    run_chunk(c, prog, fills, motion_cols, geo, vars, &frame_motion, t, uv);
                };
                if let [c] = &mut chunks[..] {
                    job(c, 0, uv);
                } else {
                    chunks
                        .par_iter_mut()
                        .zip(uv.par_chunks_mut(chunk_len))
                        .enumerate()
                        .for_each(|(i, (c, uv))| job(c, i * chunk_len, uv));
                }
                // Carried variables keep the last writer's value, in
                // row-major order, as on the scalar path.
                for (col, fill) in fills.iter().enumerate() {
                    if let Fill::Carried(id) = *fill {
                        let last = chunks.iter().rev().find_map(|c| {
                            let v = c.lanes.cols.get(col)?.get(..c.lanes.n)?;
                            v.iter().rev().find(|x| x.to_bits() != UNWRITTEN).copied()
                        });
                        if let Some(v) = last {
                            set(&mut self.vars, id, v);
                        }
                    }
                }
                chunks.iter().any(|c| c.capped)
            }
        }
    }

    fn fill_draw(&mut self, audio: &ms_analysis::AudioFrame) {
        let v = |id| get(&self.vars, id);
        let d = &mut self.out.draw;
        d.main_wave = (v(WAVE_A) > 0.0).then(|| MainWave {
            mode: (gpu(v(WAVE_MODE)).floor() as i64).rem_euclid(8) as u8,
            wave: audio.wave,
            spectrum: audio.spectrum,
            color: [
                unit(v(WAVE_R)),
                unit(v(WAVE_G)),
                unit(v(WAVE_B)),
                unit(v(WAVE_A)),
            ],
            x: gpu(v(WAVE_X)),
            y: gpu(v(WAVE_Y)),
            mystery: gpu(v(WAVE_MYSTERY)),
            dots: flag(v(WAVE_USEDOTS)),
            thick: flag(v(WAVE_THICK)),
            additive: flag(v(WAVE_ADDITIVE)),
            brighten: flag(v(WAVE_BRIGHTEN)),
        });
        d.motion = (v(MV_A) > 0.0).then(|| MotionVectors {
            nx: gpu(v(MV_X)),
            ny: gpu(v(MV_Y)),
            dx: gpu(v(MV_DX)),
            dy: gpu(v(MV_DY)),
            len: gpu(v(MV_L)),
            color: [unit(v(MV_R)), unit(v(MV_G)), unit(v(MV_B)), unit(v(MV_A))],
        });
        d.borders = Borders {
            outer: (
                gpu(v(OB_SIZE)),
                [unit(v(OB_R)), unit(v(OB_G)), unit(v(OB_B)), unit(v(OB_A))],
            ),
            inner: (
                gpu(v(IB_SIZE)),
                [unit(v(IB_R)), unit(v(IB_G)), unit(v(IB_B)), unit(v(IB_A))],
            ),
        };
    }

    fn fill_uniforms(&mut self, input: &TickInput) {
        let v = |id| gpu(get(&self.vars, id));
        let a = input.audio;
        let au = |x: f32| gpu(f64::from(x));
        let [w, h] = input.pixels.map(|p| p as f32);
        let inv = |s: f32| if s > 0.0 { 1.0 / s } else { 0.0 };
        let mut q = [[0.0f32; 4]; 8];
        for (i, &x) in self.q.iter().enumerate() {
            q[i / 4][i % 4] = gpu(x);
        }
        let rand_frame = [0; 4].map(|_| self.rng.next_f64() as f32);
        self.out.uniforms = Uniforms {
            time: gpu(input.time),
            fps: gpu(input.fps),
            frame: gpu(input.frame as f64),
            progress: gpu(input.progress),
            bass: au(a.bass),
            mid: au(a.mid),
            treb: au(a.treb),
            beat: au(a.beat),
            bass_att: au(a.bass_att),
            mid_att: au(a.mid_att),
            treb_att: au(a.treb_att),
            beat_strength: au(a.beat_strength),
            decay: v(DECAY),
            gamma: v(GAMMA),
            echo_zoom: v(ECHO_ZOOM),
            echo_alpha: v(ECHO_ALPHA),
            echo_orient: v(ECHO_ORIENT),
            invert: v(INVERT),
            solarize: v(SOLARIZE),
            brighten: v(BRIGHTEN),
            darken: v(DARKEN),
            darken_center: v(DARKEN_CENTER),
            aspect_x: gpu(input.aspect[0]),
            aspect_y: gpu(input.aspect[1]),
            texsize: [w, h, inv(w), inv(h)],
            rand_frame,
            rand_preset: self.rand_preset,
            roam_cos: ROAM.map(|f| gpu((input.time * f).cos())),
            roam_sin: ROAM.map(|f| gpu((input.time * f).sin())),
            blur_min: [v(B1N), v(B2N), v(B3N), 0.0],
            blur_max: [v(B1X), v(B2X), v(B3X), 1.0],
            q,
        };
    }
}

/// Chooses the vertex path (see the module docs) and allocates its buffers.
fn plan_vertex(p: &CompiledPreset, n: usize, chunk_len: usize) -> Plan {
    let Some(vp) = p.vertex.as_ref() else {
        return Plan::None;
    };
    let carried_read = vp
        .reads_before_write()
        .iter()
        .any(|v| is_carried(v.0 as usize));
    if !vp.batchable() || carried_read {
        return Plan::Scalar {
            restore: restored_writes(vp),
            motion: vec![[0.0; 10]; n],
        };
    }
    let (prog, map) = vp.compact_vars();
    let fills: Vec<Fill> = map
        .iter()
        .map(|v| match v.0 as usize {
            X => Fill::X,
            Y => Fill::Y,
            RAD => Fill::Rad,
            ANG => Fill::Ang,
            id if is_carried(id) => Fill::Carried(id),
            id => Fill::Frame(id),
        })
        .collect();
    let motion_cols = MOTION.map(|id| map.iter().position(|v| v.0 as usize == id));
    let chunks = (0..n.div_ceil(chunk_len))
        .map(|_| Chunk {
            lanes: Lanes {
                n: 0,
                cols: vec![vec![0.0; chunk_len]; fills.len()],
            },
            scratch: BatchScratch::default(),
            megabuf: Megabuf::new(),
            gmegabuf: Megabuf::new(),
            rng: Rng::seeded(0),
            capped: false,
        })
        .collect();
    Plan::Batched(Box::new(Batched {
        prog,
        fills,
        motion_cols,
        chunks,
    }))
}

/// Fills `uv` from per-vertex motion values, in parallel over chunks.
fn fill_uv(
    uv: &mut [[f32; 2]],
    geo: &[Geo],
    chunk_len: usize,
    t: f64,
    motion: impl Fn(usize) -> Motion + Sync,
) {
    let job = |base: usize, uv: &mut [[f32; 2]]| {
        for (i, (out, g)) in uv
            .iter_mut()
            .zip(geo.get(base..).unwrap_or_default())
            .enumerate()
        {
            *out = warp_uv(&motion(base + i), g, t);
        }
    };
    if uv.len() <= chunk_len {
        job(0, uv);
    } else {
        uv.par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(i, uv)| job(i * chunk_len, uv));
    }
}

/// Runs one batching chunk: fills its lanes, runs the program, writes the
/// chunk's UVs.
#[allow(clippy::too_many_arguments)]
fn run_chunk(
    c: &mut Chunk,
    prog: &Program,
    fills: &[Fill],
    motion_cols: &[Option<usize>; 10],
    geo: &[Geo],
    vars: &[f64],
    frame_motion: &Motion,
    t: f64,
    uv: &mut [[f32; 2]],
) {
    let n = geo.len().min(uv.len());
    c.lanes.n = n;
    for (col, fill) in c.lanes.cols.iter_mut().zip(fills) {
        let len = n.min(col.len());
        let col = &mut col[..len];
        match *fill {
            Fill::X => col.iter_mut().zip(geo).for_each(|(d, g)| *d = g.x),
            Fill::Y => col.iter_mut().zip(geo).for_each(|(d, g)| *d = g.y),
            Fill::Rad => col.iter_mut().zip(geo).for_each(|(d, g)| *d = g.rad),
            Fill::Ang => col.iter_mut().zip(geo).for_each(|(d, g)| *d = g.ang),
            Fill::Frame(id) => col.fill(get(vars, id)),
            Fill::Carried(_) => col.fill(f64::from_bits(UNWRITTEN)),
        }
    }
    let mut shared = BatchShared {
        megabuf: &mut c.megabuf,
        gmegabuf: &mut c.gmegabuf,
        rng: &mut c.rng,
    };
    c.capped = prog.run_batched_with(&mut c.lanes, &mut shared, &mut c.scratch);
    let cols = &c.lanes.cols;
    for (i, (out, g)) in uv.iter_mut().zip(geo).enumerate() {
        let mut m = *frame_motion;
        for (slot, col) in m.iter_mut().zip(motion_cols) {
            if let Some(v) = col.and_then(|c| cols.get(c)).and_then(|c| c.get(i)) {
                *slot = *v;
            }
        }
        *out = warp_uv(&m, g, t);
    }
}
