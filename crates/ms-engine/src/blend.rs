//! Preset blending (spec §7.1): while a transition runs, both presets tick
//! and their warp meshes are mixed per vertex with a randomly chosen
//! [`BlendPattern`]. Both warp shaders draw into the one shared feedback
//! chain over the *mixed* mesh, so the incoming preset inherits the current
//! image.
//!
//! # Weights
//!
//! A vertex's weight `w` is the share of the *incoming* preset. With `x`,
//! `y` in [0, 1] preset space (y up) and the progress `p` clamped to [0, 1]:
//! - `Uniform`: `w = p` exactly;
//! - every other pattern: `w = smoothstep(0, 1, p * 2 - f)`, with
//!   `smoothstep(a, b, t) = t' * t' * (3 - 2 * t')` and
//!   `t' = clamp((t - a) / (b - a), 0, 1)`, where `f` in [0, 1] is:
//!   - `RadialIn`: `1 - d`, and `RadialOut`: `d`, with
//!     `d = min(1, dist((x, y), (0.5, 0.5)) / sqrt(0.5))`;
//!   - `Sweep { angle }`:
//!     `clamp(((x - 0.5) * cos(angle) + (y - 0.5) * sin(angle)) / sqrt(0.5) * 0.5 + 0.5, 0, 1)`,
//!     so the change starts on the far side of `angle` and moves along it;
//!   - `Plasma { seed }`: two octaves of seeded value noise (a hashed
//!     lattice, bilinearly interpolated, at scales 4 and 8 with weights 2/3
//!     and 1/3), in [0, 1].
//!
//! Every pattern gives exactly 0 at `p = 0` and exactly 1 at `p = 1`, and
//! `w` never decreases as `p` grows.
//!
//! Non-finite input is never an error: a NaN or infinite `p` counts as 0
//! (the outgoing preset only, so `p = +inf` is 0, not 1), a non-finite
//! coordinate counts as 0, and a weight that would not be finite is 0.
//!
//! # Renderer use during a blend
//!
//! Per tick, [`render_blend`]:
//! 1. builds the weights over the mixed grid ([`blend_weights`]);
//! 2. mixes the two meshes ([`mix_meshes`]);
//! 3. calls `Renderer::warp` with two layers over the mixed mesh: the
//!    outgoing preset's shader and uniforms with weights `1 - w`, then the
//!    incoming preset's with `w` (the warp pass adds the layers, each
//!    scaled by its weight);
//! 4. draws the outgoing overlays with alpha `1 - p`, then the incoming
//!    ones with alpha `p`.
//!
//! Blur levels during a blend are the larger of the two presets'
//! `ShaderSet::blur_levels`; the caller passes that to `Renderer::blur`.
//!
//! Compositing runs at the display rate, separately: [`composite_layers`]
//! gives the two layers for `Renderer::composite`.

use std::f32::consts::TAU;

use ms_expr::Rng;
use ms_render::{
    CompShader, CompositeLayer, Gpu, MeshData, Renderer, Uniforms, WarpLayer, WarpShader,
};

use crate::tick::TickOutput;

/// How the incoming preset's share spreads over the screen during a blend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendPattern {
    /// The whole screen at once.
    Uniform,
    /// From the corners inwards.
    RadialIn,
    /// From the centre outwards.
    RadialOut,
    /// A straight front moving along `angle` (radians, counter-clockwise
    /// from +x in preset space).
    Sweep { angle: f32 },
    /// Blobs of seeded value noise.
    Plasma { seed: u32 },
}

impl BlendPattern {
    /// A pattern chosen uniformly among the five variants; a `Sweep` angle
    /// is uniform in [0, 2π) and a `Plasma` seed is a fresh 32-bit draw.
    pub fn random(rng: &mut Rng) -> Self {
        match rng.next_u64() % 5 {
            0 => BlendPattern::Uniform,
            1 => BlendPattern::RadialIn,
            2 => BlendPattern::RadialOut,
            3 => {
                // An f64 just below 2π can round to 2π in f32.
                let angle = (rng.next_f64() * std::f64::consts::TAU) as f32;
                BlendPattern::Sweep {
                    angle: if angle < TAU { angle } else { 0.0 },
                }
            }
            _ => BlendPattern::Plasma {
                seed: (rng.next_u64() >> 32) as u32,
            },
        }
    }

    /// The incoming preset's weight at preset `(x, y)` for progress `p`
    /// (see the module docs), in [0, 1].
    pub fn weight(&self, x: f32, y: f32, p: f32) -> f32 {
        self.weight_clamped(finite_or_zero(x), finite_or_zero(y), progress(p))
    }

    /// [`weight`](Self::weight) with `p` already clamped to [0, 1] and
    /// finite coordinates.
    fn weight_clamped(&self, x: f32, y: f32, p: f32) -> f32 {
        let f = match *self {
            BlendPattern::Uniform => return p,
            BlendPattern::RadialIn => 1.0 - radial(x, y),
            BlendPattern::RadialOut => radial(x, y),
            BlendPattern::Sweep { angle } => {
                let (s, c) = finite_or_zero(angle).sin_cos();
                let t = ((x - 0.5) * c + (y - 0.5) * s) / SQRT_HALF * 0.5 + 0.5;
                t.clamp(0.0, 1.0)
            }
            BlendPattern::Plasma { seed } => plasma(seed, x, y),
        };
        let w = smoothstep(p * 2.0 - f);
        if w.is_finite() {
            w
        } else {
            0.0
        }
    }
}

const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Blend progress: clamped to [0, 1], with NaN and ±inf counting as 0.
fn progress(p: f32) -> f32 {
    if p.is_finite() {
        p.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// `smoothstep(0, 1, t)`.
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Distance from the centre, 1 in the corners (and beyond).
fn radial(x: f32, y: f32) -> f32 {
    let (dx, dy) = (x - 0.5, y - 0.5);
    ((dx * dx + dy * dy).sqrt() / SQRT_HALF).min(1.0)
}

/// The murmur3 32-bit finaliser.
fn fmix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^ (h >> 16)
}

/// The lattice value at `(ix, iy)`, in [0, 1).
fn lattice(seed: u32, ix: i32, iy: i32) -> f32 {
    let h = fmix(fmix(fmix(seed) ^ ix as u32) ^ iy as u32);
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// One octave of value noise at `scale` lattice cells per unit, in [0, 1).
fn value_noise(seed: u32, x: f32, y: f32, scale: f32) -> f32 {
    let (gx, gy) = (x * scale, y * scale);
    let (fx0, fy0) = (gx.floor(), gy.floor());
    let (fx, fy) = (gx - fx0, gy - fy0);
    // Saturating casts; coordinates are finite.
    let (ix, iy) = (fx0 as i32, fy0 as i32);
    let at = |dx: i32, dy: i32| lattice(seed, ix.wrapping_add(dx), iy.wrapping_add(dy));
    let top = at(0, 0) + (at(1, 0) - at(0, 0)) * fx;
    let bottom = at(0, 1) + (at(1, 1) - at(0, 1)) * fx;
    top + (bottom - top) * fy
}

/// Two-octave value noise in [0, 1].
fn plasma(seed: u32, x: f32, y: f32) -> f32 {
    let coarse = value_noise(seed, x, y, 4.0);
    let fine = value_noise(seed.wrapping_add(0x9E37_79B9), x, y, 8.0);
    (coarse * (2.0 / 3.0) + fine * (1.0 / 3.0)).clamp(0.0, 1.0)
}

/// `i / n`, or 0 for an empty axis.
fn frac(i: u32, n: u32) -> f32 {
    if n == 0 {
        0.0
    } else {
        i as f32 / n as f32
    }
}

/// The incoming preset's weight at every vertex of a `cols x rows` mesh:
/// `(cols + 1) * (rows + 1)` values, row-major from the top row, vertex
/// `(col, row)` at preset `x = col / cols`, `y = 1 - row / rows` (as in
/// [`PresetInstance`](crate::PresetInstance)). A zero `cols` or `rows`
/// puts its single column at `x = 0` or row at `y = 1`.
pub fn blend_weights(pattern: &BlendPattern, cols: u32, rows: u32, p: f32) -> Vec<f32> {
    let mut out = Vec::new();
    fill_weights(&mut out, pattern, cols, rows, p);
    out
}

fn fill_weights(out: &mut Vec<f32>, pattern: &BlendPattern, cols: u32, rows: u32, p: f32) {
    let p = progress(p);
    out.clear();
    out.reserve((cols as usize + 1) * (rows as usize + 1));
    for row in 0..=rows {
        let y = 1.0 - frac(row, rows);
        for col in 0..=cols {
            out.push(pattern.weight_clamped(frac(col, cols), y, p));
        }
    }
}

/// Mixes two warp meshes per vertex: `uv = a.uv * (1 - w) + b.uv * w`, and
/// the same for the colour, with `w[k]` the incoming weight of vertex `k`.
///
/// The result has `max(a.cols, b.cols) x max(a.rows, b.rows)` cells. A mesh
/// on a different grid (normally the smaller one) is first resampled to it
/// by bilinear interpolation over its own vertex grid (both grids span the
/// same [0, 1] square).
///
/// Never panics on malformed input:
/// - `w` should have one value per vertex of the mixed grid; a missing value
///   counts as 0 (the outgoing mesh), a non-finite one counts as 0 and the
///   rest are clamped to [0, 1];
/// - a mesh whose `uv` or `color` is shorter than its grid uses the identity
///   UV and white for the missing vertices, and non-finite UV or colour
///   values count as 0.
pub fn mix_meshes(a: &MeshData, b: &MeshData, w: &[f32]) -> MeshData {
    let mut out = MeshData::default();
    mix_into(&mut out, a, b, w);
    out
}

fn mix_into(out: &mut MeshData, a: &MeshData, b: &MeshData, w: &[f32]) {
    let (cols, rows) = (a.cols.max(b.cols), a.rows.max(b.rows));
    let n = (cols as usize + 1) * (rows as usize + 1);
    out.cols = cols;
    out.rows = rows;
    out.uv.clear();
    out.color.clear();
    out.uv.reserve(n);
    out.color.reserve(n);
    let mut k = 0;
    for row in 0..=rows {
        for col in 0..=cols {
            let wk = w.get(k).copied().map_or(0.0, progress);
            let (ua, ca) = sample(a, cols, rows, col, row);
            let (ub, cb) = sample(b, cols, rows, col, row);
            out.uv
                .push(std::array::from_fn(|i| ua[i] * (1.0 - wk) + ub[i] * wk));
            out.color
                .push(std::array::from_fn(|i| ca[i] * (1.0 - wk) + cb[i] * wk));
            k += 1;
        }
    }
}

type Vertex = ([f32; 2], [f32; 4]);

/// Vertex `(col, row)` of `m`'s own grid, with the fallbacks of
/// [`mix_meshes`].
fn vertex(m: &MeshData, col: u32, row: u32) -> Vertex {
    let k = row as usize * (m.cols as usize + 1) + col as usize;
    let uv =
        m.uv.get(k)
            .copied()
            .unwrap_or([frac(col, m.cols), frac(row, m.rows)]);
    let color = m.color.get(k).copied().unwrap_or([1.0; 4]);
    (uv.map(finite_or_zero), color.map(finite_or_zero))
}

/// `m` at vertex `(col, row)` of a `cols x rows` grid: read directly when
/// `m` is on that grid, bilinearly interpolated otherwise.
fn sample(m: &MeshData, cols: u32, rows: u32, col: u32, row: u32) -> Vertex {
    if (m.cols, m.rows) == (cols, rows) {
        return vertex(m, col, row);
    }
    let axis = |i: u32, n: u32, own: u32| {
        let s = frac(i, n) * own as f32;
        let i0 = (s.floor() as u32).min(own);
        let i1 = (i0 + 1).min(own);
        (i0, i1, (s - i0 as f32).clamp(0.0, 1.0))
    };
    let (c0, c1, fx) = axis(col, cols, m.cols);
    let (r0, r1, fy) = axis(row, rows, m.rows);
    let (v00, v10) = (vertex(m, c0, r0), vertex(m, c1, r0));
    let (v01, v11) = (vertex(m, c0, r1), vertex(m, c1, r1));
    let bilerp = |p00: f32, p10: f32, p01: f32, p11: f32| {
        let top = p00 + (p10 - p00) * fx;
        let bottom = p01 + (p11 - p01) * fx;
        top + (bottom - top) * fy
    };
    (
        std::array::from_fn(|i| bilerp(v00.0[i], v10.0[i], v01.0[i], v11.0[i])),
        std::array::from_fn(|i| bilerp(v00.1[i], v10.1[i], v01.1[i], v11.1[i])),
    )
}

/// Buffers [`render_blend`] reuses from tick to tick, so a blend allocates
/// nothing once they have grown to the mesh size.
#[derive(Debug, Default)]
pub struct BlendScratch {
    /// Incoming weights `w`.
    incoming: Vec<f32>,
    /// Outgoing weights `1 - w`.
    outgoing: Vec<f32>,
    mesh: MeshData,
}

/// Renders one blend tick into the shared feedback chain (see the module
/// docs): the weights of `pattern` at progress `p` over the mixed grid, the
/// mixed mesh, one `warp` call with the outgoing (`1 - w`) and incoming
/// (`w`) layers, each with its own shader and uniforms, then the outgoing
/// overlays at alpha `1 - p` and the incoming ones at alpha `p`.
///
/// `from` is the outgoing preset's tick output and warp shader, `to` the
/// incoming one's. A non-finite `p` counts as 0. Does not block.
pub fn render_blend(
    renderer: &mut Renderer,
    gpu: &Gpu,
    from: (&TickOutput, &WarpShader),
    to: (&TickOutput, &WarpShader),
    pattern: &BlendPattern,
    p: f32,
    scratch: &mut BlendScratch,
) {
    let p = progress(p);
    let (a, b) = (&from.0.mesh, &to.0.mesh);
    let (cols, rows) = (a.cols.max(b.cols), a.rows.max(b.rows));
    fill_weights(&mut scratch.incoming, pattern, cols, rows, p);
    mix_into(&mut scratch.mesh, a, b, &scratch.incoming);
    scratch.outgoing.clear();
    scratch
        .outgoing
        .extend(scratch.incoming.iter().map(|w| 1.0 - w));
    renderer.warp(
        gpu,
        &[
            WarpLayer {
                mesh: &scratch.mesh,
                weight: &scratch.outgoing,
                shader: from.1,
                uniforms: &from.0.uniforms,
            },
            WarpLayer {
                mesh: &scratch.mesh,
                weight: &scratch.incoming,
                shader: to.1,
                uniforms: &to.0.uniforms,
            },
        ],
    );
    renderer.draw(gpu, &from.0.draw, 1.0 - p);
    renderer.draw(gpu, &to.0.draw, p);
}

/// The two layers for `Renderer::composite` during a blend: the outgoing
/// preset's composite shader and uniforms with alpha `1 - p`, then the
/// incoming one's with alpha `p` (a non-finite `p` counts as 0).
///
/// The composite pass draws its first layer opaque (its alpha is ignored)
/// and mixes each later layer over the result as `src * alpha + dst *
/// (1 - alpha)`. So the target ends up as `A` after the first layer and
/// `B * p + A * (1 - p)` after the second: the cross-fade
/// `(1 - p) * A + p * B`. The first layer's `1 - p` is informational.
pub fn composite_layers<'a>(
    from: (&'a CompShader, &'a Uniforms),
    to: (&'a CompShader, &'a Uniforms),
    p: f32,
) -> [CompositeLayer<'a>; 2] {
    let p = progress(p);
    [
        CompositeLayer {
            shader: from.0,
            uniforms: from.1,
            alpha: 1.0 - p,
        },
        CompositeLayer {
            shader: to.0,
            uniforms: to.1,
            alpha: p,
        },
    ]
}
