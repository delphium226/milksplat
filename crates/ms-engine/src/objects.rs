//! Custom waves and shapes (§5.2 step 7, §6.4).
//!
//! Each wave and shape has its own variable table. Every tick (and, for a
//! shape, every instance) its writable variables are reset to the
//! definition's values (or the defaults below), the read-only inputs and
//! the main preset's `q` snapshot are copied in, and its `frame` code runs.
//! `t1`..`t8` and user variables are never reset, so they persist from
//! `init` through every tick. `init` runs once, on the instance's first
//! tick. `reg00`..`reg99` are copied from and back to [`Globals`] around
//! each run when the code uses them.
//!
//! Wave `point` code runs once per sample, on the scalar VM: each point
//! starts from the post-frame values (user variables and `t1`..`t8` carry
//! from point to point), with `sample = i / (samples - 1)` (0 for a single
//! sample) and `value1`/`value2` the left/right waveform resampled linearly
//! to `samples` points.

use ms_expr::{Megabuf, Rng};
use ms_preset::{CompiledShape, CompiledWave};
use ms_render::{ShapeDraw, WaveDraw};

use crate::env::*;

/// Wave writables and their defaults (definition values override them).
const WAVE_DEFAULTS: &[(&str, f64)] = &[
    ("r", 1.0),
    ("g", 1.0),
    ("b", 1.0),
    ("a", 1.0),
    ("x", 0.5),
    ("y", 0.5),
    ("usedots", 0.0),
    ("thick", 0.0),
    ("additive", 0.0),
];

/// Shape writables and their defaults (MilkDrop-like: a red centre fading
/// to a transparent green rim, with a faint white border). `sides` comes
/// from the definition.
const SHAPE_DEFAULTS: &[(&str, f64)] = &[
    ("x", 0.5),
    ("y", 0.5),
    ("rad", 0.1),
    ("ang", 0.0),
    ("r", 1.0),
    ("g", 0.0),
    ("b", 0.0),
    ("a", 1.0),
    ("r2", 0.0),
    ("g2", 1.0),
    ("b2", 0.0),
    ("a2", 0.0),
    ("border_r", 1.0),
    ("border_g", 1.0),
    ("border_b", 1.0),
    ("border_a", 0.1),
    ("additive", 0.0),
    ("thickoutline", 0.0),
    ("textured", 0.0),
    ("tex_zoom", 1.0),
    ("tex_ang", 0.0),
];

/// Most points a wave draws.
const MAX_SAMPLES: u32 = 512;

/// (id, value) for each writable in `defaults`, overridden by `params`.
fn resets(
    vars: &ms_expr::VarTable,
    defaults: &[(&str, f64)],
    params: &std::collections::BTreeMap<String, f64>,
) -> Vec<(usize, f64)> {
    defaults
        .iter()
        .filter_map(|&(name, def)| {
            let id = vars.lookup(name)?.0 as usize;
            Some((id, params.get(name).copied().unwrap_or(def)))
        })
        .collect()
}

/// Linear resampling of a waveform at `s` in [0, 1].
fn resample(wave: &[f32], s: f64) -> f64 {
    let last = wave.len().saturating_sub(1);
    let pos = s.clamp(0.0, 1.0) * last as f64;
    let i = (pos.floor() as usize).min(last);
    let frac = pos - i as f64;
    let at = |k: usize| f64::from(wave.get(k).copied().unwrap_or(0.0));
    at(i) + (at((i + 1).min(last)) - at(i)) * frac
}

/// One enabled custom wave's persistent state.
pub(crate) struct WaveState {
    index: usize,
    name: [String; 3],
    vars: Vec<f64>,
    resets: Vec<(usize, f64)>,
    /// The wave's `samples` variable, if its code uses it. `samples` is a
    /// wave built-in (MilkDrop lets wave frame code change the point
    /// count), not a user variable: it is reset to the definition's count
    /// each tick and read back, rounded and clamped to 1..=512, after the
    /// frame code. It is not in `BUILTIN_VARS`, so it is looked up by name
    /// once, here.
    samples_var: Option<usize>,
    /// Built-ins the point code writes (restored before each point).
    restore: Vec<usize>,
    frame_vals: Vec<f64>,
    regs: bool,
    warned: [bool; 3],
}

impl WaveState {
    pub(crate) fn new(index: usize, w: &CompiledWave) -> Self {
        let mut resets = resets(&w.vars, WAVE_DEFAULTS, &w.params);
        let samples_var = w.vars.lookup("samples").map(|v| v.0 as usize);
        if let Some(id) = samples_var {
            resets.push((id, f64::from(w.samples)));
        }
        let regs = [&w.init, &w.frame, &w.point]
            .iter()
            .any(|p| uses_regs(p.as_ref()));
        WaveState {
            index,
            name: ["init", "frame", "point"].map(|b| format!("wave[{index}].{b}")),
            vars: vec![0.0; w.vars.len().max(BUILTIN_LEN)],
            resets,
            samples_var,
            restore: w.point.as_ref().map(restored_writes).unwrap_or_default(),
            frame_vals: vec![0.0; BUILTIN_LEN],
            regs,
            warned: [false; 3],
        }
    }

    /// Index into `CompiledPreset::waves`.
    pub(crate) fn index(&self) -> usize {
        self.index
    }

    /// Runs the `init` code once (on the instance's first tick), with the
    /// read-only inputs written, the writables reset, `q` given and the
    /// global regs and `gmegabuf`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn init(
        &mut self,
        w: &CompiledWave,
        input: &TickInput,
        mesh: [u32; 2],
        q: &[f64; 32],
        megabuf: &mut Megabuf,
        g: &mut Globals,
        rng: &mut Rng,
        warnings: &mut Vec<String>,
    ) {
        let vars = &mut self.vars[..];
        write_inputs(vars, input, mesh);
        for &(id, v) in &self.resets {
            set(vars, id, v);
        }
        copy_q_in(vars, q);
        if self.regs {
            regs_in(vars, &g.regs);
        }
        let capped = run(w.init.as_ref(), vars, megabuf, &mut g.gmegabuf, rng);
        note_cap(capped, &mut self.warned[0], &self.name[0], warnings);
        if self.regs {
            regs_out(vars, &mut g.regs);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tick(
        &mut self,
        w: &CompiledWave,
        input: &TickInput,
        mesh: [u32; 2],
        q: &[f64; 32],
        megabuf: &mut Megabuf,
        g: &mut Globals,
        rng: &mut Rng,
        draw: &mut WaveDraw,
        warnings: &mut Vec<String>,
    ) {
        let vars = &mut self.vars[..];
        write_inputs(vars, input, mesh);
        for &(id, v) in &self.resets {
            set(vars, id, v);
        }
        copy_q_in(vars, q);
        if self.regs {
            regs_in(vars, &g.regs);
        }
        let capped = run(w.frame.as_ref(), vars, megabuf, &mut g.gmegabuf, rng);
        note_cap(capped, &mut self.warned[1], &self.name[1], warnings);

        let samples = match self.samples_var.map(|id| get(vars, id)) {
            Some(s) if s.is_finite() => s.round().clamp(1.0, f64::from(MAX_SAMPLES)) as u32,
            _ => w.samples.clamp(1, MAX_SAMPLES),
        };
        self.frame_vals.copy_from_slice(&vars[..BUILTIN_LEN]);
        let fv = &self.frame_vals;
        draw.dots = flag(fv[USEDOTS]);
        draw.thick = flag(fv[THICK]);
        draw.additive = flag(fv[ADDITIVE]);
        draw.points.clear();
        draw.colors.clear();
        let mut capped = false;
        for i in 0..samples {
            for &id in &self.restore {
                vars[id] = fv[id];
            }
            let s = if samples > 1 {
                f64::from(i) / f64::from(samples - 1)
            } else {
                0.0
            };
            vars[SAMPLE] = s;
            vars[VALUE1] = resample(&input.audio.wave[0], s);
            vars[VALUE2] = resample(&input.audio.wave[1], s);
            capped |= run(w.point.as_ref(), vars, megabuf, &mut g.gmegabuf, rng);
            draw.points.push([gpu(vars[X]), gpu(vars[Y])]);
            draw.colors
                .push([unit(vars[R]), unit(vars[G]), unit(vars[B]), unit(vars[A])]);
        }
        note_cap(capped, &mut self.warned[2], &self.name[2], warnings);
        if self.regs {
            regs_out(vars, &mut g.regs);
        }
    }
}

/// One enabled custom shape's persistent state.
pub(crate) struct ShapeState {
    index: usize,
    name: [String; 2],
    vars: Vec<f64>,
    resets: Vec<(usize, f64)>,
    regs: bool,
    warned: [bool; 2],
}

impl ShapeState {
    pub(crate) fn new(index: usize, s: &CompiledShape) -> Self {
        let mut resets = resets(&s.vars, SHAPE_DEFAULTS, &s.params);
        resets.push((SIDES, f64::from(s.sides)));
        let regs = [&s.init, &s.frame].iter().any(|p| uses_regs(p.as_ref()));
        ShapeState {
            index,
            name: ["init", "frame"].map(|b| format!("shape[{index}].{b}")),
            vars: vec![0.0; s.vars.len().max(BUILTIN_LEN)],
            resets,
            regs,
            warned: [false; 2],
        }
    }

    /// Index into `CompiledPreset::shapes`.
    pub(crate) fn index(&self) -> usize {
        self.index
    }

    /// Runs the `init` code once (on the instance's first tick), with the
    /// read-only inputs written, the writables reset, `q` given and the
    /// global regs and `gmegabuf`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn init(
        &mut self,
        s: &CompiledShape,
        input: &TickInput,
        mesh: [u32; 2],
        q: &[f64; 32],
        megabuf: &mut Megabuf,
        g: &mut Globals,
        rng: &mut Rng,
        warnings: &mut Vec<String>,
    ) {
        let vars = &mut self.vars[..];
        write_inputs(vars, input, mesh);
        for &(id, v) in &self.resets {
            set(vars, id, v);
        }
        copy_q_in(vars, q);
        if self.regs {
            regs_in(vars, &g.regs);
        }
        let capped = run(s.init.as_ref(), vars, megabuf, &mut g.gmegabuf, rng);
        note_cap(capped, &mut self.warned[0], &self.name[0], warnings);
        if self.regs {
            regs_out(vars, &mut g.regs);
        }
    }

    /// Runs the frame code once per instance, appending one draw each.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tick(
        &mut self,
        s: &CompiledShape,
        input: &TickInput,
        mesh: [u32; 2],
        q: &[f64; 32],
        megabuf: &mut Megabuf,
        g: &mut Globals,
        rng: &mut Rng,
        draws: &mut Vec<ShapeDraw>,
        warnings: &mut Vec<String>,
    ) {
        let vars = &mut self.vars[..];
        write_inputs(vars, input, mesh);
        let mut capped = false;
        for instance in 0..s.instances {
            for &(id, v) in &self.resets {
                set(vars, id, v);
            }
            copy_q_in(vars, q);
            vars[INSTANCE] = f64::from(instance);
            if self.regs {
                regs_in(vars, &g.regs);
            }
            capped |= run(s.frame.as_ref(), vars, megabuf, &mut g.gmegabuf, rng);
            if self.regs {
                regs_out(vars, &mut g.regs);
            }
            let v = |id: usize| vars[id];
            let sides = match v(SIDES) {
                n if n.is_finite() => n.round().clamp(3.0, 100.0) as u32,
                _ => s.sides,
            };
            draws.push(ShapeDraw {
                x: gpu(v(X)),
                y: gpu(v(Y)),
                rad: gpu(v(RAD)),
                ang: gpu(v(ANG)),
                sides,
                inner: [unit(v(R)), unit(v(G)), unit(v(B)), unit(v(A))],
                outer: [unit(v(R2)), unit(v(G2)), unit(v(B2)), unit(v(A2))],
                border: [
                    unit(v(BORDER_R)),
                    unit(v(BORDER_G)),
                    unit(v(BORDER_B)),
                    unit(v(BORDER_A)),
                ],
                additive: flag(v(ADDITIVE)),
                thick_outline: flag(v(THICKOUTLINE)),
                textured: flag(v(TEXTURED)),
                tex_zoom: gpu(v(TEX_ZOOM)),
                tex_ang: gpu(v(TEX_ANG)),
            });
        }
        note_cap(capped, &mut self.warned[1], &self.name[1], warnings);
    }
}
