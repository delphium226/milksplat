//! What a preset's code runs against: global state, per-tick inputs, the
//! fixed ids of the built-in variables, and helpers shared by the main,
//! wave and shape programs.
//!
//! Every program table is seeded with [`BUILTIN_VARS`] in order, so a
//! built-in's index is the same in every table. The constants below are
//! those indices, resolved at compile time (a misspelt name fails the
//! build), so nothing looks a variable up by name while ticking.

use ms_analysis::AudioFrame;
use ms_expr::{ExecEnv, Megabuf, Op, Program, Rng};
use ms_preset::BUILTIN_VARS;

/// State shared by every preset instance, surviving preset changes (§5.2).
#[derive(Debug, Clone)]
pub struct Globals {
    /// `reg00`..`reg99`.
    pub regs: [f64; 100],
    pub gmegabuf: Megabuf,
}

impl Default for Globals {
    fn default() -> Self {
        Globals {
            regs: [0.0; 100],
            gmegabuf: Megabuf::new(),
        }
    }
}

/// The read-only inputs of one tick.
#[derive(Debug, Clone, Copy)]
pub struct TickInput<'a> {
    /// Seconds since the preset started.
    pub time: f64,
    pub fps: f64,
    pub frame: u64,
    /// 0 to 1 over the preset's display time.
    pub progress: f64,
    pub audio: &'a AudioFrame,
    /// Multipliers applied to `x - 0.5` and `y - 0.5` before the vertex
    /// `rad`/`ang` are computed; also `aspectx`/`aspecty`.
    pub aspect: [f64; 2],
    /// Internal (feedback) resolution, `pixelsx`/`pixelsy` and `texsize`.
    pub pixels: [u32; 2],
}

const fn same(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Index of a built-in variable; a name not in [`BUILTIN_VARS`] is a
/// compile-time error.
const fn builtin(name: &str) -> usize {
    let mut i = 0;
    while i < BUILTIN_VARS.len() {
        if same(BUILTIN_VARS[i].as_bytes(), name.as_bytes()) {
            return i;
        }
        i += 1;
    }
    panic!("not a built-in variable")
}

/// Number of built-in variables; any index at or above it is a user variable.
pub(crate) const BUILTIN_LEN: usize = BUILTIN_VARS.len();

macro_rules! ids {
    ($($c:ident = $n:literal),* $(,)?) => {
        $(pub(crate) const $c: usize = builtin($n);)*
    };
}

ids! {
    TIME = "time", FPS = "fps", FRAME = "frame", PROGRESS = "progress",
    BASS = "bass", MID = "mid", TREB = "treb",
    BASS_ATT = "bass_att", MID_ATT = "mid_att", TREB_ATT = "treb_att",
    BEAT = "beat", BEAT_STRENGTH = "beat_strength",
    MESHX = "meshx", MESHY = "meshy", ASPECTX = "aspectx", ASPECTY = "aspecty",
    PIXELSX = "pixelsx", PIXELSY = "pixelsy",
    X = "x", Y = "y", RAD = "rad", ANG = "ang",
    ZOOM = "zoom", ZOOMEXP = "zoomexp", ROT = "rot", WARP = "warp",
    CX = "cx", CY = "cy", DX = "dx", DY = "dy", SX = "sx", SY = "sy",
    DECAY = "decay",
    WAVE_MODE = "wave_mode", WAVE_R = "wave_r", WAVE_G = "wave_g", WAVE_B = "wave_b",
    WAVE_A = "wave_a", WAVE_X = "wave_x", WAVE_Y = "wave_y", WAVE_MYSTERY = "wave_mystery",
    WAVE_USEDOTS = "wave_usedots", WAVE_THICK = "wave_thick",
    WAVE_ADDITIVE = "wave_additive", WAVE_BRIGHTEN = "wave_brighten",
    OB_SIZE = "ob_size", OB_R = "ob_r", OB_G = "ob_g", OB_B = "ob_b", OB_A = "ob_a",
    IB_SIZE = "ib_size", IB_R = "ib_r", IB_G = "ib_g", IB_B = "ib_b", IB_A = "ib_a",
    MV_X = "mv_x", MV_Y = "mv_y", MV_DX = "mv_dx", MV_DY = "mv_dy", MV_L = "mv_l",
    MV_R = "mv_r", MV_G = "mv_g", MV_B = "mv_b", MV_A = "mv_a",
    ECHO_ZOOM = "echo_zoom", ECHO_ALPHA = "echo_alpha", ECHO_ORIENT = "echo_orient",
    GAMMA = "gamma", INVERT = "invert", SOLARIZE = "solarize", BRIGHTEN = "brighten",
    DARKEN = "darken", DARKEN_CENTER = "darken_center",
    B1N = "b1n", B2N = "b2n", B3N = "b3n", B1X = "b1x", B2X = "b2x", B3X = "b3x",
    SAMPLE = "sample", VALUE1 = "value1", VALUE2 = "value2", INSTANCE = "instance",
    R = "r", G = "g", B = "b", A = "a", R2 = "r2", G2 = "g2", B2 = "b2", A2 = "a2",
    BORDER_R = "border_r", BORDER_G = "border_g", BORDER_B = "border_b", BORDER_A = "border_a",
    ADDITIVE = "additive", THICKOUTLINE = "thickoutline", TEXTURED = "textured",
    TEX_ZOOM = "tex_zoom", TEX_ANG = "tex_ang", SIDES = "sides",
    USEDOTS = "usedots", THICK = "thick",
    Q1 = "q1", T1 = "t1", REG00 = "reg00",
}

// q1..q32, t1..t8 and reg00..reg99 are contiguous runs.
const _: () = assert!(builtin("q32") == Q1 + 31);
const _: () = assert!(builtin("t8") == T1 + 7);
const _: () = assert!(builtin("reg99") == REG00 + 99);

/// Variables that keep their value from one vertex (or wave point) to the
/// next instead of being restored to the post-frame value: user variables,
/// `reg00`..`reg99` (global) and `t1`..`t8` (per wave/shape).
pub(crate) fn is_carried(id: usize) -> bool {
    id >= BUILTIN_LEN || (REG00..REG00 + 100).contains(&id) || (T1..T1 + 8).contains(&id)
}

/// Built-in variables `p` writes that are not carried: restoring just these
/// to their post-frame values gives each point a fresh post-frame start.
pub(crate) fn restored_writes(p: &Program) -> Vec<usize> {
    let mut ids: Vec<usize> = p
        .ops()
        .iter()
        .filter_map(|op| match *op {
            Op::StoreVar { var, .. } => Some(var.0 as usize),
            _ => None,
        })
        .filter(|&id| !is_carried(id))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Whether `p` reads or writes any of `reg00`..`reg99`.
pub(crate) fn uses_regs(p: Option<&Program>) -> bool {
    p.is_some_and(|p| {
        p.ops().iter().any(|op| match *op {
            Op::LoadVar { var, .. } | Op::StoreVar { var, .. } => {
                (REG00..REG00 + 100).contains(&(var.0 as usize))
            }
            _ => false,
        })
    })
}

#[inline]
pub(crate) fn get(vars: &[f64], id: usize) -> f64 {
    vars.get(id).copied().unwrap_or(0.0)
}

#[inline]
pub(crate) fn set(vars: &mut [f64], id: usize, v: f64) {
    if let Some(slot) = vars.get_mut(id) {
        *slot = v;
    }
}

/// Writes the read-only inputs shared by every program table (§5.3).
pub(crate) fn write_inputs(vars: &mut [f64], input: &TickInput, mesh: [u32; 2]) {
    let a = input.audio;
    for (id, v) in [
        (TIME, input.time),
        (FPS, input.fps),
        (FRAME, input.frame as f64),
        (PROGRESS, input.progress),
        (BASS, f64::from(a.bass)),
        (MID, f64::from(a.mid)),
        (TREB, f64::from(a.treb)),
        (BASS_ATT, f64::from(a.bass_att)),
        (MID_ATT, f64::from(a.mid_att)),
        (TREB_ATT, f64::from(a.treb_att)),
        (BEAT, f64::from(a.beat)),
        (BEAT_STRENGTH, f64::from(a.beat_strength)),
        (MESHX, f64::from(mesh[0])),
        (MESHY, f64::from(mesh[1])),
        (ASPECTX, input.aspect[0]),
        (ASPECTY, input.aspect[1]),
        (PIXELSX, f64::from(input.pixels[0])),
        (PIXELSY, f64::from(input.pixels[1])),
    ] {
        set(vars, id, v);
    }
}

pub(crate) fn copy_q_in(vars: &mut [f64], q: &[f64; 32]) {
    if let Some(dst) = vars.get_mut(Q1..Q1 + 32) {
        dst.copy_from_slice(q);
    }
}

pub(crate) fn regs_in(vars: &mut [f64], regs: &[f64; 100]) {
    if let Some(dst) = vars.get_mut(REG00..REG00 + 100) {
        dst.copy_from_slice(regs);
    }
}

pub(crate) fn regs_out(vars: &[f64], regs: &mut [f64; 100]) {
    if let Some(src) = vars.get(REG00..REG00 + 100) {
        regs.copy_from_slice(src);
    }
}

/// Runs `p` (if the block compiled) on `vars`; returns true if it hit the
/// loop cap.
pub(crate) fn run(
    p: Option<&Program>,
    vars: &mut [f64],
    megabuf: &mut Megabuf,
    gmegabuf: &mut Megabuf,
    rng: &mut Rng,
) -> bool {
    match p {
        Some(p) => {
            let mut env = ExecEnv {
                vars,
                megabuf,
                gmegabuf,
                rng,
            };
            p.run(&mut env).loop_cap_hit
        }
        None => false,
    }
}

/// Records a loop-cap hit in `block`, once per instance lifetime (§5.1).
pub(crate) fn note_cap(capped: bool, warned: &mut bool, block: &str, out: &mut Vec<String>) {
    if capped && !*warned {
        *warned = true;
        let msg = format!(
            "{block}: loop iteration cap ({}) hit; loop stopped",
            ms_expr::LOOP_CAP
        );
        tracing::warn!("{msg}");
        out.push(msg);
    }
}

/// f64 to f32 for the GPU: NaN and ±Inf (including overflow) become 0.
#[inline]
pub(crate) fn gpu(v: f64) -> f32 {
    let f = v as f32;
    if f.is_finite() {
        f
    } else {
        0.0
    }
}

/// A colour channel: [`gpu`], clamped to [0, 1].
#[inline]
pub(crate) fn unit(v: f64) -> f32 {
    gpu(v).clamp(0.0, 1.0)
}

/// A flag variable: true when non-zero (NaN is false).
#[inline]
pub(crate) fn flag(v: f64) -> bool {
    v != 0.0 && !v.is_nan()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_match_table() {
        let t = ms_preset::builtin_var_table();
        for (id, name) in [
            (ZOOM, "zoom"),
            (Q1, "q1"),
            (T1, "t1"),
            (REG00, "reg00"),
            (THICK, "thick"),
        ] {
            assert_eq!(t.lookup(name).map(|v| v.0 as usize), Some(id), "{name}");
        }
    }

    #[test]
    fn gpu_sanitises() {
        assert_eq!(gpu(f64::NAN), 0.0);
        assert_eq!(gpu(f64::INFINITY), 0.0);
        assert_eq!(gpu(1e300), 0.0);
        assert_eq!(gpu(-2.5), -2.5);
        assert_eq!(unit(7.0), 1.0);
        assert_eq!(unit(f64::NAN), 0.0);
    }
}
