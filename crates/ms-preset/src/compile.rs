//! Compiling presets into programs with a stable built-in variable layout.

use crate::format::{Preset, ShapeDef, WaveDef};
use ms_expr::{CompileError, Program, VarTable};
use std::collections::BTreeMap;

/// Built-in variable names, registered first and in this order in every
/// program table, so their `VarId`s are identical across all presets. Any
/// `VarId >= BUILTIN_VARS.len()` is a user variable.
pub const BUILTIN_VARS: &[&str] = &[
    "time",
    "fps",
    "frame",
    "progress",
    "bass",
    "mid",
    "treb",
    "bass_att",
    "mid_att",
    "treb_att",
    "beat",
    "beat_strength",
    "meshx",
    "meshy",
    "aspectx",
    "aspecty",
    "pixelsx",
    "pixelsy",
    "x",
    "y",
    "rad",
    "ang",
    "zoom",
    "zoomexp",
    "rot",
    "warp",
    "cx",
    "cy",
    "dx",
    "dy",
    "sx",
    "sy",
    "decay",
    "wave_mode",
    "wave_r",
    "wave_g",
    "wave_b",
    "wave_a",
    "wave_x",
    "wave_y",
    "wave_mystery",
    "wave_usedots",
    "wave_thick",
    "wave_additive",
    "wave_brighten",
    "ob_size",
    "ob_r",
    "ob_g",
    "ob_b",
    "ob_a",
    "ib_size",
    "ib_r",
    "ib_g",
    "ib_b",
    "ib_a",
    "mv_x",
    "mv_y",
    "mv_dx",
    "mv_dy",
    "mv_l",
    "mv_r",
    "mv_g",
    "mv_b",
    "mv_a",
    "echo_zoom",
    "echo_alpha",
    "echo_orient",
    "gamma",
    "invert",
    "solarize",
    "brighten",
    "darken",
    "darken_center",
    "texture_wrap",
    "b1n",
    "b2n",
    "b3n",
    "b1x",
    "b2x",
    "b3x",
    "b1ed",
    "sample",
    "value1",
    "value2",
    "instance",
    "r",
    "g",
    "b",
    "a",
    "r2",
    "g2",
    "b2",
    "a2",
    "border_r",
    "border_g",
    "border_b",
    "border_a",
    "additive",
    "thickoutline",
    "textured",
    "tex_zoom",
    "tex_ang",
    "sides",
    "usedots",
    "thick",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
    "q8",
    "q9",
    "q10",
    "q11",
    "q12",
    "q13",
    "q14",
    "q15",
    "q16",
    "q17",
    "q18",
    "q19",
    "q20",
    "q21",
    "q22",
    "q23",
    "q24",
    "q25",
    "q26",
    "q27",
    "q28",
    "q29",
    "q30",
    "q31",
    "q32",
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t7",
    "t8",
    "reg00",
    "reg01",
    "reg02",
    "reg03",
    "reg04",
    "reg05",
    "reg06",
    "reg07",
    "reg08",
    "reg09",
    "reg10",
    "reg11",
    "reg12",
    "reg13",
    "reg14",
    "reg15",
    "reg16",
    "reg17",
    "reg18",
    "reg19",
    "reg20",
    "reg21",
    "reg22",
    "reg23",
    "reg24",
    "reg25",
    "reg26",
    "reg27",
    "reg28",
    "reg29",
    "reg30",
    "reg31",
    "reg32",
    "reg33",
    "reg34",
    "reg35",
    "reg36",
    "reg37",
    "reg38",
    "reg39",
    "reg40",
    "reg41",
    "reg42",
    "reg43",
    "reg44",
    "reg45",
    "reg46",
    "reg47",
    "reg48",
    "reg49",
    "reg50",
    "reg51",
    "reg52",
    "reg53",
    "reg54",
    "reg55",
    "reg56",
    "reg57",
    "reg58",
    "reg59",
    "reg60",
    "reg61",
    "reg62",
    "reg63",
    "reg64",
    "reg65",
    "reg66",
    "reg67",
    "reg68",
    "reg69",
    "reg70",
    "reg71",
    "reg72",
    "reg73",
    "reg74",
    "reg75",
    "reg76",
    "reg77",
    "reg78",
    "reg79",
    "reg80",
    "reg81",
    "reg82",
    "reg83",
    "reg84",
    "reg85",
    "reg86",
    "reg87",
    "reg88",
    "reg89",
    "reg90",
    "reg91",
    "reg92",
    "reg93",
    "reg94",
    "reg95",
    "reg96",
    "reg97",
    "reg98",
    "reg99",
];

/// A fresh table with exactly [`BUILTIN_VARS`] registered.
pub fn builtin_var_table() -> VarTable {
    let mut t = VarTable::new();
    for n in BUILTIN_VARS {
        t.get_or_insert(n);
    }
    t
}

/// A code block that failed to compile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeError {
    /// e.g. "frame", "wave[1].point", "shape[0].init".
    pub block: String,
    pub error: CompileError,
}

#[derive(Debug, Clone)]
pub struct CompiledWave {
    pub enabled: bool,
    pub samples: u32,
    pub params: BTreeMap<String, f64>,
    pub vars: VarTable,
    pub init: Option<Program>,
    pub frame: Option<Program>,
    pub point: Option<Program>,
}

#[derive(Debug, Clone)]
pub struct CompiledShape {
    pub enabled: bool,
    pub sides: u32,
    pub instances: u32,
    pub params: BTreeMap<String, f64>,
    pub vars: VarTable,
    pub init: Option<Program>,
    pub frame: Option<Program>,
}

#[derive(Debug, Clone)]
pub struct CompiledPreset {
    pub source: Preset,
    pub vars: VarTable,
    pub init: Option<Program>,
    pub frame: Option<Program>,
    pub vertex: Option<Program>,
    pub waves: Vec<CompiledWave>,
    pub shapes: Vec<CompiledShape>,
    pub errors: Vec<CodeError>,
}

/// Compile one block. Whitespace/comment-only source yields `None` with no
/// error; a failing block yields `None` and is recorded in `errors`.
fn block(
    name: String,
    src: &str,
    vars: &mut VarTable,
    errors: &mut Vec<CodeError>,
) -> Option<Program> {
    if let Ok(toks) = ms_expr::lex(src) {
        let empty = toks
            .iter()
            .all(|t| matches!(t.tok, ms_expr::Tok::Eof | ms_expr::Tok::Semi));
        if empty {
            return None;
        }
    }
    match ms_expr::compile(src, vars) {
        Ok(p) => Some(p),
        Err(error) => {
            errors.push(CodeError { block: name, error });
            None
        }
    }
}

fn compile_wave(i: usize, w: &WaveDef, errors: &mut Vec<CodeError>) -> CompiledWave {
    let mut vars = builtin_var_table();
    let init = block(format!("wave[{i}].init"), &w.init, &mut vars, errors);
    let frame = block(format!("wave[{i}].frame"), &w.frame, &mut vars, errors);
    let point = block(format!("wave[{i}].point"), &w.point, &mut vars, errors);
    CompiledWave {
        enabled: w.enabled,
        samples: w.samples,
        params: w.params.clone(),
        vars,
        init,
        frame,
        point,
    }
}

fn compile_shape(i: usize, s: &ShapeDef, errors: &mut Vec<CodeError>) -> CompiledShape {
    let mut vars = builtin_var_table();
    let init = block(format!("shape[{i}].init"), &s.init, &mut vars, errors);
    let frame = block(format!("shape[{i}].frame"), &s.frame, &mut vars, errors);
    CompiledShape {
        enabled: s.enabled,
        sides: s.sides,
        instances: s.instances,
        params: s.params.clone(),
        vars,
        init,
        frame,
    }
}

/// Compile every block of `p`. Blocks with errors become `None` and are
/// listed in `errors`; the preset itself still compiles. Run on a thread
/// with at least 16 MB of stack (deeply nested expressions).
pub fn compile_preset(p: Preset) -> CompiledPreset {
    let mut errors = Vec::new();
    let mut vars = builtin_var_table();
    let init = block("init".into(), &p.code.init, &mut vars, &mut errors);
    let frame = block("frame".into(), &p.code.frame, &mut vars, &mut errors);
    let vertex = block("vertex".into(), &p.code.vertex, &mut vars, &mut errors);
    let waves = p
        .waves
        .iter()
        .enumerate()
        .map(|(i, w)| compile_wave(i, w, &mut errors))
        .collect();
    let shapes = p
        .shapes
        .iter()
        .enumerate()
        .map(|(i, s)| compile_shape(i, s, &mut errors))
        .collect();
    CompiledPreset { source: p, vars, init, frame, vertex, waves, shapes, errors }
}
