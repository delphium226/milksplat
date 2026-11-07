//! `.splat` preset file format (TOML): parsing, validation and serialisation.
//!
//! Preset files are untrusted user data. Nothing here panics: malformed input
//! becomes a [`PresetError`], and recoverable problems become [`Warning`]s.

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use toml::{Table, Value};

/// Highest file format version this build understands.
pub const FORMAT_VERSION: u32 = 1;
/// Maximum number of `[[wave]]` and of `[[shape]]` entries.
pub const MAX_CUSTOM: usize = 4;

/// Every writable main-preset variable (spec section 5.3) with its default.
pub const PARAM_DEFAULTS: &[(&str, f64)] = &[
    // motion
    ("zoom", 1.0),
    ("zoomexp", 1.0),
    ("rot", 0.0),
    ("warp", 1.0),
    ("cx", 0.5),
    ("cy", 0.5),
    ("dx", 0.0),
    ("dy", 0.0),
    ("sx", 1.0),
    ("sy", 1.0),
    ("decay", 0.98),
    // main wave
    ("wave_mode", 0.0),
    ("wave_r", 1.0),
    ("wave_g", 1.0),
    ("wave_b", 1.0),
    ("wave_a", 0.8),
    ("wave_x", 0.5),
    ("wave_y", 0.5),
    ("wave_mystery", 0.0),
    ("wave_usedots", 0.0),
    ("wave_thick", 0.0),
    ("wave_additive", 0.0),
    ("wave_brighten", 1.0),
    // borders
    ("ob_size", 0.01),
    ("ob_r", 0.0),
    ("ob_g", 0.0),
    ("ob_b", 0.0),
    ("ob_a", 0.0),
    ("ib_size", 0.01),
    ("ib_r", 0.25),
    ("ib_g", 0.25),
    ("ib_b", 0.25),
    ("ib_a", 0.0),
    // motion vectors
    ("mv_x", 12.0),
    ("mv_y", 9.0),
    ("mv_dx", 0.0),
    ("mv_dy", 0.0),
    ("mv_l", 0.9),
    ("mv_r", 1.0),
    ("mv_g", 1.0),
    ("mv_b", 1.0),
    ("mv_a", 0.0),
    // composite effects
    ("echo_zoom", 1.0),
    ("echo_alpha", 0.0),
    ("echo_orient", 0.0),
    ("gamma", 1.0),
    ("invert", 0.0),
    ("solarize", 0.0),
    ("brighten", 0.0),
    ("darken", 0.0),
    ("darken_center", 0.0),
    // sampling and blur
    ("texture_wrap", 1.0),
    ("b1n", 0.0),
    ("b2n", 0.0),
    ("b3n", 0.0),
    ("b1x", 1.0),
    ("b2x", 1.0),
    ("b3x", 1.0),
    ("b1ed", 0.25),
];

fn param_default(name: &str) -> Option<f64> {
    PARAM_DEFAULTS
        .iter()
        .find(|(k, _)| *k == name)
        .map(|&(_, v)| v)
}

/// A non-fatal problem found while parsing (unknown key, clamped value, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Warning {
    pub message: String,
}

/// A fatal parse error with a 1-based position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresetError {
    pub line: u32,
    pub col: u32,
    pub message: String,
}

impl fmt::Display for PresetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for PresetError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeedbackFormat {
    #[default]
    Rgba8,
    Fp16,
}

impl FeedbackFormat {
    fn as_str(self) -> &'static str {
        match self {
            FeedbackFormat::Rgba8 => "rgba8",
            FeedbackFormat::Fp16 => "fp16",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Meta {
    pub name: String,
    pub author: String,
    pub license: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    pub feedback_format: FeedbackFormat,
    pub mesh: [u32; 2],
    pub sim_rate: f64,
    /// Only the writable variables the file set (lowercased names).
    pub values: BTreeMap<String, f64>,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            feedback_format: FeedbackFormat::Rgba8,
            mesh: [64, 48],
            sim_rate: 60.0,
            values: BTreeMap::new(),
        }
    }
}

impl Params {
    /// The file's value for `name`, else the MilkDrop default (0.0 if unknown).
    pub fn param(&self, name: &str) -> f64 {
        self.values
            .get(name)
            .copied()
            .or_else(|| param_default(name))
            .unwrap_or(0.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Code {
    pub init: String,
    pub frame: String,
    pub vertex: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WaveDef {
    pub enabled: bool,
    pub samples: u32,
    pub params: BTreeMap<String, f64>,
    pub init: String,
    pub frame: String,
    pub point: String,
}

impl Default for WaveDef {
    fn default() -> Self {
        WaveDef {
            enabled: true,
            samples: 512,
            params: BTreeMap::new(),
            init: String::new(),
            frame: String::new(),
            point: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeDef {
    pub enabled: bool,
    pub sides: u32,
    pub instances: u32,
    pub params: BTreeMap<String, f64>,
    pub init: String,
    pub frame: String,
}

impl Default for ShapeDef {
    fn default() -> Self {
        ShapeDef {
            enabled: true,
            sides: 4,
            instances: 1,
            params: BTreeMap::new(),
            init: String::new(),
            frame: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ShaderDefs {
    pub warp: Option<String>,
    pub comp: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Preset {
    pub version: u32,
    pub meta: Meta,
    pub params: Params,
    pub code: Code,
    pub waves: Vec<WaveDef>,
    pub shapes: Vec<ShapeDef>,
    pub shader: ShaderDefs,
}

impl Default for Preset {
    fn default() -> Self {
        Preset {
            version: FORMAT_VERSION,
            meta: Meta::default(),
            params: Params::default(),
            code: Code::default(),
            waves: Vec::new(),
            shapes: Vec::new(),
            shader: ShaderDefs::default(),
        }
    }
}

// ---------------------------------------------------------------- parsing

fn err(line: u32, col: u32, message: impl Into<String>) -> PresetError {
    PresetError {
        line,
        col,
        message: message.into(),
    }
}

fn clamp_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// 1-based line/column (in chars) of byte offset `off` in `text`.
fn line_col(text: &str, off: usize) -> (u32, u32) {
    let mut off = off.min(text.len());
    while !text.is_char_boundary(off) {
        off -= 1;
    }
    let before = &text[..off];
    let line = before.matches('\n').count();
    let col = match before.rfind('\n') {
        Some(i) => before[i + 1..].chars().count(),
        None => before.chars().count(),
    };
    (clamp_u32(line + 1), clamp_u32(col + 1))
}

/// Parses a `.splat` file. A UTF-8 BOM is stripped; CRLF is accepted.
pub fn parse(bytes: &[u8]) -> Result<(Preset, Vec<Warning>), PresetError> {
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF][..]).unwrap_or(bytes);
    let text = match std::str::from_utf8(bytes) {
        Ok(t) => t,
        Err(e) => {
            // The prefix is valid UTF-8, so columns can count chars.
            let at = e.valid_up_to();
            let (line, col) = std::str::from_utf8(&bytes[..at]).map_or((1, 1), |p| line_col(p, at));
            return Err(err(line, col, "file is not valid UTF-8"));
        }
    };
    let table: Table = match text.parse() {
        Ok(t) => t,
        Err(e) => {
            let (line, col) = e.span().map_or((1, 1), |s| line_col(text, s.start));
            return Err(err(line, col, e.message().to_string()));
        }
    };
    let mut w = Warnings::default();
    let p = build(&table, &mut w)?;
    Ok((p, w.0))
}

#[derive(Default)]
struct Warnings(Vec<Warning>);

impl Warnings {
    fn add(&mut self, message: String) {
        self.0.push(Warning { message });
    }
    fn unknown(&mut self, path: &str) {
        self.add(format!("unknown key `{path}` ignored"));
    }
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Float(f) if f.is_finite() => Some(*f),
        Value::Integer(i) => Some(*i as f64),
        Value::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn type_name(v: &Value) -> &'static str {
    v.type_str()
}

fn text_field(v: &Value, path: &str, w: &mut Warnings) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        other => {
            w.add(format!(
                "`{path}` must be a string, found {}; ignored",
                type_name(other)
            ));
            None
        }
    }
}

fn bool_field(v: &Value, path: &str, w: &mut Warnings) -> Option<bool> {
    match v {
        Value::Boolean(b) => Some(*b),
        other => {
            w.add(format!(
                "`{path}` must be true or false, found {}; ignored",
                type_name(other)
            ));
            None
        }
    }
}

/// An integer field clamped into `lo..=hi`; `default` if it is not an integer.
fn uint_field(v: &Value, path: &str, (lo, hi): (u32, u32), default: u32, w: &mut Warnings) -> u32 {
    match v {
        Value::Integer(i) => {
            let c = (*i).clamp(i64::from(lo), i64::from(hi));
            let c = u32::try_from(c).unwrap_or(default);
            if i64::from(c) != *i {
                w.add(format!("`{path}` = {i} is out of range; clamped to {c}"));
            }
            c
        }
        other => {
            w.add(format!(
                "`{path}` must be an integer, found {}; using {default}",
                type_name(other)
            ));
            default
        }
    }
}

fn build(root: &Table, w: &mut Warnings) -> Result<Preset, PresetError> {
    let mut p = Preset::default();
    for (key, val) in root {
        match key.as_str() {
            "version" => p.version = read_version(val)?,
            "meta" => with_table(val, "meta", w, |t, w| read_meta(t, &mut p.meta, w)),
            "params" => with_table(val, "params", w, |t, w| read_params(t, &mut p.params, w)),
            "code" => with_table(val, "code", w, |t, w| read_code(t, &mut p.code, w)),
            "shader" => with_table(val, "shader", w, |t, w| read_shader(t, &mut p.shader, w)),
            "wave" => {
                p.waves = read_list(val, "wave", w, read_wave);
            }
            "shape" => {
                p.shapes = read_list(val, "shape", w, read_shape);
            }
            _ => w.unknown(key),
        }
    }
    Ok(p)
}

fn read_version(v: &Value) -> Result<u32, PresetError> {
    match v {
        Value::Integer(i) if *i > i64::from(FORMAT_VERSION) => Err(err(
            1,
            1,
            format!(
                "this preset (version {i}) was made by a newer Milksplat; \
                 this build reads up to version {FORMAT_VERSION}"
            ),
        )),
        Value::Integer(i) if *i >= 1 => Ok(u32::try_from(*i).unwrap_or(FORMAT_VERSION)),
        Value::Integer(i) => Err(err(1, 1, format!("invalid version {i}"))),
        other => Err(err(
            1,
            1,
            format!("version must be an integer, found {}", type_name(other)),
        )),
    }
}

fn with_table(v: &Value, path: &str, w: &mut Warnings, f: impl FnOnce(&Table, &mut Warnings)) {
    match v {
        Value::Table(t) => f(t, w),
        other => w.add(format!(
            "`{path}` must be a table, found {}; ignored",
            type_name(other)
        )),
    }
}

fn read_list<T>(
    v: &Value,
    name: &str,
    w: &mut Warnings,
    f: impl Fn(&Table, &str, &mut Warnings) -> T,
) -> Vec<T> {
    let Value::Array(items) = v else {
        w.add(format!(
            "`{name}` must be an array of tables ([[{name}]]), found {}; ignored",
            type_name(v)
        ));
        return Vec::new();
    };
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if i >= MAX_CUSTOM {
            w.add(format!(
                "`{name}[{i}]` dropped: at most {MAX_CUSTOM} {name} entries are supported"
            ));
            continue;
        }
        let path = format!("{name}[{i}]");
        match item {
            Value::Table(t) => out.push(f(t, &path, w)),
            other => w.add(format!(
                "`{path}` must be a table, found {}; ignored",
                type_name(other)
            )),
        }
    }
    out
}

fn read_meta(t: &Table, m: &mut Meta, w: &mut Warnings) {
    for (k, v) in t {
        let path = format!("meta.{k}");
        let slot = match k.as_str() {
            "name" => &mut m.name,
            "author" => &mut m.author,
            "license" => &mut m.license,
            _ => {
                w.unknown(&path);
                continue;
            }
        };
        if let Some(s) = text_field(v, &path, w) {
            *slot = s;
        }
    }
}

fn read_code(t: &Table, c: &mut Code, w: &mut Warnings) {
    for (k, v) in t {
        let path = format!("code.{k}");
        let slot = match k.as_str() {
            "init" => &mut c.init,
            "frame" => &mut c.frame,
            "vertex" => &mut c.vertex,
            _ => {
                w.unknown(&path);
                continue;
            }
        };
        if let Some(s) = text_field(v, &path, w) {
            *slot = s;
        }
    }
}

fn read_shader(t: &Table, s: &mut ShaderDefs, w: &mut Warnings) {
    for (k, v) in t {
        let path = format!("shader.{k}");
        let slot = match k.as_str() {
            "warp" => &mut s.warp,
            "comp" => &mut s.comp,
            _ => {
                w.unknown(&path);
                continue;
            }
        };
        if let Some(text) = text_field(v, &path, w) {
            *slot = Some(text);
        }
    }
}

fn read_params(t: &Table, p: &mut Params, w: &mut Warnings) {
    for (k, v) in t {
        let name = k.to_lowercase();
        let path = format!("params.{k}");
        match name.as_str() {
            "feedback_format" => match v {
                Value::String(s) if s.eq_ignore_ascii_case("rgba8") => {
                    p.feedback_format = FeedbackFormat::Rgba8;
                }
                Value::String(s) if s.eq_ignore_ascii_case("fp16") => {
                    p.feedback_format = FeedbackFormat::Fp16;
                }
                _ => w.add(format!(
                    "`{path}` must be \"rgba8\" or \"fp16\"; using rgba8"
                )),
            },
            "mesh" => read_mesh(v, &path, p, w),
            "sim_rate" => match num(v) {
                Some(r) => {
                    let c = r.clamp(1.0, 240.0);
                    if c != r {
                        w.add(format!("`{path}` = {r} is out of range; clamped to {c}"));
                    }
                    p.sim_rate = c;
                }
                None => w.add(format!("`{path}` must be a number; using 60")),
            },
            _ if param_default(&name).is_some() => match num(v) {
                Some(x) => {
                    p.values.insert(name, x);
                }
                None => w.add(format!(
                    "`{path}` must be a finite number or boolean, found {}; using the default",
                    type_name(v)
                )),
            },
            _ => w.unknown(&path),
        }
    }
}

fn read_mesh(v: &Value, path: &str, p: &mut Params, w: &mut Warnings) {
    let dims = match v {
        Value::Array(a) if a.len() == 2 => match (&a[0], &a[1]) {
            (Value::Integer(c), Value::Integer(r)) => Some((*c, *r)),
            _ => None,
        },
        _ => None,
    };
    let Some((c, r)) = dims else {
        w.add(format!(
            "`{path}` must be [columns, rows] integers; using [64, 48]"
        ));
        return;
    };
    let cc = c.clamp(8, 192);
    let rr = r.clamp(6, 144);
    if (cc, rr) != (c, r) {
        w.add(format!(
            "`{path}` = [{c}, {r}] is out of range; clamped to [{cc}, {rr}]"
        ));
    }
    p.mesh = [
        u32::try_from(cc).unwrap_or(64),
        u32::try_from(rr).unwrap_or(48),
    ];
}

/// Fields shared by wave and shape tables. Returns true if the key was handled.
fn common_field(
    key: &str,
    v: &Value,
    path: &str,
    enabled: &mut bool,
    texts: &mut [(&str, &mut String)],
    w: &mut Warnings,
) -> bool {
    if key == "enabled" {
        if let Some(b) = bool_field(v, path, w) {
            *enabled = b;
        }
        return true;
    }
    for (name, slot) in texts.iter_mut() {
        if key == *name {
            if let Some(s) = text_field(v, path, w) {
                **slot = s;
            }
            return true;
        }
    }
    false
}

fn read_wave(t: &Table, base: &str, w: &mut Warnings) -> WaveDef {
    let mut d = WaveDef::default();
    for (k, v) in t {
        let key = k.to_lowercase();
        let path = format!("{base}.{k}");
        let WaveDef {
            enabled,
            samples,
            params,
            init,
            frame,
            point,
        } = &mut d;
        let mut texts = [("init", init), ("frame", frame), ("point", point)];
        if common_field(&key, v, &path, enabled, &mut texts, w) {
            continue;
        }
        if key == "samples" {
            *samples = uint_field(v, &path, (1, 512), 512, w);
        } else if let Some(x) = num(v) {
            params.insert(key, x);
        } else {
            w.unknown(&path);
        }
    }
    d
}

fn read_shape(t: &Table, base: &str, w: &mut Warnings) -> ShapeDef {
    let mut d = ShapeDef::default();
    for (k, v) in t {
        let key = k.to_lowercase();
        let path = format!("{base}.{k}");
        let ShapeDef {
            enabled,
            sides,
            instances,
            params,
            init,
            frame,
        } = &mut d;
        let mut texts = [("init", init), ("frame", frame)];
        if common_field(&key, v, &path, enabled, &mut texts, w) {
            continue;
        }
        match key.as_str() {
            "sides" => *sides = uint_field(v, &path, (3, 100), 4, w),
            "instances" => *instances = uint_field(v, &path, (1, 1024), 1, w),
            _ => {
                if let Some(x) = num(v) {
                    params.insert(key, x);
                } else {
                    w.unknown(&path);
                }
            }
        }
    }
    d
}

// ----------------------------------------------------------- serialising

fn is_bare_key(k: &str) -> bool {
    !k.is_empty()
        && k.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn push_escaped(out: &mut String, s: &str, multiline: bool) {
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' if multiline => out.push('\n'),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                let _ = write!(out, "\\u{:04X}", c as u32);
            }
            c => out.push(c),
        }
    }
}

/// A single-line basic string.
fn quoted(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    push_escaped(&mut out, s, false);
    out.push('"');
    out
}

/// A multi-line string that parses back to exactly `s`. The newline right
/// after the opening delimiter is trimmed by TOML, so it is always emitted.
fn multiline(s: &str) -> String {
    let literal_ok = !s.contains("'''")
        && !s.ends_with('\'')
        && s.chars()
            .all(|c| c == '\n' || c == '\t' || (c >= ' ' && c != '\u{7f}'));
    let mut out = String::with_capacity(s.len() + 8);
    if literal_ok {
        out.push_str("'''\n");
        out.push_str(s);
        out.push_str("'''");
    } else {
        out.push_str("\"\"\"\n");
        push_escaped(&mut out, s, true);
        out.push_str("\"\"\"");
    }
    out
}

fn float(f: f64) -> String {
    format!("{f:?}")
}

fn key(k: &str) -> String {
    if is_bare_key(k) {
        k.to_string()
    } else {
        quoted(k)
    }
}

fn put_text(out: &mut String, name: &str, s: &str) {
    if !s.is_empty() {
        let _ = writeln!(out, "{name} = {}", multiline(s));
    }
}

fn put_params(out: &mut String, params: &BTreeMap<String, f64>) {
    for (k, v) in params {
        let _ = writeln!(out, "{} = {}", key(k), float(*v));
    }
}

impl Preset {
    /// Canonical `.splat` text; `parse` of the result equals `self`.
    pub fn to_toml(&self) -> String {
        let mut o = String::new();
        let _ = writeln!(o, "version = {}", self.version);

        let _ = writeln!(o, "\n[meta]");
        let _ = writeln!(o, "name = {}", quoted(&self.meta.name));
        let _ = writeln!(o, "author = {}", quoted(&self.meta.author));
        let _ = writeln!(o, "license = {}", quoted(&self.meta.license));

        let _ = writeln!(o, "\n[params]");
        let _ = writeln!(
            o,
            "feedback_format = \"{}\"",
            self.params.feedback_format.as_str()
        );
        let _ = writeln!(
            o,
            "mesh = [{}, {}]",
            self.params.mesh[0], self.params.mesh[1]
        );
        let _ = writeln!(o, "sim_rate = {}", float(self.params.sim_rate));
        put_params(&mut o, &self.params.values);

        let _ = writeln!(o, "\n[code]");
        put_text(&mut o, "init", &self.code.init);
        put_text(&mut o, "frame", &self.code.frame);
        put_text(&mut o, "vertex", &self.code.vertex);

        for wv in &self.waves {
            let _ = writeln!(o, "\n[[wave]]");
            let _ = writeln!(o, "enabled = {}", wv.enabled);
            let _ = writeln!(o, "samples = {}", wv.samples);
            put_params(&mut o, &wv.params);
            put_text(&mut o, "init", &wv.init);
            put_text(&mut o, "frame", &wv.frame);
            put_text(&mut o, "point", &wv.point);
        }
        for sh in &self.shapes {
            let _ = writeln!(o, "\n[[shape]]");
            let _ = writeln!(o, "enabled = {}", sh.enabled);
            let _ = writeln!(o, "sides = {}", sh.sides);
            let _ = writeln!(o, "instances = {}", sh.instances);
            put_params(&mut o, &sh.params);
            put_text(&mut o, "init", &sh.init);
            put_text(&mut o, "frame", &sh.frame);
        }

        if self.shader.warp.is_some() || self.shader.comp.is_some() {
            let _ = writeln!(o, "\n[shader]");
            if let Some(s) = &self.shader.warp {
                let _ = writeln!(o, "warp = {}", multiline(s));
            }
            if let Some(s) = &self.shader.comp {
                let _ = writeln!(o, "comp = {}", multiline(s));
            }
        }
        o
    }
}
