//! Scans user shader bodies for user textures and blur levels (spec §6.5).
//!
//! The scan works on identifiers outside comments, so a commented-out
//! `sampler_foo` neither takes a texture slot nor loads a file.

/// Most user textures a shader set can bind (group 1 slots).
pub(crate) const MAX_USER_TEXTURES: usize = 16;

/// The names after `sampler_` that the prelude declares itself.
const PRELUDE_TEXTURES: [&str; 10] = [
    "main",
    "blur1",
    "blur2",
    "blur3",
    "noise_lq",
    "noise_mq",
    "noise_hq",
    "noisevol_lq",
    "noisevol_hq",
    "spectrum",
];

/// What one shader body references.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Scan {
    /// User texture binding names (`X` of `sampler_X`), in first-appearance
    /// order, without duplicates.
    pub textures: Vec<String>,
    /// Highest blur level used via `GetBlurN` or `sampler_blurN` (0 if none).
    pub blur_levels: u8,
    /// Problems found while scanning (e.g. an empty texture file name).
    pub errors: Vec<String>,
}

/// Scans a user shader body.
///
/// A user texture is an identifier `sampler_X` with `X` matching
/// `[a-z0-9_]+` (WGSL is case-sensitive) that the prelude does not declare.
/// `sampler_X_s` (the texture's sampler) refers to texture `X`, so texture
/// names cannot themselves end in `_s`. A name that is only a prefix
/// (`fc_`, ...) is rejected with an error.
pub(crate) fn scan(source: &str) -> Scan {
    let mut out = Scan::default();
    for ident in identifiers(source) {
        if let Some(level) = blur_level(ident) {
            out.blur_levels = out.blur_levels.max(level);
            continue;
        }
        let Some(rest) = ident.strip_prefix("sampler_") else {
            continue;
        };
        let valid = !rest.is_empty()
            && rest
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
        if !valid {
            continue;
        }
        let name = match rest.strip_suffix("_s") {
            Some(stem) if !stem.is_empty() => stem,
            _ => rest,
        };
        if PRELUDE_TEXTURES.contains(&name) {
            continue;
        }
        if split_prefix(name).1.is_empty() {
            // `sampler_fc_` and the like: no file name after the prefix. The
            // identifier stays undeclared, so the stage then fails to compile.
            let e = format!("`{ident}`: empty texture name after the `{name}` prefix");
            if !out.errors.contains(&e) {
                out.errors.push(e);
            }
            continue;
        }
        if !out.textures.iter().any(|t| t == name) {
            out.textures.push(name.to_string());
        }
    }
    out
}

fn blur_level(ident: &str) -> Option<u8> {
    let digit = ident
        .strip_prefix("GetBlur")
        .or_else(|| ident.strip_prefix("sampler_blur"))?;
    match digit {
        "1" => Some(1),
        "2" => Some(2),
        "3" => Some(3),
        _ => None,
    }
}

/// How a user texture is sampled, from its name prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SamplerKind {
    /// `fw_` (and no prefix): linear filtering, wrap addressing.
    LinearWrap,
    /// `fc_`: linear filtering, clamp addressing.
    LinearClamp,
    /// `pw_`: point sampling, wrap addressing.
    PointWrap,
    /// `pc_`: point sampling, clamp addressing.
    PointClamp,
}

/// Splits a binding name into its sampler kind and the texture file name
/// (the name with the prefix stripped): `fc_checker` -> (LinearClamp,
/// `checker`); `checker` -> (LinearWrap, `checker`).
pub(crate) fn split_prefix(name: &str) -> (SamplerKind, &str) {
    let prefixes = [
        ("fw_", SamplerKind::LinearWrap),
        ("fc_", SamplerKind::LinearClamp),
        ("pw_", SamplerKind::PointWrap),
        ("pc_", SamplerKind::PointClamp),
    ];
    for (prefix, kind) in prefixes {
        if let Some(file) = name.strip_prefix(prefix) {
            return (kind, file);
        }
    }
    (SamplerKind::LinearWrap, name)
}

/// Assigns group 1 slots in first-appearance order across `scans` (warp
/// first, then comp). Returns the slotted names (at most
/// [`MAX_USER_TEXTURES`]) and the names left without a slot.
pub(crate) fn allocate_slots<'a>(
    scans: impl IntoIterator<Item = &'a Scan>,
) -> (Vec<String>, Vec<String>) {
    let mut slotted: Vec<String> = Vec::new();
    let mut overflow: Vec<String> = Vec::new();
    for name in scans.into_iter().flat_map(|s| &s.textures) {
        if slotted.contains(name) || overflow.contains(name) {
            continue;
        }
        if slotted.len() < MAX_USER_TEXTURES {
            slotted.push(name.clone());
        } else {
            overflow.push(name.clone());
        }
    }
    (slotted, overflow)
}

/// ASCII identifiers (`[A-Za-z_][A-Za-z0-9_]*`) outside `//` and (nested)
/// `/* */` comments. Runs starting with a digit (numbers) are skipped.
fn identifiers(source: &str) -> Vec<&str> {
    let bytes = source.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if b == b'/' && bytes.get(i + 1) == Some(&b'*') {
            let mut depth = 1usize;
            i += 2;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
        } else if is_word(b) {
            let start = i;
            while i < bytes.len() && is_word(bytes[i]) {
                i += 1;
            }
            if !b.is_ascii_digit() {
                // Word bytes are ASCII, so these are char boundaries.
                out.push(&source[start..i]);
            }
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_user_textures_in_order() {
        let s = scan(
            "let a = textureSample(sampler_fc_checker, sampler_fc_checker_s, uv);\n\
             let b = textureSample(sampler_main, s_fw, uv) + GetBlur1(uv);\n\
             let c = textureSample(sampler_clouds, sampler_clouds_s, uv);\n\
             let d = sampler_fc_checker; let e = sampler_Upper; let f = xsampler_no;",
        );
        assert_eq!(s.textures, ["fc_checker", "clouds"]);
        assert_eq!(s.blur_levels, 1);
    }

    #[test]
    fn sampler_suffix_alone_names_the_texture() {
        assert_eq!(scan("sampler_pw_dots_s").textures, ["pw_dots"]);
        assert_eq!(scan("sampler_main_s").textures, Vec::<String>::new());
    }

    #[test]
    fn empty_file_name_is_rejected() {
        let s = scan("sampler_fc_ sampler_fc__s sampler_pw_ sampler_ok");
        assert_eq!(s.textures, ["ok"]);
        assert_eq!(
            s.errors,
            [
                "`sampler_fc_`: empty texture name after the `fc_` prefix",
                "`sampler_fc__s`: empty texture name after the `fc_` prefix",
                "`sampler_pw_`: empty texture name after the `pw_` prefix",
            ]
        );
    }

    #[test]
    fn comments_are_ignored() {
        let s = scan("// sampler_a GetBlur3\n/* sampler_b /* nested */ GetBlur2 */ sampler_c");
        assert_eq!(s.textures, ["c"]);
        assert_eq!(s.blur_levels, 0);
    }

    #[test]
    fn blur_levels_take_the_highest() {
        assert_eq!(scan("sampler_blur3 GetBlur1 GetBlur2").blur_levels, 3);
        assert_eq!(scan("GetBlur4 sampler_blur0 GetBlur").blur_levels, 0);
        assert_eq!(scan("1.0 + 2u").blur_levels, 0);
    }

    #[test]
    fn prefixes_split() {
        assert_eq!(
            split_prefix("fc_checker"),
            (SamplerKind::LinearClamp, "checker")
        );
        assert_eq!(split_prefix("fw_a"), (SamplerKind::LinearWrap, "a"));
        assert_eq!(split_prefix("pw_a"), (SamplerKind::PointWrap, "a"));
        assert_eq!(split_prefix("pc_a"), (SamplerKind::PointClamp, "a"));
        assert_eq!(
            split_prefix("checker"),
            (SamplerKind::LinearWrap, "checker")
        );
    }

    #[test]
    fn slots_overflow_after_sixteen() {
        let warp = Scan {
            textures: (0..10).map(|i| format!("w{i}")).collect(),
            blur_levels: 0,
            errors: Vec::new(),
        };
        let comp = Scan {
            textures: ["w0".to_string()]
                .into_iter()
                .chain((0..8).map(|i| format!("c{i}")))
                .collect(),
            blur_levels: 0,
            errors: Vec::new(),
        };
        let (slotted, overflow) = allocate_slots([&warp, &comp]);
        assert_eq!(slotted.len(), 16);
        assert_eq!(slotted[..2], ["w0", "w1"]);
        assert_eq!(slotted[10..], ["c0", "c1", "c2", "c3", "c4", "c5"]);
        assert_eq!(overflow, ["c6", "c7"]);
    }
}
