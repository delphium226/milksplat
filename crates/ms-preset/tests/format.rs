use ms_preset::{parse, FeedbackFormat, Params, Preset, Warning, PARAM_DEFAULTS};

const FULL: &str = include_str!("fixtures/full.splat");
const MINIMAL: &str = include_str!("fixtures/minimal.splat");

fn ok(s: &[u8]) -> (Preset, Vec<Warning>) {
    parse(s).unwrap()
}

fn lf(s: &str) -> String {
    s.chars().filter(|&c| c != char::from(13u8)).collect()
}

fn warned(ws: &[Warning], needle: &str) -> bool {
    ws.iter().any(|w| w.message.contains(needle))
}

#[test]
fn minimal_file_defaults() {
    let (p, w) = ok(MINIMAL.as_bytes());
    assert!(w.is_empty());
    assert_eq!(p.version, 1);
    assert_eq!(p.params.mesh, [64, 48]);
    assert_eq!(p.params.sim_rate, 60.0);
    assert_eq!(p.params.feedback_format, FeedbackFormat::Rgba8);
    assert!(p.params.values.is_empty());
    assert!(p.waves.is_empty() && p.shapes.is_empty());
    assert_eq!(p.shader.warp, None);
}

#[test]
fn empty_file_is_version_one() {
    let (p, _) = ok(b"");
    assert_eq!(p.version, 1);
}

#[test]
fn full_fixture_parses() {
    let (p, w) = ok(FULL.as_bytes());
    assert!(w.is_empty(), "{w:?}");
    assert_eq!(p.meta.name, "Ink Bloom");
    assert_eq!(p.params.feedback_format, FeedbackFormat::Fp16);
    assert_eq!(p.params.mesh, [48, 36]);
    assert_eq!(p.params.sim_rate, 30.0);
    assert_eq!(p.params.values["echo_zoom"], 1.5);
    assert_eq!(p.params.values["wave_additive"], 1.0);
    assert_eq!(p.params.values["rot"], 0.0);
    assert_eq!(lf(&p.code.init), "q1 = 0.5;\nt = 'quoted';\n");
    assert_eq!(p.waves.len(), 2);
    assert!(!p.waves[1].enabled);
    assert_eq!(p.waves[0].samples, 256);
    assert_eq!(p.waves[0].params["g"], 0.5);
    assert_eq!(p.shapes.len(), 2);
    assert_eq!(p.shapes[0].instances, 4);
    assert_eq!(p.shapes[0].params["rad"], 0.2);
    assert_eq!(p.shapes[1].sides, 3);
    assert!(p.shader.warp.as_deref().unwrap().contains("fn warp"));
    assert!(p.shader.comp.as_deref().unwrap().contains("fn comp"));
}

#[test]
fn full_fixture_roundtrip() {
    let (p, _) = ok(FULL.as_bytes());
    let text = p.to_toml();
    let (q, w) = ok(text.as_bytes());
    assert!(w.is_empty(), "{w:?}");
    assert_eq!(p, q);
    assert_eq!(text, q.to_toml());
}

#[test]
fn roundtrip_nasty_text() {
    let nasty = [
        "",
        "\nleading newline",
        "trailing quote'",
        "has ''' triple",
        "tab\there\r\nCRLF \\ \" \u{7f} \u{1} é 日本",
        "a\n\n\n",
    ];
    for s in nasty {
        let mut p = ok(MINIMAL.as_bytes()).0;
        p.meta.name = s.to_string();
        p.code.init = s.to_string();
        p.code.frame = format!("{s}{s}");
        p.shader.warp = Some(s.to_string());
        p.shader.comp = Some(String::new());
        let q = ok(p.to_toml().as_bytes()).0;
        assert_eq!(p, q, "text {s:?}");
    }
}

#[test]
fn unknown_key_warns() {
    let (p, w) = ok(b"version = 1\nbogus = 1\n[params]\nzoon = 1.0\nzoom = 1.1\n[[wave]]\ncolour = \"red\"\n[nope]\nx = 1\n[code]\nfoo = \"x\"\n");
    assert_eq!(p.params.values["zoom"], 1.1);
    assert!(!p.params.values.contains_key("zoon"));
    assert!(warned(&w, "bogus"));
    assert!(warned(&w, "params.zoon"));
    assert!(warned(&w, "wave[0].colour"));
    assert!(warned(&w, "nope"));
    assert!(warned(&w, "code.foo"));
}

#[test]
fn non_numeric_param_warns_and_uses_default() {
    let (p, w) = ok(b"[params]\nzoom = \"big\"\nmesh = \"x\"\n");
    assert!(warned(&w, "zoom"));
    assert!(warned(&w, "mesh"));
    assert!(!p.params.values.contains_key("zoom"));
    assert_eq!(p.params.param("zoom"), 1.0);
    assert_eq!(p.params.mesh, [64, 48]);
}

#[test]
fn param_defaults_and_lookup() {
    let p = Params::default();
    assert_eq!(p.param("zoom"), 1.0);
    assert_eq!(p.param("decay"), 0.98);
    assert_eq!(p.param("wave_a"), 0.8);
    assert_eq!(p.param("b2x"), 1.0);
    assert_eq!(p.param("b1ed"), 0.25);
    assert_eq!(p.param("rot"), 0.0);
    assert_eq!(p.param("no_such_var"), 0.0);
    let (q, _) = ok(b"[params]\nzoom = 2\n");
    assert_eq!(q.params.param("zoom"), 2.0);
}

#[test]
fn param_default_values() {
    let want: &[(&str, f64)] = &[
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
        ("mv_x", 12.0),
        ("mv_y", 9.0),
        ("mv_dx", 0.0),
        ("mv_dy", 0.0),
        ("mv_l", 0.9),
        ("mv_r", 1.0),
        ("mv_g", 1.0),
        ("mv_b", 1.0),
        ("mv_a", 0.0),
        ("echo_zoom", 1.0),
        ("echo_alpha", 0.0),
        ("echo_orient", 0.0),
        ("gamma", 1.0),
        ("invert", 0.0),
        ("solarize", 0.0),
        ("brighten", 0.0),
        ("darken", 0.0),
        ("darken_center", 0.0),
        ("texture_wrap", 1.0),
        ("b1n", 0.0),
        ("b2n", 0.0),
        ("b3n", 0.0),
        ("b1x", 1.0),
        ("b2x", 1.0),
        ("b3x", 1.0),
        ("b1ed", 0.25),
    ];
    assert_eq!(want.len(), PARAM_DEFAULTS.len());
    for &(n, v) in want {
        assert_eq!(Params::default().param(n), v, "default of {n}");
    }
}

#[test]
fn sim_rate_clamped() {
    let (p, w) = ok(b"[params]
sim_rate = 1000
");
    assert_eq!(p.params.sim_rate, 240.0);
    assert!(warned(&w, "sim_rate") && warned(&w, "240"));
    let (p, w) = ok(b"[params]
sim_rate = 0.5
");
    assert_eq!(p.params.sim_rate, 1.0);
    assert!(warned(&w, "sim_rate"));
    let (p, w) = ok(b"[params]
sim_rate = 30
");
    assert_eq!(p.params.sim_rate, 30.0);
    assert!(w.is_empty());
}

#[test]
fn param_defaults_cover_spec() {
    let spec = "zoom zoomexp rot warp cx cy dx dy sx sy decay wave_mode wave_r wave_g wave_b wave_a wave_x wave_y wave_mystery wave_usedots wave_thick wave_additive wave_brighten ob_size ob_r ob_g ob_b ob_a ib_size ib_r ib_g ib_b ib_a mv_x mv_y mv_dx mv_dy mv_l mv_r mv_g mv_b mv_a echo_zoom echo_alpha echo_orient gamma invert solarize brighten darken darken_center texture_wrap b1n b2n b3n b1x b2x b3x b1ed";
    for n in spec.split(' ') {
        assert!(PARAM_DEFAULTS.iter().any(|(k, _)| *k == n), "missing {n}");
    }
    assert_eq!(PARAM_DEFAULTS.len(), spec.split(' ').count());
}

#[test]
fn extra_waves_dropped_with_warning() {
    let mut s = String::from("version = 1\n");
    for _ in 0..6 {
        s.push_str("[[wave]]\nsamples = 10\n[[shape]]\nsides = 5\n");
    }
    let (p, w) = ok(s.as_bytes());
    assert_eq!(p.waves.len(), 4);
    assert_eq!(p.shapes.len(), 4);
    assert!(warned(&w, "wave[4]"));
    assert!(warned(&w, "shape[4]"));
}

#[test]
fn clamps() {
    let (p, w) = ok(b"[params]\nmesh = [2, 9999]\n[[wave]]\nsamples = 100000\n[[wave]]\nsamples = 0\n[[shape]]\nsides = 1\ninstances = 99999\n[[shape]]\nsides = 500\ninstances = 0\n");
    assert_eq!(p.params.mesh, [8, 144]);
    assert_eq!(p.waves[0].samples, 512);
    assert_eq!(p.waves[1].samples, 1);
    assert_eq!((p.shapes[0].sides, p.shapes[0].instances), (3, 1024));
    assert_eq!((p.shapes[1].sides, p.shapes[1].instances), (100, 1));
    assert!(warned(&w, "mesh"));
    assert!(warned(&w, "512"));
    assert!(warned(&w, "1024"));
    assert!(warned(&w, "100"));
    let (p, _) = ok(b"[params]\nmesh = [9999, 2]\n");
    assert_eq!(p.params.mesh, [192, 6]);
}

#[test]
fn bom_crlf_tabs_ok() {
    let mut b = vec![0xEF, 0xBB, 0xBF];
    b.extend_from_slice(
        b"version = 1\r\n[params]\r\n\tzoom\t=\t1.5\r\n[code]\r\nframe = \"x=1;\"\r\n",
    );
    let (p, w) = ok(&b);
    assert!(w.is_empty(), "{w:?}");
    assert_eq!(p.params.values["zoom"], 1.5);
    assert_eq!(p.code.frame, "x=1;");
}

#[test]
fn invalid_utf8_located() {
    let e = parse(b"version = 1\n[meta]\nname = \"ab\xFFcd\"\n").unwrap_err();
    assert_eq!((e.line, e.col), (3, 11));
    let e = parse(b"\xC3\x28").unwrap_err();
    assert_eq!((e.line, e.col), (1, 1));
}

#[test]
fn toml_syntax_error_located() {
    let e = parse(b"version = 1\n[params]\nzoom = = 2\n").unwrap_err();
    assert_eq!(e.line, 3);
    assert_eq!(e.col, 8);
    // CRLF counts as one newline; BOM stripped
    let e = parse(b"\xEF\xBB\xBFversion = 1\r\n[params]\r\nzoom = = 2\r\n").unwrap_err();
    assert_eq!((e.line, e.col), (3, 8));
    // columns count chars, not bytes
    let e = parse("[meta]\nname = \"é\" x\n".as_bytes()).unwrap_err();
    assert_eq!(e.line, 2);
    assert_eq!(e.col, 12); // 13 if bytes were counted
}

#[test]
fn future_version_rejected() {
    let e = parse(b"version = 2\n").unwrap_err();
    assert!(e.message.contains("made by a newer Milksplat"));
    assert!(parse(b"version = 0\n").is_err());
    assert!(parse(b"version = -1\n").is_err());
    assert!(parse(b"version = \"1\"\n").is_err());
    assert!(parse(b"version = 1.0\n").is_err());
}

// ---- no-panic fuzzing (seeded, no deps) ----

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn fuzz_never_panics() {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..2000 {
        let n = r.below(300);
        let b: Vec<u8> = (0..n).map(|_| r.next() as u8).collect();
        let _ = parse(&b);
    }
    let base = FULL.as_bytes();
    for _ in 0..2000 {
        let mut b = base.to_vec();
        for _ in 0..1 + r.below(6) {
            if b.is_empty() {
                break;
            }
            match r.below(4) {
                0 => {
                    let i = r.below(b.len());
                    b[i] ^= 1 << r.below(8);
                }
                1 => {
                    let i = r.below(b.len());
                    b[i] = r.next() as u8;
                }
                2 => {
                    let n = r.below(b.len());
                    b.truncate(n);
                }
                _ => {
                    let i = r.below(b.len());
                    b.remove(i);
                }
            }
        }
        if let Ok((p, _)) = parse(&b) {
            let (q, _) = parse(p.to_toml().as_bytes()).unwrap();
            assert_eq!(p, q);
        }
    }
}

#[test]
fn invalid_utf8_column_counts_chars() {
    let mut b = "name = \"é日".as_bytes().to_vec();
    b.push(0xFF);
    let e = parse(&b).unwrap_err();
    assert_eq!((e.line, e.col), (1, 11));
}
