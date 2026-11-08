use ms_preset::*;
use std::collections::HashSet;

fn preset(frame: &str, vertex: &str) -> Preset {
    let mut p = Preset::default();
    p.code.frame = frame.into();
    p.code.vertex = vertex.into();
    p
}

#[test]
fn builtin_ids_stable() {
    let a = compile_preset(preset("zoom = 1.1;", ""));
    let b = compile_preset(preset("myvar = 3; zoom = 1;", "x = x + 1;"));
    assert!(a.vars.lookup("zoom").is_some());
    assert_eq!(a.vars.lookup("zoom"), b.vars.lookup("zoom"));
    assert_eq!(a.vars.lookup("q1"), b.vars.lookup("q1"));
}

#[test]
fn bad_block_disabled_others_kept() {
    let c = compile_preset(preset("zoom = = 1;", "x = x + 1;"));
    assert!(c.frame.is_none());
    assert!(c.vertex.is_some());
    assert_eq!(c.errors.len(), 1);
    assert_eq!(c.errors[0].block, "frame");
}

#[test]
fn builtin_table_unique_and_stable() {
    let t = builtin_var_table();
    assert_eq!(t.len(), BUILTIN_VARS.len());
    let set: HashSet<_> = BUILTIN_VARS.iter().collect();
    assert_eq!(set.len(), BUILTIN_VARS.len());
    assert_eq!(t.lookup("time").unwrap().0, 0);
    assert_eq!(t.lookup("q1"), builtin_var_table().lookup("q1"));
    assert_eq!(t.name(t.lookup("reg99").unwrap()), "reg99");
    assert_eq!(
        t.lookup("reg99").unwrap().0 as usize,
        BUILTIN_VARS.len() - 1
    );
    assert_eq!(
        t.lookup("q32").unwrap().0 as usize + 1,
        t.lookup("q1").unwrap().0 as usize + 32
    );
}

#[test]
fn builtin_covers_param_defaults_in_order() {
    let t = builtin_var_table();
    let mut last = 0;
    for (name, _) in PARAM_DEFAULTS {
        let id = t.lookup(name).unwrap().0;
        assert!(id >= last, "{name} out of order");
        last = id;
    }
}

#[test]
fn user_vars_after_builtins() {
    let c = compile_preset(preset("myvar = 3;", ""));
    assert!(c.vars.lookup("myvar").unwrap().0 as usize >= BUILTIN_VARS.len());
}

#[test]
fn empty_blocks_are_none_without_error() {
    let c = compile_preset(preset("  // nothing\n ", ";"));
    assert!(c.frame.is_none() && c.vertex.is_none() && c.init.is_none());
    assert!(c.errors.is_empty());
}

#[test]
fn disabled_wave_still_reports_error() {
    let mut p = Preset::default();
    p.waves.push(WaveDef::default());
    p.waves.push(WaveDef {
        enabled: false,
        point: "r = = 1;".into(),
        samples: 64,
        ..WaveDef::default()
    });
    p.shapes.push(ShapeDef {
        frame: "x = ;".into(),
        ..ShapeDef::default()
    });
    let c = compile_preset(p);
    assert_eq!(c.waves.len(), 2);
    assert!(!c.waves[1].enabled);
    assert_eq!(c.waves[1].samples, 64);
    assert!(c.waves[1].point.is_none());
    let blocks: Vec<_> = c.errors.iter().map(|e| e.block.as_str()).collect();
    assert_eq!(blocks, ["wave[1].point", "shape[0].frame"]);
    assert_eq!(
        c.waves[1].vars.lookup("q1"),
        builtin_var_table().lookup("q1")
    );
}
