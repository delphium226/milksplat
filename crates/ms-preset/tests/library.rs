use ms_preset::*;
use std::fs;
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ms_preset_lib_{}_{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn scan_nested_filters_and_overrides() {
    let user = tmp("user");
    let bundled = tmp("bundled");
    fs::create_dir_all(bundled.join("Sub/Deep")).unwrap();
    fs::write(bundled.join("A.splat"), "x").unwrap();
    fs::write(bundled.join("Sub/Deep/B.SPLAT"), "x").unwrap();
    fs::write(bundled.join("notes.txt"), "x").unwrap();
    fs::write(user.join("a.splat"), "x").unwrap();
    let lib = Library::scan(&[user.clone(), bundled.clone()]);
    let ids: Vec<_> = lib.entries.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, ["a.splat", "sub/deep/b.splat"]);
    assert!(lib.entries[0].path.starts_with(&user));
    let _ = fs::remove_dir_all(&user);
    let _ = fs::remove_dir_all(&bundled);
}

#[test]
fn scan_missing_root_is_empty() {
    let lib = Library::scan(&[PathBuf::from("definitely/not/here")]);
    assert!(lib.entries.is_empty());
}
