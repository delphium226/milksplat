//! The config file (spec §9.3): defaults on first run, a bad file renamed
//! aside, one bad field keeping the rest, unknown keys, atomic saves, the
//! debounced saver, the per-device A/V offset and the reactivity levels.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use milksplat::config::{
    load_config, load_config_checked, save_config, Config, ConfigStore, Quality, WindowState,
    AV_OFFSET_MAX_MS, AV_OFFSET_MIN_MS, REACTIVITY_MAX, REACTIVITY_MIN,
};

/// A fresh temp directory, removed on drop.
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tempdir() -> TempDir {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "milksplat-config-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

#[test]
fn missing_config_defaults_written() {
    let dir = tempdir();
    // The parent directory does not exist yet either.
    let path = dir.0.join("Milksplat").join("config.toml");
    let (c, warnings) = load_config(&path);
    assert_eq!(c, Config::default());
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(path.is_file(), "the defaults are written out");
    // What was written loads back cleanly.
    let (again, warnings) = load_config(&path);
    assert_eq!(again, Config::default());
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn garbage_config_renamed_bad() {
    let dir = tempdir();
    let path = dir.0.join("config.toml");
    let bad = sibling(&path, ".bad");
    std::fs::write(&bad, "an older bad file").unwrap();
    std::fs::write(&path, "this is = = not [toml").unwrap();
    let (c, warnings) = load_config(&path);
    assert_eq!(c, Config::default());
    assert!(!warnings.is_empty());
    assert_eq!(
        std::fs::read_to_string(&bad).unwrap(),
        "this is = = not [toml",
        "the bad file is kept aside (replacing an older one)"
    );
    // A fresh defaults file takes its place.
    let (again, warnings) = load_config(&path);
    assert_eq!(again, Config::default());
    assert!(warnings.is_empty(), "{warnings:?}");

    // Not UTF-8 counts as invalid too.
    std::fs::write(&path, [0xff, 0xfe, 0x00, 0x80]).unwrap();
    let (c, warnings) = load_config(&path);
    assert_eq!(c, Config::default());
    assert!(!warnings.is_empty());
    assert_eq!(std::fs::read(&bad).unwrap(), [0xff, 0xfe, 0x00, 0x80]);
}

#[test]
fn one_bad_field_others_kept() {
    let dir = tempdir();
    let path = dir.0.join("config.toml");
    std::fs::write(&path, "auto_advance_s = \"x\"\nblend_s = 5\n").unwrap();
    let (c, warnings) = load_config(&path);
    assert_eq!(c.auto_advance_s, 60.0);
    assert_eq!(c.blend_s, 5.0);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("auto_advance_s"), "{warnings:?}");
    assert!(
        !sibling(&path, ".bad").exists(),
        "a bad field is not a bad file"
    );

    // Out-of-range and non-finite values are refused the same way.
    std::fs::write(
        &path,
        "blend_s = nan\nhard_cut_strength = -1.0\nquality = \"ultra\"\nhard_cuts = false\n\
         [window]\nsize = [0, 0]\nposition = [10, 20]\n",
    )
    .unwrap();
    let (c, warnings) = load_config(&path);
    let d = Config::default();
    assert_eq!(c.blend_s, d.blend_s);
    assert_eq!(c.hard_cut_strength, d.hard_cut_strength);
    assert_eq!(c.quality, d.quality);
    assert!(!c.hard_cuts, "the good field is kept");
    assert_eq!(c.window.size, d.window.size);
    assert_eq!(c.window.position, Some([10, 20]));
    assert_eq!(warnings.len(), 4, "{warnings:?}");

    // A/V offsets: a bad entry is dropped, an out-of-range one clamped.
    std::fs::write(
        &path,
        "[av_offsets_ms]\n\"Speakers\" = 30\n\"Headset\" = 99999\n\"Bad\" = \"x\"\n\
         \"Old\" = -40\n",
    )
    .unwrap();
    let (c, warnings) = load_config(&path);
    let want: BTreeMap<String, i32> = [
        ("Speakers".to_string(), 30),
        ("Headset".to_string(), AV_OFFSET_MAX_MS),
        ("Old".to_string(), 0),
    ]
    .into();
    assert_eq!(c.av_offsets_ms, want, "negative offsets clamp to 0");
    assert_eq!(warnings.len(), 3, "{warnings:?}");
}

#[test]
fn unknown_key_warns() {
    let dir = tempdir();
    let path = dir.0.join("config.toml");
    std::fs::write(
        &path,
        "blend_s = 3.0\nfrobnicate = 1\n[window]\nsize = [800, 600]\nwidth = 3\n",
    )
    .unwrap();
    let (c, warnings) = load_config(&path);
    assert_eq!(c.blend_s, 3.0);
    assert_eq!(c.window.size, [800, 600]);
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(warnings.iter().any(|w| w.contains("frobnicate")));
    assert!(warnings.iter().any(|w| w.contains("window.width")));
}

fn everything_set() -> Config {
    Config {
        audio_device: Some("Speakers (Realtek(R) Audio)".into()),
        av_offsets_ms: [
            ("Speakers (Realtek(R) Audio)".to_string(), 40),
            ("WH-1000XM4 \"Stereo\"".to_string(), 230),
        ]
        .into(),
        auto_advance_s: 12.5,
        blend_s: 0.0,
        hard_cuts: false,
        hard_cut_strength: 3.25,
        quality: Quality::Low,
        favourites: ["a.splat".to_string(), "dir/b c.splat".to_string()].into(),
        bans: ["x.splat".to_string()].into(),
        blocklist: BTreeSet::from(["crash.splat".to_string()]),
        window: WindowState {
            size: [1600, 900],
            position: Some([-1200, 40]),
            monitor: Some(r"\\.\DISPLAY2".into()),
        },
        fullscreen_monitor: Some(r"\\.\DISPLAY1".into()),
        reactivity: 0.75,
        preset_reactivity: [
            ("a.splat".to_string(), 1.5),
            ("dir/b c.splat".to_string(), 0.25),
        ]
        .into(),
    }
}

#[test]
fn save_load_roundtrip() {
    let dir = tempdir();
    let path = dir.0.join("config.toml");
    let c = everything_set();
    save_config(&path, &c).unwrap();
    let (back, warnings) = load_config(&path);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(back, c);
    for q in [Quality::Auto, Quality::High, Quality::Medium, Quality::Low] {
        let c = Config {
            quality: q,
            ..Config::default()
        };
        save_config(&path, &c).unwrap();
        assert_eq!(load_config(&path).0.quality, q);
    }
}

/// The file is written next to the target and renamed over it, so a
/// reader never sees a half-written config: when the temp file cannot be
/// written, the old config is untouched.
#[test]
fn save_is_atomic() {
    let dir = tempdir();
    let path = dir.0.join("config.toml");
    save_config(&path, &Config::default()).unwrap();
    let before = std::fs::read(&path).unwrap();

    // Block the temp path: the save fails and the target is unchanged.
    let tmp = sibling(&path, ".tmp");
    std::fs::create_dir(&tmp).unwrap();
    assert!(save_config(&path, &everything_set()).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);

    std::fs::remove_dir(&tmp).unwrap();
    save_config(&path, &everything_set()).unwrap();
    assert!(!tmp.exists(), "the temp file is renamed away");
    assert_eq!(load_config(&path).0, everything_set());
}

/// The saver writes at most once a second while changes keep coming, and
/// always writes the latest config when it is finished.
#[test]
fn store_debounces_and_flushes() {
    let dir = tempdir();
    let path = dir.0.join("config.toml");
    let (store, saver) = ConfigStore::start(path.clone(), Config::default());
    let start = Instant::now();
    let mut n = 0;
    while start.elapsed() < Duration::from_millis(1500) {
        n += 1;
        store.update(|c| c.auto_advance_s = f64::from(n));
        std::thread::sleep(Duration::from_millis(5));
    }
    let saves = store.saves();
    assert!(
        (1..=3).contains(&saves),
        "{saves} saves in 1.5 s of changes"
    );
    saver.finish();
    assert_eq!(load_config(&path).0.auto_advance_s, f64::from(n));
    assert_eq!(store.get().auto_advance_s, f64::from(n));
}

#[test]
fn av_offset_per_device() {
    let mut c = Config::default();
    assert_eq!(c.av_offset_ms("Speakers", false), 0);
    assert_eq!(c.av_offset_ms("Buds", true), 200, "Bluetooth default");
    c.av_offsets_ms.insert("Buds".into(), 120);
    assert_eq!(c.av_offset_ms("Buds", true), 120, "a stored value wins");
    assert_eq!((AV_OFFSET_MIN_MS, AV_OFFSET_MAX_MS), (0, 2000));
    // A stored value out of range (an older file) is clamped on use.
    c.av_offsets_ms.insert("Old".into(), -40);
    assert_eq!(c.av_offset_ms("Old", false), 0);
}

/// A config that can be neither read nor moved aside is never
/// overwritten: the store is told not to save.
#[test]
fn unreadable_unmovable_config_is_not_overwritten() {
    let dir = tempdir();
    // A directory where the file should be cannot be read as a file, and a
    // non-empty `.bad` directory cannot be replaced by the rename.
    let path = dir.0.join("config.toml");
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("keep.txt"), "user data").unwrap();
    let bad = sibling(&path, ".bad");
    std::fs::create_dir(&bad).unwrap();
    std::fs::write(bad.join("x"), "x").unwrap();

    let loaded = load_config_checked(&path);
    assert_eq!(loaded.config, Config::default());
    assert!(!loaded.persist, "{:?}", loaded.warnings);

    let (store, saver) = ConfigStore::start(path.clone(), loaded.config);
    store.disable_saving();
    store.update(|c| c.blend_s = 9.0);
    saver.finish();
    assert_eq!(store.saves(), 0);
    assert!(
        path.is_dir() && path.join("keep.txt").is_file(),
        "left alone"
    );

    // A normal file is fine to save.
    let ok = dir.0.join("ok.toml");
    std::fs::write(&ok, "blend_s = 1.0\n").unwrap();
    assert!(load_config_checked(&ok).persist);
}

/// The reactivity settings: the global multiplier is clamped to
/// 0..=2.5, per-preset levels to 0.25..=2.5; 1x entries are dropped and a
/// bad entry is ignored with a warning.
#[test]
fn reactivity_values_checked() {
    let dir = tempdir();
    let path = dir.0.join("config.toml");
    std::fs::write(
        &path,
        "reactivity = 9.0\n[preset_reactivity]\n\"a.splat\" = 1.5\n\"b.splat\" = 9\n\
         \"c.splat\" = 0.1\n\"d.splat\" = 1.0\n\"e.splat\" = \"x\"\n\"f.splat\" = 2\n",
    )
    .unwrap();
    let (c, warnings) = load_config(&path);
    assert_eq!(c.reactivity, REACTIVITY_MAX);
    let want: BTreeMap<String, f64> = [
        ("a.splat".to_string(), 1.5),
        ("b.splat".to_string(), REACTIVITY_MAX),
        ("c.splat".to_string(), REACTIVITY_MIN),
        ("f.splat".to_string(), 2.0),
    ]
    .into();
    assert_eq!(c.preset_reactivity, want);
    // reactivity, b, c and e.
    assert_eq!(warnings.len(), 4, "{warnings:?}");

    for bad in ["-1.0", "nan", "\"loud\""] {
        std::fs::write(&path, format!("reactivity = {bad}\n")).unwrap();
        let (c, warnings) = load_config(&path);
        assert_eq!(c.reactivity, 1.0, "{bad}");
        assert_eq!(warnings.len(), 1, "{bad}: {warnings:?}");
    }
}

/// Setting a preset's level stores it; setting 1x removes the entry, so
/// the saved config never holds 1.0 levels.
#[test]
fn preset_reactivity_set_and_reset() {
    let mut c = Config::default();
    assert_eq!(c.preset_reactivity("a.splat"), 1.0);
    c.set_preset_reactivity("a.splat", 1.75);
    assert_eq!(c.preset_reactivity("a.splat"), 1.75);
    c.set_preset_reactivity("a.splat", 1.0);
    assert!(c.preset_reactivity.is_empty());
    assert_eq!(c.preset_reactivity("a.splat"), 1.0);

    let dir = tempdir();
    let path = dir.0.join("config.toml");
    c.set_preset_reactivity("b.splat", 0.5);
    save_config(&path, &c).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("\"b.splat\" = 0.5"), "{text}");
    c.set_preset_reactivity("b.splat", 1.0);
    save_config(&path, &c).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(!text.contains("b.splat"), "{text}");
    assert_eq!(load_config(&path).0, c);
}
