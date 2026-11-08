//! Hot reload (spec §9.3): the watcher reports a changed `.splat` file by
//! library id, and rescans the library when a preset is added or removed.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use milksplat::hotreload::{preset_id, watch, Change};

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tempdir() -> TempDir {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "milksplat-hotreload-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("Sub")).unwrap();
    TempDir(dir)
}

/// Waits (up to 5 s) for a change that satisfies `want`.
fn wait_for(w: &milksplat::hotreload::Watch, want: impl Fn(&Change) -> bool) -> Change {
    let start = Instant::now();
    loop {
        if let Some(c) = w.try_recv() {
            if want(&c) {
                return c;
            }
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "no matching change"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn ids_match_the_library() {
    let root = PathBuf::from("root");
    assert_eq!(
        preset_id(&root, &root.join("Sub").join("Ink Bloom.SPLAT")),
        Some("sub/ink bloom.splat".to_string())
    );
    assert_eq!(preset_id(&root, &root.join("notes.txt")), None);
    assert_eq!(preset_id(&root, &PathBuf::from("elsewhere/a.splat")), None);
}

#[test]
fn edits_and_new_files_are_reported() {
    let dir = tempdir();
    let preset = dir.0.join("Sub").join("A.splat");
    std::fs::write(&preset, "version = 1\n").unwrap();
    let w = watch(std::slice::from_ref(&dir.0)).expect("the watcher starts");
    // Give the OS watch a moment to be in place.
    std::thread::sleep(Duration::from_millis(100));

    let t0 = Instant::now();
    std::fs::write(&preset, "version = 1\n[params]\nzoom = 1.1\n").unwrap();
    let c = wait_for(&w, |c| c.ids.iter().any(|id| id == "sub/a.splat"));
    assert!(
        t0.elapsed() < Duration::from_millis(1000),
        "reported after {:?}",
        t0.elapsed()
    );
    assert!(c.library.is_none(), "an edit needs no rescan");

    // Other files are ignored; a new preset rescans the library.
    std::fs::write(dir.0.join("readme.txt"), "x").unwrap();
    std::fs::write(dir.0.join("b.splat"), "version = 1\n").unwrap();
    let c = wait_for(&w, |c| c.library.is_some());
    let ids: Vec<String> = c
        .library
        .unwrap()
        .entries
        .into_iter()
        .map(|e| e.id)
        .collect();
    assert_eq!(ids, ["b.splat", "sub/a.splat"]);

    // So does a removed one.
    std::fs::remove_file(&preset).unwrap();
    let c = wait_for(&w, |c| {
        c.library.as_ref().is_some_and(|l| l.entries.len() == 1)
    });
    assert_eq!(c.library.unwrap().entries[0].id, "b.splat");
}
