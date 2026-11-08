//! Crash handling (spec §9.4): the run marker, the panic hook (which only
//! records), the crash report written when a panic escapes, and the log
//! ring the report quotes.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use milksplat::crash::{
    install_panic_hook, last_panic, report_escaped_panic, set_adapter_info, write_report,
    CrashInfo, PanicRecord, RunMarker,
};
use milksplat::logging::LogRing;
use milksplat::paths::{KnownDirs, Paths};

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tempdir() -> TempDir {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "milksplat-crash-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

fn crash_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    rd.filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("crash-") && n.ends_with(".txt"))
        })
        .collect()
}

#[test]
fn marker_detects_crash() {
    let dir = tempdir();
    let (marker, crashed) = RunMarker::begin(&dir.0);
    assert_eq!(crashed, None, "a first run");
    marker.set_preset("a");
    drop(marker); // no `end`: the run "crashed"
    let (marker, crashed) = RunMarker::begin(&dir.0);
    assert_eq!(crashed.as_deref(), Some("a"));
    // The new run starts with no preset recorded.
    drop(marker);
    let (marker, crashed) = RunMarker::begin(&dir.0);
    assert_eq!(crashed, None, "it crashed before showing a preset");
    marker.end();
}

#[test]
fn marker_clean_exit() {
    let dir = tempdir();
    let (marker, _) = RunMarker::begin(&dir.0);
    marker.set_preset("a");
    marker.set_preset("b");
    marker.end();
    assert!(!dir.0.join("running").exists(), "end deletes the marker");
    let (marker, crashed) = RunMarker::begin(&dir.0);
    assert_eq!(crashed, None);
    marker.end();
}

#[test]
fn marker_keeps_the_latest_preset() {
    let dir = tempdir();
    let (marker, _) = RunMarker::begin(&dir.0);
    for i in 0..50 {
        marker.set_preset(&format!("p{i}.splat"));
    }
    drop(marker);
    let (marker, crashed) = RunMarker::begin(&dir.0);
    assert_eq!(crashed.as_deref(), Some("p49.splat"));
    marker.end();
}

fn record() -> PanicRecord {
    PanicRecord {
        message: "index out of bounds".into(),
        location: Some("src/render_thread/mod.rs:42:7".into()),
        thread: "ms-render".into(),
        backtrace: "   0: milksplat::fake\n".into(),
    }
}

#[test]
fn write_report_contents() {
    let dir = tempdir();
    let ring = LogRing::new(200);
    for i in 0..250 {
        ring.push(format!("log line {i}"));
    }
    let info = CrashInfo {
        reason: "the render thread panicked".into(),
        panic: Some(record()),
        adapter: Some("NVIDIA GeForce RTX 3090 (Dx12), driver 32.0.16.1088".into()),
        preset: Some("dir/ink.splat".into()),
        version: "9.8.7",
        time: SystemTime::UNIX_EPOCH + Duration::from_secs(1_794_150_025),
        log_lines: ring.lines(),
    };
    let path = write_report(&dir.0, &info).unwrap();
    let name = path.file_name().unwrap().to_str().unwrap();
    assert_eq!(name, "crash-20261108-150025.txt");
    let text = std::fs::read_to_string(&path).unwrap();
    for want in [
        "Milksplat 9.8.7 crashed",
        "2026-11-08 15:00:25 UTC",
        "the render thread panicked",
        "index out of bounds",
        "src/render_thread/mod.rs:42:7",
        "ms-render",
        "milksplat::fake",
        "NVIDIA GeForce RTX 3090 (Dx12), driver 32.0.16.1088",
        "dir/ink.splat",
        "log line 50\n",
        "log line 249\n",
    ] {
        assert!(text.contains(want), "missing {want:?} in:\n{text}");
    }
    assert!(!text.contains("log line 49\n"), "only the last 200 lines");

    // A second report in the same second gets its own file.
    let again = write_report(&dir.0, &info).unwrap();
    assert_ne!(again, path);
    assert_eq!(crash_files(&dir.0).len(), 2);
}

#[test]
fn log_ring_keeps_the_last_lines() {
    let ring = LogRing::new(3);
    assert!(ring.lines().is_empty());
    for i in 0..5 {
        ring.push(format!("{i}"));
    }
    assert_eq!(ring.lines(), ["2", "3", "4"]);
}

/// The hook is process-wide, so everything about it is in this one test.
#[test]
fn panic_hook_records_and_only_escaped_panics_report() {
    let base = tempdir();
    let dirs = KnownDirs {
        roaming_app_data: Some(base.0.join("roaming")),
        local_app_data: Some(base.0.join("local")),
        documents: Some(base.0.join("docs")),
        exe_dir: base.0.join("exe"),
    };
    let mut paths = Paths::from_dirs(&dirs, None);
    assert!(paths.create_dirs().is_empty());
    let preset = Arc::new(Mutex::new(Some("x.splat".to_string())));
    install_panic_hook(&paths, Arc::clone(&preset));
    set_adapter_info("Test Adapter (Vulkan), driver 1.2.3".into());

    // A panic caught with catch_unwind (as naga's are) is only recorded.
    let caught = std::panic::catch_unwind(|| panic!("caught one"));
    assert!(caught.is_err());
    assert!(
        crash_files(&paths.crash_dir).is_empty(),
        "a caught panic writes no crash file"
    );
    let rec = last_panic().expect("the hook records the panic");
    assert_eq!(rec.message, "caught one");
    assert!(rec.location.as_deref().unwrap().contains("crash.rs"));

    // A panic that escapes the render thread's body is reported at its
    // boundary, on that thread, with the recorded details (the test
    // thread's own caught panic above is not mixed in).
    let handle = std::thread::Builder::new()
        .name("ms-render".into())
        .spawn(|| {
            let body = std::panic::catch_unwind(|| panic!("escaped one"));
            assert!(body.is_err());
            report_escaped_panic("the render thread panicked", false)
        })
        .unwrap();
    let report = handle.join().unwrap().expect("a report is written");
    assert_eq!(crash_files(&paths.crash_dir), std::slice::from_ref(&report));
    let text = std::fs::read_to_string(&report).unwrap();
    assert!(!text.contains("caught one"), "{text}");
    for want in [
        "escaped one",
        "ms-render",
        "x.splat",
        "Test Adapter (Vulkan), driver 1.2.3",
        env!("CARGO_PKG_VERSION"),
        "the render thread panicked",
    ] {
        assert!(text.contains(want), "missing {want:?} in:\n{text}");
    }
}
