//! Preset hot reload (spec §9.3): watches the preset roots (recursively,
//! debounced 300 ms) and reports changed `.splat` files by library id.
//!
//! The watcher's own thread (notify's debouncer) does the work: it maps
//! paths to ids and, when a preset was added or removed, rescans the
//! library, so the render thread only receives finished [`Change`]s and
//! never touches the disk for this. The render thread reloads the preset on
//! screen if its id changed, and hands a new library to the director.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use ms_preset::Library;
use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_mini::{new_debouncer, DebounceEventResult, Debouncer};

/// Changes to the same file within this are reported once.
pub const DEBOUNCE: Duration = Duration::from_millis(300);

/// One debounced batch of changes.
#[derive(Debug, Clone)]
pub struct Change {
    /// Library ids of the `.splat` files that were written, added or removed.
    pub ids: Vec<String>,
    /// The library rescanned, when a preset was added or removed (or a
    /// directory changed).
    pub library: Option<Library>,
}

/// A running watch; dropping it stops watching.
pub struct Watch {
    rx: Receiver<Change>,
    _debouncer: Debouncer<RecommendedWatcher>,
}

impl Watch {
    /// The next change, if any. Never blocks.
    pub fn try_recv(&self) -> Option<Change> {
        self.rx.try_recv().ok()
    }
}

/// The library id of `path` under `root` (as [`Library::scan`] makes it:
/// relative, `/`-separated, lowercased), if it is a `.splat` file there.
pub fn preset_id(root: &Path, path: &Path) -> Option<String> {
    let splat = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("splat"));
    if !splat {
        return None;
    }
    let rel = path.strip_prefix(root).ok()?;
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    Some(parts.join("/").to_lowercase())
}

/// Starts watching `roots` (the preset roots, user first, as scanned).
/// Roots that do not exist are skipped. `None` (logged) if nothing could be
/// watched.
pub fn watch(roots: &[PathBuf]) -> Option<Watch> {
    let (tx, rx) = crossbeam_channel::unbounded();
    // Events name paths under the root as given, or under its canonical
    // form; keep both for the id mapping.
    let mapped: Vec<(PathBuf, Option<PathBuf>)> = roots
        .iter()
        .map(|r| (r.clone(), std::fs::canonicalize(r).ok()))
        .collect();
    let scan_roots = roots.to_vec();
    let mut library = Library::scan(&scan_roots);
    let handler = move |res: DebounceEventResult| match res {
        Ok(events) => {
            let paths: Vec<PathBuf> = events.into_iter().map(|e| e.path).collect();
            on_events(&paths, &mapped, &scan_roots, &mut library, &tx);
        }
        Err(e) => tracing::warn!("preset watcher: {e}"),
    };
    let mut debouncer = match new_debouncer(DEBOUNCE, handler) {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!("cannot watch the preset directories ({e}); hot reload is off");
            return None;
        }
    };
    let mut watching = 0;
    for root in roots {
        if !root.is_dir() {
            continue;
        }
        match debouncer.watcher().watch(root, RecursiveMode::Recursive) {
            Ok(()) => watching += 1,
            Err(e) => tracing::warn!("cannot watch {}: {e}", root.display()),
        }
    }
    if watching == 0 {
        tracing::warn!("no preset directory to watch; hot reload is off");
        return None;
    }
    Some(Watch {
        rx,
        _debouncer: debouncer,
    })
}

/// Turns a batch of changed paths into a [`Change`] (on the watcher's
/// thread). A `.splat` path that is not in the library (added), no longer
/// exists (removed), or a changed directory, rescans the library.
fn on_events(
    paths: &[PathBuf],
    roots: &[(PathBuf, Option<PathBuf>)],
    scan_roots: &[PathBuf],
    library: &mut Library,
    tx: &Sender<Change>,
) {
    let mut ids = Vec::new();
    let mut rescan = false;
    for path in paths {
        let id = roots.iter().find_map(|(root, canon)| {
            preset_id(root, path).or_else(|| canon.as_deref().and_then(|c| preset_id(c, path)))
        });
        match id {
            Some(id) => {
                let known = library.entries.iter().any(|e| e.id == id);
                if !known || !path.exists() {
                    rescan = true;
                }
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
            // A directory added, removed or renamed may hold presets.
            None if path.is_dir() || (!path.exists() && path.extension().is_none()) => {
                rescan = true;
            }
            None => {}
        }
    }
    if ids.is_empty() && !rescan {
        return;
    }
    let library = rescan.then(|| {
        *library = Library::scan(scan_roots);
        library.clone()
    });
    tracing::debug!(
        "presets changed: {ids:?} (rescanned: {})",
        library.is_some()
    );
    let _ = tx.send(Change { ids, library });
}
