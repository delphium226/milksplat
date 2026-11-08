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
/// Both paths are compared as text with `/` separators and without a
/// `\\?\` prefix, ignoring case on Windows (its file names are
/// case-insensitive, and notify may report a different case or form than
/// the root was given in). Pass both in the same form (both absolute, see
/// [`watch`]).
pub fn preset_id(root: &Path, path: &Path) -> Option<String> {
    let splat = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("splat"));
    if !splat {
        return None;
    }
    let root = normalise(root);
    let path = normalise(path);
    let rel = path.strip_prefix(root.as_str())?.strip_prefix('/')?;
    let parts: Vec<&str> = rel.split('/').filter(|p| !p.is_empty()).collect();
    (!parts.is_empty()).then(|| parts.join("/").to_lowercase())
}

/// `p` as comparable text (see [`preset_id`]).
fn normalise(p: &Path) -> String {
    let text = p.to_string_lossy().replace('\\', "/");
    let text = text.strip_prefix("//?/").unwrap_or(&text);
    let text = text.trim_end_matches('/');
    if cfg!(windows) {
        text.to_lowercase()
    } else {
        text.to_string()
    }
}

/// Starts watching `roots` (the preset roots, user first, as scanned).
/// Roots that do not exist are skipped. `None` (logged) if nothing could be
/// watched.
pub fn watch(roots: &[PathBuf]) -> Option<Watch> {
    let (tx, rx) = crossbeam_channel::unbounded();
    // notify reports absolute paths under the root it watches, so watch and
    // map absolute roots (`absolute`, unlike `canonicalize`, adds no `\\?\`
    // prefix; on Windows it also resolves `..`).
    let roots: Vec<PathBuf> = roots
        .iter()
        .map(|r| std::path::absolute(r).unwrap_or_else(|_| r.clone()))
        .collect();
    let mapped = roots.clone();
    let scan_roots = roots.clone();
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
    for root in &roots {
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
    roots: &[PathBuf],
    scan_roots: &[PathBuf],
    library: &mut Library,
    tx: &Sender<Change>,
) {
    let mut ids = Vec::new();
    let mut rescan = false;
    for path in paths {
        let id = roots.iter().find_map(|root| preset_id(root, path));
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
