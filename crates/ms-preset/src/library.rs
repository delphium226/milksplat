//! Preset library discovery.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Path relative to its root, `/`-separated, lowercased.
    pub id: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub struct Library {
    pub entries: Vec<Entry>,
}

impl Library {
    /// Recursively scan `roots` for `*.splat` files (symlinks are not
    /// followed; unreadable entries are skipped with a warning). Entries are
    /// sorted by id. When several roots yield the same id the FIRST root
    /// wins, so callers pass the user root before the bundled one.
    pub fn scan(roots: &[PathBuf]) -> Library {
        let mut entries: Vec<Entry> = Vec::new();
        for root in roots {
            let mut found = Vec::new();
            walk(root, root, &mut found);
            for e in found {
                if !entries.iter().any(|x| x.id == e.id) {
                    entries.push(e);
                }
            }
        }
        entries.sort_by(|a, b| a.id.cmp(&b.id));
        Library { entries }
    }
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<Entry>) {
    let rd = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            tracing::warn!("cannot read directory {}: {e}", dir.display());
            return;
        }
    };
    for item in rd {
        let item = match item {
            Ok(i) => i,
            Err(e) => {
                tracing::warn!("cannot read entry in {}: {e}", dir.display());
                continue;
            }
        };
        // file_type() does not follow symlinks.
        let ft = match item.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                tracing::warn!("cannot stat {}: {e}", item.path().display());
                continue;
            }
        };
        let path = item.path();
        if ft.is_dir() {
            walk(root, &path, out);
        } else if ft.is_file() {
            let is_splat = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("splat"));
            if !is_splat {
                continue;
            }
            let Ok(rel) = path.strip_prefix(root) else {
                continue;
            };
            let parts: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            out.push(Entry {
                id: parts.join("/").to_lowercase(),
                path,
            });
        }
    }
}
