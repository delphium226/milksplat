//! Where presets and textures live (spec §9.3). Task 24 adds the config,
//! log and crash directories.

use std::path::{Path, PathBuf};

/// The preset and texture directories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    /// `presets\` next to the exe, or the `--presets` override.
    pub bundled_presets: PathBuf,
    /// `textures\` beside the bundled preset directory.
    pub bundled_textures: PathBuf,
    /// `%USERPROFILE%\Documents\Milksplat\Presets`, if it exists.
    pub user_presets: Option<PathBuf>,
}

impl Paths {
    /// Finds the directories: `presets_override` (from `--presets`) or
    /// `presets\` next to the executable (the current directory if the
    /// executable's path is unknown), and the user preset directory when it
    /// exists.
    pub fn discover(presets_override: Option<PathBuf>) -> Paths {
        let bundled_presets = presets_override.unwrap_or_else(|| exe_dir().join("presets"));
        let user_presets = std::env::var_os("USERPROFILE")
            .map(|home| user_presets_in(Path::new(&home)))
            .filter(|dir| dir.is_dir());
        Paths::new(bundled_presets, user_presets)
    }

    /// The paths for a given bundled preset directory and user directory.
    pub fn new(bundled_presets: PathBuf, user_presets: Option<PathBuf>) -> Paths {
        let base = bundled_presets
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        Paths {
            bundled_textures: base.join("textures"),
            bundled_presets,
            user_presets,
        }
    }

    /// Preset roots to scan, user first (its ids win, §9.3).
    pub fn preset_roots(&self) -> Vec<PathBuf> {
        let mut roots: Vec<PathBuf> = self.user_presets.iter().cloned().collect();
        roots.push(self.bundled_presets.clone());
        roots
    }

    /// Texture roots for user textures and `randNN` picks.
    pub fn texture_roots(&self) -> Vec<PathBuf> {
        vec![self.bundled_textures.clone()]
    }
}

/// `<home>\Documents\Milksplat\Presets`.
pub fn user_presets_in(home: &Path) -> PathBuf {
    home.join("Documents").join("Milksplat").join("Presets")
}

fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .unwrap_or_default()
}
