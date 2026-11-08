//! Where everything lives (spec §9.3):
//!
//! | What | Where |
//! |---|---|
//! | Config | `%APPDATA%\Milksplat\config.toml` |
//! | Logs, crash reports, the `running` marker | `%LOCALAPPDATA%\Milksplat\logs\` |
//! | User presets | `%USERPROFILE%\Documents\Milksplat\Presets\` (created on first run) |
//! | Bundled presets, textures | `presets\`, `textures\` next to the exe (or `--presets`) |
//!
//! The known folders come from `SHGetKnownFolderPath`, falling back to the
//! environment variables, then to the exe's directory. Everything is built
//! from a [`KnownDirs`], so tests use temp directories.

use std::path::{Path, PathBuf};

/// The system folders the paths are built from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KnownDirs {
    /// `%APPDATA%` (FOLDERID_RoamingAppData).
    pub roaming_app_data: Option<PathBuf>,
    /// `%LOCALAPPDATA%` (FOLDERID_LocalAppData).
    pub local_app_data: Option<PathBuf>,
    /// The user's Documents folder (FOLDERID_Documents).
    pub documents: Option<PathBuf>,
    /// The directory of the executable.
    pub exe_dir: PathBuf,
}

impl KnownDirs {
    /// This user's folders. Never panics; a folder that cannot be found is
    /// `None`.
    pub fn system() -> KnownDirs {
        let env = |name: &str| {
            std::env::var_os(name)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        };
        KnownDirs {
            roaming_app_data: known::roaming_app_data().or_else(|| env("APPDATA")),
            local_app_data: known::local_app_data().or_else(|| env("LOCALAPPDATA")),
            documents: known::documents()
                .or_else(|| env("USERPROFILE").map(|home| home.join("Documents"))),
            exe_dir: exe_dir(),
        }
    }
}

/// Every file and directory the app uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    /// The config file.
    pub config: PathBuf,
    /// The log directory (daily files, 7 kept), also holding the `running`
    /// marker.
    pub logs: PathBuf,
    /// Where crash reports go (next to the logs).
    pub crash_dir: PathBuf,
    /// The user preset directory, if there is one.
    pub user_presets: Option<PathBuf>,
    /// `presets\` next to the exe, or the `--presets` override.
    pub bundled_presets: PathBuf,
    /// `textures\` beside the bundled preset directory.
    pub bundled_textures: PathBuf,
}

impl Paths {
    /// The paths for this user (see the module docs).
    pub fn discover(presets_override: Option<PathBuf>) -> Paths {
        Paths::from_dirs(&KnownDirs::system(), presets_override)
    }

    /// The paths under `dirs`. A missing roaming or local folder falls back
    /// to the other, then to the exe's directory; without a Documents
    /// folder there is no user preset directory.
    pub fn from_dirs(dirs: &KnownDirs, presets_override: Option<PathBuf>) -> Paths {
        let roaming = dirs
            .roaming_app_data
            .as_ref()
            .or(dirs.local_app_data.as_ref())
            .unwrap_or(&dirs.exe_dir)
            .join("Milksplat");
        let local = dirs
            .local_app_data
            .as_ref()
            .or(dirs.roaming_app_data.as_ref())
            .unwrap_or(&dirs.exe_dir)
            .join("Milksplat");
        let bundled_presets = presets_override.unwrap_or_else(|| dirs.exe_dir.join("presets"));
        let base = bundled_presets
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let logs = local.join("logs");
        Paths {
            config: roaming.join("config.toml"),
            crash_dir: logs.clone(),
            logs,
            user_presets: dirs.documents.as_deref().map(user_presets_in),
            bundled_textures: base.join("textures"),
            bundled_presets,
        }
    }

    /// Creates the config, log and crash directories and the user preset
    /// directory. A user preset directory that cannot be created is
    /// dropped. Returns warnings for the log (logging may not be up yet).
    pub fn create_dirs(&mut self) -> Vec<String> {
        let mut warnings = Vec::new();
        let config_dir = self.config.parent().map(Path::to_path_buf);
        for dir in config_dir.iter().chain([&self.logs, &self.crash_dir]) {
            if let Err(e) = std::fs::create_dir_all(dir) {
                warnings.push(format!("cannot create {}: {e}", dir.display()));
            }
        }
        if let Some(dir) = &self.user_presets {
            if let Err(e) = std::fs::create_dir_all(dir) {
                warnings.push(format!(
                    "cannot create the user preset directory {}: {e}",
                    dir.display()
                ));
                self.user_presets = None;
            }
        }
        warnings
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

/// `<documents>\Milksplat\Presets`.
pub fn user_presets_in(documents: &Path) -> PathBuf {
    documents.join("Milksplat").join("Presets")
}

fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .unwrap_or_default()
}

#[cfg(windows)]
mod known {
    use std::path::PathBuf;

    use windows::core::GUID;
    use windows::Win32::System::Com::CoTaskMemFree;
    use windows::Win32::UI::Shell::{
        FOLDERID_Documents, FOLDERID_LocalAppData, FOLDERID_RoamingAppData, SHGetKnownFolderPath,
        KF_FLAG_DEFAULT,
    };

    fn folder(id: &GUID) -> Option<PathBuf> {
        // SAFETY: `id` is a valid known-folder id; on success the returned
        // string is owned by us and freed with CoTaskMemFree below.
        let p = unsafe { SHGetKnownFolderPath(id, KF_FLAG_DEFAULT, None) }.ok()?;
        // SAFETY: `p` is a valid NUL-terminated wide string from the call.
        let path = unsafe { p.to_string() }.ok().map(PathBuf::from);
        // SAFETY: allocated by SHGetKnownFolderPath with the COM allocator.
        unsafe { CoTaskMemFree(Some(p.0 as *const _)) };
        path.filter(|p| !p.as_os_str().is_empty())
    }

    pub fn roaming_app_data() -> Option<PathBuf> {
        folder(&FOLDERID_RoamingAppData)
    }

    pub fn local_app_data() -> Option<PathBuf> {
        folder(&FOLDERID_LocalAppData)
    }

    pub fn documents() -> Option<PathBuf> {
        folder(&FOLDERID_Documents)
    }
}

#[cfg(not(windows))]
mod known {
    use std::path::PathBuf;

    pub fn roaming_app_data() -> Option<PathBuf> {
        None
    }

    pub fn local_app_data() -> Option<PathBuf> {
        None
    }

    pub fn documents() -> Option<PathBuf> {
        None
    }
}
