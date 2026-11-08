//! The app configuration (spec §9.3): `%APPDATA%\Milksplat\config.toml`.
//!
//! **Loading** ([`load_config`]) never fails: a missing file gives the
//! defaults (and writes them out); an unreadable or invalid file is renamed
//! to `config.toml.bad` and the defaults are used; inside a valid file each
//! key is read on its own, so one bad value keeps the default for that key
//! only. Every problem comes back as a warning for the log.
//!
//! **Saving** ([`save_config`]) writes `config.toml.tmp` and renames it over
//! the config, so a reader never sees a half-written file. While the app
//! runs, saves go through a [`ConfigStore`]: a background thread that
//! writes at most once a second and always once more when it is finished,
//! so no frame ever waits on the disk.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{RecvTimeoutError, Sender};
use serde::{Deserialize, Serialize};
use toml::{Table, Value};

/// The A/V offset range, in milliseconds (§8.4).
pub const AV_OFFSET_MIN_MS: i32 = -500;
pub const AV_OFFSET_MAX_MS: i32 = 2000;
/// The A/V offset for a Bluetooth output with no stored value (§8.4).
pub const BLUETOOTH_AV_OFFSET_MS: i32 = 200;
/// The smallest and largest window side restored from the config.
const WINDOW_MIN: [u32; 2] = [160, 120];
const WINDOW_MAX: u32 = 16_384;
/// The saver writes at most this often.
pub const SAVE_INTERVAL: Duration = Duration::from_secs(1);
/// How long finishing the saver waits for its last write.
const SAVER_JOIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Render quality: the internal resolution scale and the mesh-size cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    /// Full mesh; the governor picks the scale (§6.6).
    #[default]
    Auto,
    High,
    Medium,
    Low,
}

impl Quality {
    /// The next setting for the `Q` hotkey: Auto, High, Medium, Low, Auto...
    pub fn next(self) -> Quality {
        match self {
            Quality::Auto => Quality::High,
            Quality::High => Quality::Medium,
            Quality::Medium => Quality::Low,
            Quality::Low => Quality::Auto,
        }
    }

    /// The fixed internal resolution scale, or `None` for the governor's.
    pub fn fixed_scale(self) -> Option<f32> {
        match self {
            Quality::Auto => None,
            Quality::High => Some(1.0),
            Quality::Medium => Some(0.75),
            Quality::Low => Some(0.5),
        }
    }

    /// The warp-mesh cap (columns, rows); 192x144 at most (§6.6).
    pub fn mesh_cap(self) -> [u32; 2] {
        match self {
            Quality::Auto | Quality::High => [192, 144],
            Quality::Medium => [128, 96],
            Quality::Low => [64, 48],
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Quality::Auto => "Auto",
            Quality::High => "High",
            Quality::Medium => "Medium",
            Quality::Low => "Low",
        }
    }
}

/// The window's size, position and monitor, restored at startup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowState {
    /// Inner size in logical pixels.
    pub size: [u32; 2],
    /// Outer position in physical pixels; `None` lets the OS place it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<[i32; 2]>,
    /// The monitor the window was last on, by name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitor: Option<String>,
}

impl Default for WindowState {
    fn default() -> Self {
        WindowState {
            size: [1280, 720],
            position: None,
            monitor: None,
        }
    }
}

/// Everything the user can configure. Serialised for saving; loading
/// reads each key on its own (see [`load_config`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// The output device to capture, by name; `None` is the system default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_device: Option<String>,
    pub auto_advance_s: f64,
    pub blend_s: f64,
    pub hard_cuts: bool,
    pub hard_cut_strength: f32,
    pub quality: Quality,
    pub favourites: BTreeSet<String>,
    pub bans: BTreeSet<String>,
    /// Presets blamed for a GPU device loss (§6.7).
    pub blocklist: BTreeSet<String>,
    /// The monitor to go fullscreen on, by name; `None` is the window's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fullscreen_monitor: Option<String>,
    /// A/V offset per device name, in milliseconds.
    pub av_offsets_ms: BTreeMap<String, i32>,
    pub window: WindowState,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            audio_device: None,
            av_offsets_ms: BTreeMap::new(),
            auto_advance_s: 20.0,
            blend_s: 2.7,
            hard_cuts: true,
            hard_cut_strength: 2.0,
            quality: Quality::Auto,
            favourites: BTreeSet::new(),
            bans: BTreeSet::new(),
            blocklist: BTreeSet::new(),
            window: WindowState::default(),
            fullscreen_monitor: None,
        }
    }
}

impl Config {
    /// The A/V offset for `device`: the stored value, else 200 ms for a
    /// Bluetooth output and 0 for any other (§8.4).
    pub fn av_offset_ms(&self, device: &str, bluetooth: bool) -> i32 {
        match self.av_offsets_ms.get(device) {
            Some(&ms) => ms.clamp(AV_OFFSET_MIN_MS, AV_OFFSET_MAX_MS),
            None if bluetooth => BLUETOOTH_AV_OFFSET_MS,
            None => 0,
        }
    }
}

/// `<path><suffix>`, e.g. `config.toml.bad`.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path
        .file_name()
        .map_or_else(|| OsString::from("config.toml"), OsString::from);
    name.push(suffix);
    path.with_file_name(name)
}

/// Where a bad config is kept: `config.toml.bad`.
pub fn bad_path(path: &Path) -> PathBuf {
    with_suffix(path, ".bad")
}

/// Where a save is written before the rename: `config.toml.tmp`.
pub fn tmp_path(path: &Path) -> PathBuf {
    with_suffix(path, ".tmp")
}

/// Loads the config at `path` (see the module docs). Never fails; the
/// warnings say what was wrong and what was used instead.
pub fn load_config(path: &Path) -> (Config, Vec<String>) {
    let mut warnings = Vec::new();
    let shown = path.display();
    let text = match std::fs::read(path) {
        Ok(bytes) => String::from_utf8(bytes).map_err(|_| "it is not UTF-8 text".to_string()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let cfg = Config::default();
            if let Err(e) = save_config(path, &cfg) {
                warnings.push(format!("cannot write the default config to {shown}: {e}"));
            }
            return (cfg, warnings);
        }
        Err(e) => Err(format!("cannot read it ({e})")),
    };
    let table = text.and_then(|t| t.parse::<Table>().map_err(|e| e.to_string()));
    match table {
        Ok(table) => {
            let cfg = from_table(&table, &mut warnings);
            (cfg, warnings)
        }
        Err(why) => {
            let bad = bad_path(path);
            warnings.push(format!("the config {shown} is invalid: {why}"));
            match std::fs::rename(path, &bad) {
                Ok(()) => warnings.push(format!(
                    "using the defaults; the bad file was renamed to {}",
                    bad.display()
                )),
                Err(e) => {
                    warnings.push(format!(
                        "using the defaults; cannot rename the bad file: {e}"
                    ));
                    return (Config::default(), warnings);
                }
            }
            let cfg = Config::default();
            if let Err(e) = save_config(path, &cfg) {
                warnings.push(format!("cannot write the default config to {shown}: {e}"));
            }
            (cfg, warnings)
        }
    }
}

/// Reads every known key of `t` on its own; a bad value keeps the default.
fn from_table(t: &Table, warnings: &mut Vec<String>) -> Config {
    let mut c = Config::default();
    let mut r = Reader { warnings };
    for (key, value) in t {
        match key.as_str() {
            "audio_device" => r.set(key, value, &mut c.audio_device, |s: String| {
                Ok(Some(s).filter(|s| !s.is_empty()))
            }),
            "auto_advance_s" => r.set(key, value, &mut c.auto_advance_s, non_negative),
            "blend_s" => r.set(key, value, &mut c.blend_s, non_negative),
            "hard_cuts" => r.set(key, value, &mut c.hard_cuts, Ok),
            "hard_cut_strength" => r.set(key, value, &mut c.hard_cut_strength, |v: f32| {
                non_negative(f64::from(v)).map(|_| v)
            }),
            "quality" => r.set(key, value, &mut c.quality, Ok),
            "favourites" => r.set(key, value, &mut c.favourites, Ok),
            "bans" => r.set(key, value, &mut c.bans, Ok),
            "blocklist" => r.set(key, value, &mut c.blocklist, Ok),
            "fullscreen_monitor" => r.set(key, value, &mut c.fullscreen_monitor, |s: String| {
                Ok(Some(s).filter(|s| !s.is_empty()))
            }),
            "av_offsets_ms" => c.av_offsets_ms = r.av_offsets(value),
            "window" => c.window = r.window(value),
            _ => r.warn(format!("config: unknown key {key:?} ignored")),
        }
    }
    c
}

fn non_negative(v: f64) -> Result<f64, String> {
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err(format!("{v} is not a finite number >= 0"))
    }
}

/// Reads typed values out of TOML, collecting warnings.
struct Reader<'a> {
    warnings: &'a mut Vec<String>,
}

impl Reader<'_> {
    fn warn(&mut self, w: String) {
        self.warnings.push(w);
    }

    /// Sets `out` to `value` as a `T`, checked by `check`; on any error
    /// warns and leaves `out` (the default) alone.
    fn set<T, U>(
        &mut self,
        key: &str,
        value: &Value,
        out: &mut U,
        check: impl FnOnce(T) -> Result<U, String>,
    ) where
        T: serde::de::DeserializeOwned,
        U: std::fmt::Debug,
    {
        match value.clone().try_into::<T>() {
            Ok(v) => match check(v) {
                Ok(v) => *out = v,
                Err(e) => self.warn(format!("config: {key}: {e}; using the default {out:?}")),
            },
            Err(e) => self.warn(format!(
                "config: {key}: {}; using the default {out:?}",
                e.to_string().trim()
            )),
        }
    }

    fn av_offsets(&mut self, value: &Value) -> BTreeMap<String, i32> {
        let mut out = BTreeMap::new();
        let Some(t) = value.as_table() else {
            self.warn("config: av_offsets_ms is not a table; ignored".into());
            return out;
        };
        for (device, v) in t {
            match v.as_integer() {
                Some(ms) => {
                    let clamped =
                        ms.clamp(i64::from(AV_OFFSET_MIN_MS), i64::from(AV_OFFSET_MAX_MS));
                    if clamped != ms {
                        self.warn(format!(
                            "config: av_offsets_ms.{device:?}: {ms} is outside \
                             {AV_OFFSET_MIN_MS}..={AV_OFFSET_MAX_MS}; using {clamped}"
                        ));
                    }
                    // In range after the clamp.
                    out.insert(device.clone(), clamped as i32);
                }
                None => self.warn(format!(
                    "config: av_offsets_ms.{device:?}: not a whole number of milliseconds; ignored"
                )),
            }
        }
        out
    }

    fn window(&mut self, value: &Value) -> WindowState {
        let mut w = WindowState::default();
        let Some(t) = value.as_table() else {
            self.warn("config: window is not a table; using the default".into());
            return w;
        };
        for (key, v) in t {
            let name = format!("window.{key}");
            match key.as_str() {
                "size" => self.set(&name, v, &mut w.size, |s: [u32; 2]| {
                    if s[0] >= WINDOW_MIN[0]
                        && s[1] >= WINDOW_MIN[1]
                        && s[0] <= WINDOW_MAX
                        && s[1] <= WINDOW_MAX
                    {
                        Ok(s)
                    } else {
                        Err(format!("{s:?} is not a usable window size"))
                    }
                }),
                "position" => self.set(&name, v, &mut w.position, |p: [i32; 2]| Ok(Some(p))),
                "monitor" => self.set(&name, v, &mut w.monitor, |s: String| {
                    Ok(Some(s).filter(|s| !s.is_empty()))
                }),
                _ => self.warn(format!("config: unknown key {name:?} ignored")),
            }
        }
        w
    }
}

/// Saves `c` to `path` atomically: writes and flushes `config.toml.tmp`,
/// then renames it over `path` (creating the directory if needed).
///
/// `std::fs::rename` replaces an existing target on Windows (it uses
/// `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING`, or a POSIX-semantics
/// rename), which NTFS performs atomically. Should the rename still fail
/// while the target exists (e.g. a filter driver refuses the replace), the
/// target is removed and the rename tried once more; a crash between those
/// two steps leaves only the complete `.tmp` file, never a truncated config.
pub fn save_config(path: &Path, c: &Config) -> std::io::Result<()> {
    let text = toml::to_string(c).map_err(std::io::Error::other)?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = tmp_path(path);
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(b"# Milksplat settings. Unknown keys are ignored; a bad value falls back\n# to its default (see the log).\n\n")?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
    }
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(first) if path.exists() => {
            tracing::debug!(
                "replacing {} by rename failed ({first}); removing it first",
                path.display()
            );
            std::fs::remove_file(path)?;
            std::fs::rename(&tmp, path)
        }
        Err(e) => Err(e),
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

struct StoreInner {
    path: PathBuf,
    cfg: Mutex<Config>,
    dirty: AtomicBool,
    saves: AtomicU64,
}

impl StoreInner {
    /// Writes the config if it changed since the last write. The lock is
    /// held only to copy it, never during the write.
    fn save_if_dirty(&self) {
        if !self.dirty.swap(false, Ordering::SeqCst) {
            return;
        }
        let snapshot = lock(&self.cfg).clone();
        match save_config(&self.path, &snapshot) {
            Ok(()) => tracing::debug!("config saved to {}", self.path.display()),
            Err(e) => tracing::warn!("cannot save the config to {}: {e}", self.path.display()),
        }
        self.saves.fetch_add(1, Ordering::Relaxed);
    }
}

enum SaverMsg {
    Changed,
    Finish,
}

/// The live config, shared by the main and render threads. Changes are
/// written by the saver thread (`ms-config`), at most once per
/// [`SAVE_INTERVAL`]; [`update`](Self::update) never touches the disk.
#[derive(Clone)]
pub struct ConfigStore {
    inner: Arc<StoreInner>,
    tx: Sender<SaverMsg>,
}

/// The saver thread; [`finish`](Self::finish) writes any last change.
pub struct ConfigSaver {
    tx: Sender<SaverMsg>,
    thread: Option<JoinHandle<()>>,
    inner: Arc<StoreInner>,
}

impl ConfigStore {
    /// Starts the saver for the config file at `path`, holding `cfg`.
    pub fn start(path: PathBuf, cfg: Config) -> (ConfigStore, ConfigSaver) {
        let inner = Arc::new(StoreInner {
            path,
            cfg: Mutex::new(cfg),
            dirty: AtomicBool::new(false),
            saves: AtomicU64::new(0),
        });
        let (tx, rx) = crossbeam_channel::unbounded();
        let worker = Arc::clone(&inner);
        let thread = std::thread::Builder::new()
            .name("ms-config".into())
            .spawn(move || {
                let mut last_save: Option<Instant> = None;
                // Until `Finish`, or every store is gone.
                while let Ok(SaverMsg::Changed) = rx.recv() {
                    // Debounce: wait out the interval since the last write,
                    // collecting more changes.
                    if let Some(due) = last_save.map(|t| t + SAVE_INTERVAL) {
                        let mut finish = false;
                        loop {
                            let left = due.saturating_duration_since(Instant::now());
                            if left.is_zero() {
                                break;
                            }
                            match rx.recv_timeout(left) {
                                Ok(SaverMsg::Changed) => {}
                                Err(RecvTimeoutError::Timeout) => break,
                                Ok(SaverMsg::Finish) | Err(RecvTimeoutError::Disconnected) => {
                                    finish = true;
                                    break;
                                }
                            }
                        }
                        if finish {
                            break;
                        }
                    }
                    worker.save_if_dirty();
                    last_save = Some(Instant::now());
                }
                worker.save_if_dirty();
            });
        let thread = match thread {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::error!("cannot start the config saver thread ({e}); saving on exit only");
                None
            }
        };
        let store = ConfigStore {
            inner: Arc::clone(&inner),
            tx: tx.clone(),
        };
        (store, ConfigSaver { tx, thread, inner })
    }

    /// Changes the config and schedules a save.
    pub fn update(&self, f: impl FnOnce(&mut Config)) {
        f(&mut lock(&self.inner.cfg));
        self.inner.dirty.store(true, Ordering::SeqCst);
        let _ = self.tx.send(SaverMsg::Changed);
    }

    /// A copy of the current config.
    pub fn get(&self) -> Config {
        lock(&self.inner.cfg).clone()
    }

    /// How many times the config was written (for tests).
    #[doc(hidden)]
    pub fn saves(&self) -> u64 {
        self.inner.saves.load(Ordering::Relaxed)
    }
}

impl ConfigSaver {
    /// Writes any change not yet saved and stops the thread (waiting at
    /// most 2 s for it).
    pub fn finish(mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        let _ = self.tx.send(SaverMsg::Finish);
        match self.thread.take() {
            Some(t) => {
                let start = Instant::now();
                while !t.is_finished() && start.elapsed() < SAVER_JOIN_TIMEOUT {
                    std::thread::sleep(Duration::from_millis(2));
                }
                if t.is_finished() {
                    let _ = t.join();
                } else {
                    tracing::warn!("the config saver did not finish within 2 s");
                }
            }
            // No thread: save here.
            None => self.inner.save_if_dirty(),
        }
    }
}

impl Drop for ConfigSaver {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.stop();
        }
    }
}
