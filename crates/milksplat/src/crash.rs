//! Crash handling (spec §9.4).
//!
//! **Panic hook.** Some panics are caught on purpose (naga inside
//! ms-render, the loader's per-request guard, the GPU timer), and a panic
//! hook runs before `catch_unwind` sees the panic. So the hook only
//! *records* the panic (message, location, thread, backtrace) and logs it
//! at error level, then calls the previous hook; it never writes a report.
//!
//! **Crash report.** Written by [`report_escaped_panic`] only where a panic
//! really escapes: at the render thread's boundary and around the main
//! thread's event loop. `crash-<UTC time>.txt` next to the logs holds the
//! recorded panic, the GPU adapter, the preset on screen, the app version,
//! and the last 200 log lines. The app then shows a message box ("Milksplat
//! crashed. A report was saved to <path>.") and exits with code 1.
//!
//! **Run marker.** [`RunMarker`] keeps a `running` file holding the preset
//! on screen while the app runs, and deletes it on a clean exit. A marker
//! found at startup means the last run crashed while that preset showed.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, TryLockError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};

use crossbeam_channel::{RecvTimeoutError, Sender};

use crate::paths::Paths;

/// The marker file's name, in the logs directory.
pub const MARKER_FILE: &str = "running";
/// The marker is rewritten at most this often (shorter than the config's
/// second: a crash right after a switch should still name the new preset).
pub const MARKER_INTERVAL: Duration = Duration::from_millis(250);
/// Panics remembered for the report (the latest of the reporting thread
/// is used).
const RECENT_PANICS: usize = 8;
/// How long dropping or ending the marker waits for its writer.
const MARKER_JOIN_TIMEOUT: Duration = Duration::from_secs(2);

/// A panic, as the hook saw it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanicRecord {
    pub message: String,
    /// `file:line:column`.
    pub location: Option<String>,
    pub thread: String,
    pub backtrace: String,
}

/// Everything a crash report holds.
#[derive(Debug, Clone)]
pub struct CrashInfo {
    /// What escaped, e.g. "the render thread panicked".
    pub reason: String,
    pub panic: Option<PanicRecord>,
    /// GPU adapter name, backend and driver.
    pub adapter: Option<String>,
    /// The preset on screen.
    pub preset: Option<String>,
    pub version: &'static str,
    pub time: SystemTime,
    pub log_lines: Vec<String>,
}

/// What the report needs that the hook and the render thread provide.
#[derive(Default)]
struct State {
    crash_dir: Mutex<Option<PathBuf>>,
    preset: Mutex<Option<Arc<Mutex<Option<String>>>>>,
    adapter: Mutex<Option<String>>,
    panics: Mutex<Vec<PanicRecord>>,
}

fn state() -> &'static State {
    static STATE: OnceLock<State> = OnceLock::new();
    STATE.get_or_init(State::default)
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Installs the panic hook (once per process; later calls only update the
/// crash directory and the preset slot). `current_preset` is the preset on
/// screen, kept up to date by the render thread.
pub fn install_panic_hook(paths: &Paths, current_preset: Arc<Mutex<Option<String>>>) {
    let s = state();
    *lock(&s.crash_dir) = Some(paths.crash_dir.clone());
    *lock(&s.preset) = Some(current_preset);
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let record = record_of(info);
            tracing::error!(
                "panic on thread {}: {} (at {})",
                record.thread,
                record.message,
                record.location.as_deref().unwrap_or("an unknown location")
            );
            // Never block in a panic: if the slot is busy, skip recording.
            let slot = match state().panics.try_lock() {
                Ok(g) => Some(g),
                Err(TryLockError::Poisoned(p)) => Some(p.into_inner()),
                Err(TryLockError::WouldBlock) => None,
            };
            if let Some(mut panics) = slot {
                if panics.len() >= RECENT_PANICS {
                    panics.remove(0);
                }
                panics.push(record);
            }
            prev(info);
        }));
    });
}

fn record_of(info: &std::panic::PanicHookInfo<'_>) -> PanicRecord {
    let payload = info.payload();
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with a non-text payload".to_string());
    let thread = std::thread::current()
        .name()
        .map_or_else(|| "unnamed".to_string(), str::to_string);
    PanicRecord {
        message,
        location: info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())),
        thread,
        backtrace: std::backtrace::Backtrace::force_capture().to_string(),
    }
}

/// The latest panic the hook recorded, on any thread.
pub fn last_panic() -> Option<PanicRecord> {
    lock(&state().panics).last().cloned()
}

/// The latest recorded panic of the calling thread, else the latest one.
fn panic_for_this_thread() -> Option<PanicRecord> {
    let name = std::thread::current().name().map(str::to_string);
    let panics = lock(&state().panics);
    panics
        .iter()
        .rev()
        .find(|p| Some(&p.thread) == name.as_ref())
        .or(panics.last())
        .cloned()
}

/// Records the GPU adapter for crash reports (the render thread calls this
/// when it opens a device).
pub fn set_adapter_info(info: String) {
    *lock(&state().adapter) = Some(info);
}

/// Writes a crash report for a panic that escaped (`reason` says where)
/// and, if `show_box`, shows the crash message box. Returns the report's
/// path, or `None` if it could not be written (that is logged).
pub fn report_escaped_panic(reason: &str, show_box: bool) -> Option<PathBuf> {
    let s = state();
    let dir = lock(&s.crash_dir)
        .clone()
        .unwrap_or_else(std::env::temp_dir);
    let preset = lock(&s.preset).as_ref().and_then(|slot| lock(slot).clone());
    let info = CrashInfo {
        reason: reason.to_string(),
        panic: panic_for_this_thread(),
        adapter: lock(&s.adapter).clone(),
        preset,
        version: env!("CARGO_PKG_VERSION"),
        time: SystemTime::now(),
        log_lines: crate::logging::ring().lines(),
    };
    let path = match write_report(&dir, &info) {
        Ok(path) => {
            tracing::error!("crash report written to {}", path.display());
            Some(path)
        }
        Err(e) => {
            tracing::error!("cannot write a crash report in {}: {e}", dir.display());
            None
        }
    };
    if show_box {
        show_crash_box(path.as_deref());
    }
    path
}

/// Writes `crash-<YYYYMMDD-HHMMSS>.txt` (UTC; `-2`, `-3`... if taken) in
/// `dir` and returns its path.
pub fn write_report(dir: &Path, info: &CrashInfo) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let t = Utc::of(info.time);
    let stem = format!(
        "crash-{:04}{:02}{:02}-{:02}{:02}{:02}",
        t.year, t.month, t.day, t.hour, t.minute, t.second
    );
    let mut text = format!(
        "Milksplat {} crashed\nTime: {:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC\nReason: {}\n\n",
        info.version, t.year, t.month, t.day, t.hour, t.minute, t.second, info.reason
    );
    match &info.panic {
        Some(p) => text.push_str(&format!(
            "Panic: {}\nLocation: {}\nThread: {}\n\n",
            p.message,
            p.location.as_deref().unwrap_or("unknown"),
            p.thread
        )),
        None => text.push_str("Panic: not recorded\n\n"),
    }
    text.push_str(&format!(
        "GPU adapter: {}\nPreset: {}\n\n",
        info.adapter.as_deref().unwrap_or("unknown"),
        info.preset.as_deref().unwrap_or("none")
    ));
    if let Some(p) = &info.panic {
        text.push_str("Backtrace:\n");
        text.push_str(&p.backtrace);
        if !p.backtrace.ends_with('\n') {
            text.push('\n');
        }
        text.push('\n');
    }
    text.push_str(&format!("Last {} log lines:\n", info.log_lines.len()));
    for line in &info.log_lines {
        text.push_str(line);
        text.push('\n');
    }

    for n in 1..1000 {
        let name = if n == 1 {
            format!("{stem}.txt")
        } else {
            format!("{stem}-{n}.txt")
        };
        let path = dir.join(name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                f.write_all(text.as_bytes())?;
                f.sync_all()?;
                return Ok(path);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::other(
        "too many crash reports with this time stamp",
    ))
}

/// Shows "Milksplat crashed. A report was saved to <path>." in a native
/// message box (an error icon); blocks until it is dismissed.
#[cfg(windows)]
pub fn show_crash_box(report: Option<&Path>) {
    use windows::core::{w, PCWSTR};
    use windows::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONERROR, MB_OK};

    let text = match report {
        Some(p) => format!("Milksplat crashed. A report was saved to {}.", p.display()),
        None => "Milksplat crashed. The crash report could not be saved.".to_string(),
    };
    let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: `wide` is NUL-terminated and outlives the call; no owner
    // window.
    unsafe {
        MessageBoxW(
            None,
            PCWSTR(wide.as_ptr()),
            w!("Milksplat"),
            MB_ICONERROR | MB_OK,
        );
    }
}

#[cfg(not(windows))]
pub fn show_crash_box(report: Option<&Path>) {
    match report {
        Some(p) => eprintln!("Milksplat crashed. A report was saved to {}.", p.display()),
        None => eprintln!("Milksplat crashed. The crash report could not be saved."),
    }
}

/// A UTC calendar time.
struct Utc {
    year: i64,
    month: u32,
    day: u32,
    hour: u64,
    minute: u64,
    second: u64,
}

impl Utc {
    fn of(t: SystemTime) -> Utc {
        let secs = t
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let days = (secs / 86_400) as i64;
        let rem = secs % 86_400;
        // Days to civil date (Howard Hinnant's algorithm).
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        let year = yoe + era * 400 + i64::from(month <= 2);
        Utc {
            year,
            month,
            day,
            hour: rem / 3600,
            minute: rem % 3600 / 60,
            second: rem % 60,
        }
    }
}

/// The `running` marker (see the module docs). Writes happen on a small
/// thread (`ms-marker`), at most every [`MARKER_INTERVAL`], so
/// [`set_preset`](Self::set_preset) never touches the disk.
pub struct RunMarker {
    path: PathBuf,
    tx: Option<Sender<String>>,
    thread: Option<JoinHandle<()>>,
}

impl RunMarker {
    /// Starts a run: returns the marker and, if the last run did not end
    /// cleanly while a preset was recorded, that preset's id.
    pub fn begin(dir: &Path) -> (RunMarker, Option<String>) {
        let path = dir.join(MARKER_FILE);
        let crashed = std::fs::read_to_string(&path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        if let Err(e) = std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, "")) {
            tracing::warn!("cannot write the run marker {}: {e}", path.display());
        }
        let (tx, rx) = crossbeam_channel::unbounded::<String>();
        let target = path.clone();
        let thread = std::thread::Builder::new()
            .name("ms-marker".into())
            .spawn(move || {
                let write = |id: &str| {
                    if let Err(e) = std::fs::write(&target, id) {
                        tracing::warn!("cannot update the run marker: {e}");
                    }
                };
                let mut last: Option<Instant> = None;
                while let Ok(mut id) = rx.recv() {
                    let mut done = false;
                    if let Some(due) = last.map(|t| t + MARKER_INTERVAL) {
                        loop {
                            let left = due.saturating_duration_since(Instant::now());
                            if left.is_zero() {
                                break;
                            }
                            match rx.recv_timeout(left) {
                                Ok(newer) => id = newer,
                                Err(RecvTimeoutError::Timeout) => break,
                                Err(RecvTimeoutError::Disconnected) => {
                                    done = true;
                                    break;
                                }
                            }
                        }
                    }
                    // Anything queued meanwhile is newer.
                    if let Some(newest) = rx.try_iter().last() {
                        id = newest;
                    }
                    write(&id);
                    last = Some(Instant::now());
                    if done {
                        break;
                    }
                }
            });
        let (tx, thread) = match thread {
            Ok(t) => (Some(tx), Some(t)),
            Err(e) => {
                tracing::warn!("cannot start the run marker thread ({e}); crash detection is off");
                (None, None)
            }
        };
        (RunMarker { path, tx, thread }, crashed)
    }

    /// Records the preset on screen.
    pub fn set_preset(&self, id: &str) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(id.to_string());
        }
    }

    /// A clean exit: deletes the marker.
    pub fn end(mut self) {
        self.stop();
        if let Err(e) = std::fs::remove_file(&self.path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!("cannot remove the run marker: {e}");
            }
        }
    }

    /// Writes the last recorded preset and stops the writer.
    fn stop(&mut self) {
        self.tx = None;
        if let Some(t) = self.thread.take() {
            let start = Instant::now();
            while !t.is_finished() && start.elapsed() < MARKER_JOIN_TIMEOUT {
                std::thread::sleep(Duration::from_millis(2));
            }
            if t.is_finished() {
                let _ = t.join();
            }
        }
    }
}

impl Drop for RunMarker {
    /// Without [`end`](Self::end) the marker stays: the run did not end
    /// cleanly.
    fn drop(&mut self) {
        self.stop();
    }
}
