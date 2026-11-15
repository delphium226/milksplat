//! Running as a Windows screensaver (`milksplat.scr`), and the console.
//!
//! A `.scr` is the exe renamed. Windows starts it with:
//!
//! | Argument | When | What Milksplat does |
//! |---|---|---|
//! | `/s` | the screensaver starts | fullscreen, no cursor; any key, click, wheel or real mouse movement quits |
//! | `/p <hwnd>` | the preview in Screen Saver Settings | exits (the preview stays blank) |
//! | `/c` or `/c:<hwnd>` | the Settings… button | a message box: there are no settings |
//! | nothing (as `.scr`) | double-clicked or "Configure" | the same message box |
//!
//! Windows may write the switches upper case or with `-`, and may append
//! `:<hwnd>` or pass the handle as the next argument.
//!
//! Release builds use the Windows GUI subsystem (no console window), so on
//! startup the app attaches to the console of the terminal that launched it,
//! if any, to keep `--help`, argument errors and `--log-fps` visible there.

/// How Windows asked the screensaver to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaverMode {
    /// `/s`: run fullscreen until the user moves.
    Run,
    /// `/p <hwnd>`: draw the Settings preview (not supported: exit).
    Preview,
    /// `/c`: show the settings.
    Configure,
}

/// The screensaver mode for these arguments (without the program name), or
/// `None` for a normal run. `exe_is_scr` says whether the program was
/// started as a `.scr`, where no arguments means [`SaverMode::Configure`].
pub fn saver_mode(args: &[String], exe_is_scr: bool) -> Option<SaverMode> {
    let Some(first) = args.first() else {
        return exe_is_scr.then_some(SaverMode::Configure);
    };
    let rest = first
        .strip_prefix('/')
        .or_else(|| first.strip_prefix('-').filter(|r| !r.starts_with('-')))?;
    let mut chars = rest.chars();
    let mode = match chars.next()?.to_ascii_lowercase() {
        's' => SaverMode::Run,
        'p' => SaverMode::Preview,
        'c' => SaverMode::Configure,
        _ => return None,
    };
    let tail = chars.as_str();
    let tail_ok = tail.is_empty()
        || tail
            .strip_prefix(':')
            .unwrap_or(tail)
            .chars()
            .all(|c| c.is_ascii_digit());
    tail_ok.then_some(mode)
}

/// Whether the running program's file name ends in `.scr`.
pub fn exe_is_scr() -> bool {
    std::env::current_exe()
        .is_ok_and(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("scr")))
}

/// Mouse movement, in physical pixels from where the cursor was first seen,
/// that ends the screensaver (a nudged desk or a jittery mouse does not).
pub const MOVE_THRESHOLD: f64 = 12.0;

/// Tracks the cursor in screensaver mode.
#[derive(Debug, Default)]
pub struct MoveGuard {
    origin: Option<(f64, f64)>,
}

impl MoveGuard {
    /// Records a cursor position; true once it has moved further than
    /// [`MOVE_THRESHOLD`] from the first position seen.
    pub fn moved(&mut self, x: f64, y: f64) -> bool {
        match self.origin {
            None => {
                self.origin = Some((x, y));
                false
            }
            Some((ox, oy)) => (x - ox).hypot(y - oy) > MOVE_THRESHOLD,
        }
    }
}

/// Attaches to the launching terminal's console, if there is one, so
/// output still appears there from the GUI-subsystem release build.
/// Harmless when there is no such console or the program already has one.
#[cfg(windows)]
pub fn attach_parent_console() {
    use windows::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
    // SAFETY: AttachConsole has no preconditions; failure is fine.
    let _ = unsafe { AttachConsole(ATTACH_PARENT_PROCESS) };
}

#[cfg(not(windows))]
pub fn attach_parent_console() {}

/// The `/c` message box.
#[cfg(windows)]
pub fn show_settings_box() {
    use windows::core::w;
    use windows::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONINFORMATION, MB_OK};
    // SAFETY: both strings are static, NUL-terminated wide literals.
    unsafe {
        MessageBoxW(
            None,
            w!("The Milksplat screensaver has no settings of its own.\n\n\
                It plays the same presets as milksplat.exe and uses its \
                settings, including each preset's reactivity. To change them, \
                run milksplat.exe.\n\n\
                It reacts to whatever audio is playing."),
            w!("Milksplat screensaver"),
            MB_ICONINFORMATION | MB_OK,
        );
    }
}

#[cfg(not(windows))]
pub fn show_settings_box() {}
