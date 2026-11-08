//! Keeping the display awake (spec §9.1): while the window is fullscreen
//! and audio is playing (less than [`SILENCE_S`] of silence), the display
//! must not sleep or blank.
//!
//! `SetThreadExecutionState` is per thread: the request lasts while the
//! calling thread lives, or until that thread clears it. All calls are made
//! from the **main thread** (it owns the window and lives for the whole
//! run), and only when the wanted state changes.

/// Seconds of continuous silence after which the display may sleep again
/// and the no-audio hint shows (§8.2).
pub const SILENCE_S: f64 = 10.0;

/// Whether the display should be kept awake.
pub fn wanted(fullscreen: bool, silent_for: f64) -> bool {
    fullscreen && silent_for < SILENCE_S
}

/// Asks Windows to keep the display on (`ES_CONTINUOUS |
/// ES_DISPLAY_REQUIRED`), or clears the request (`ES_CONTINUOUS`), for the
/// calling thread.
#[cfg(windows)]
pub fn set(on: bool) {
    use windows::Win32::System::Power::{
        SetThreadExecutionState, ES_CONTINUOUS, ES_DISPLAY_REQUIRED,
    };
    let flags = if on {
        ES_CONTINUOUS | ES_DISPLAY_REQUIRED
    } else {
        ES_CONTINUOUS
    };
    // SAFETY: a plain Win32 call with valid flags.
    let prev = unsafe { SetThreadExecutionState(flags) };
    if prev.0 == 0 {
        tracing::warn!("SetThreadExecutionState failed");
    } else {
        tracing::debug!("keep display awake: {on}");
    }
}

#[cfg(not(windows))]
pub fn set(_on: bool) {}

/// The current request, so the OS is only called on a change. Dropping it
/// clears the request (call and drop it on the main thread).
#[derive(Debug, Default)]
pub struct KeepAwake {
    on: bool,
}

impl KeepAwake {
    /// Makes the request match `on` (a no-op if it already does).
    pub fn update(&mut self, on: bool) {
        if on != self.on {
            self.on = on;
            set(on);
        }
    }
}

impl Drop for KeepAwake {
    fn drop(&mut self) {
        self.update(false);
    }
}
