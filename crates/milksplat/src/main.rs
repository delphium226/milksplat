//! Milksplat: a MilkDrop-style music visualizer.

// Release builds open no console window; the app attaches to a launching
// terminal's console instead (see `screensaver::attach_parent_console`).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() -> std::process::ExitCode {
    milksplat::app::main()
}
