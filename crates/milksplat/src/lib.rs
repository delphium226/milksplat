//! The Milksplat app (spec §9): the window and event loop on the main
//! thread ([`app`]), the render thread ([`render_thread`]), the hotkeys,
//! the configuration, the file locations, logging, crash handling,
//! keep-awake and preset hot reload. `main.rs` only calls [`app::main`], so
//! integration tests can use everything here.

pub mod app;
pub mod config;
pub mod crash;
pub mod hotkeys;
pub mod hotreload;
pub mod keepawake;
pub mod logging;
pub mod paths;
pub mod render_thread;
