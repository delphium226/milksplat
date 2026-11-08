//! The Milksplat app (spec §9): the window and event loop on the main
//! thread ([`app`]), the render thread ([`render_thread`]), the hotkeys,
//! the configuration and the file locations. `main.rs` only calls
//! [`app::main`], so integration tests can use everything here.

pub mod app;
pub mod config;
pub mod hotkeys;
pub mod paths;
pub mod render_thread;
