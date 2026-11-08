//! The app configuration (spec §9.3). Loading and saving
//! (`%APPDATA%\Milksplat\config.toml`) come with Task 24; for now the app
//! runs on [`Config::default`].

use std::collections::{BTreeMap, BTreeSet};

/// Render quality: the internal resolution scale and the mesh-size cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowState {
    /// Inner size in logical pixels.
    pub size: [u32; 2],
    /// Outer position in physical pixels; `None` lets the OS place it.
    pub position: Option<[i32; 2]>,
    /// The monitor the window was last on, by name.
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

/// Everything the user can configure.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// The output device to capture, by name; `None` is the system default.
    pub audio_device: Option<String>,
    /// A/V offset per device name, in milliseconds.
    pub av_offsets_ms: BTreeMap<String, i32>,
    pub auto_advance_s: f64,
    pub blend_s: f64,
    pub hard_cuts: bool,
    pub hard_cut_strength: f32,
    pub quality: Quality,
    pub favourites: BTreeSet<String>,
    pub bans: BTreeSet<String>,
    /// Presets blamed for a GPU device loss (§6.7).
    pub blocklist: BTreeSet<String>,
    pub window: WindowState,
    /// The monitor to go fullscreen on, by name; `None` is the window's.
    pub fullscreen_monitor: Option<String>,
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
