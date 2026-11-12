//! ms-engine: fixed-tick clock, render-quality governor, and preset
//! instances that run a compiled preset's code each tick and produce plain
//! render data for `ms-render`, the blending of two presets during a
//! transition, per-preset reactivity (audio scaling), the loader thread
//! that prepares presets off the render thread, and the director that
//! decides what is shown.

mod blend;
mod clock;
mod director;
mod env;
mod governor;
mod loader;
mod objects;
mod reactivity;
mod tick;

pub use blend::{
    blend_weights, composite_layers, mix_meshes, render_blend, BlendPattern, BlendScratch,
};
pub use clock::{FixedClock, Ticks};
pub use director::{
    Command, Director, DirectorConfig, FramePlan, State, TickInputBase, DEFAULT_PRESET_ID,
    DEFAULT_PRESET_SOURCE,
};
pub use env::{Globals, TickInput};
pub use governor::Governor;
pub use loader::{
    feedback_format, LoadResult, Loader, LOADER_STACK_BYTES, MAX_PRESET_BYTES, PICKS_SALT,
};
pub use reactivity::{
    effective_reactivity, step_reactivity, Reactivity, BAND_CEILING, BAND_SLEW_ALLOWANCE,
    REACTIVITY_MAX, REACTIVITY_MIN, REACTIVITY_STEP, SLEW_FACTOR, SPECTRUM_CEILING,
    SPECTRUM_SLEW_ALLOWANCE, WAVE_CEILING,
};
pub use tick::{PresetInstance, TickOutput, VertexPath};
