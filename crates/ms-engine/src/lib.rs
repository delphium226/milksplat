//! ms-engine: fixed-tick clock, render-quality governor, and preset
//! instances that run a compiled preset's code each tick and produce plain
//! render data for `ms-render`, and the blending of two presets during a
//! transition.

mod blend;
mod clock;
mod env;
mod governor;
mod objects;
mod tick;

pub use blend::{
    blend_weights, composite_layers, mix_meshes, render_blend, BlendPattern, BlendScratch,
};
pub use clock::{FixedClock, Ticks};
pub use env::{Globals, TickInput};
pub use governor::Governor;
pub use tick::{PresetInstance, TickOutput, VertexPath};
