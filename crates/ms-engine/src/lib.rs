//! ms-engine: fixed-tick clock, render-quality governor, and preset
//! instances that run a compiled preset's code each tick and produce plain
//! render data for `ms-render`.

mod clock;
mod env;
mod governor;
mod objects;
mod tick;

pub use clock::{FixedClock, Ticks};
pub use env::{Globals, TickInput};
pub use governor::Governor;
pub use tick::{PresetInstance, TickOutput, VertexPath};
