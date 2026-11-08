//! ms-engine: fixed-tick clock and render-quality governor.

mod clock;
mod governor;

pub use clock::{FixedClock, Ticks};
pub use governor::Governor;
