//! ms-preset: the `.splat` preset file format, compilation and selection.

mod compile;
mod format;
mod library;
mod select;
pub use compile::*;
pub use format::*;
pub use library::*;
pub use select::*;
