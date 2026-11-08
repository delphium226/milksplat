//! ms-render: wgpu device setup, texture readback and the golden-image test
//! harness that every rendering pass is tested against.

pub mod golden;
mod gpu;

pub use gpu::{read_texture, Gpu, GpuError, FORCE_WARP_ENV};
