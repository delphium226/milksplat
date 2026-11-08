//! ms-render: wgpu device setup, the feedback chain and warp pass, texture
//! readback and the golden-image test harness that every rendering pass is
//! tested against.

mod feedback;
pub mod golden;
mod gpu;
mod warp;

pub use feedback::{FeedbackFormat, RenderConfig, Renderer};
pub use gpu::{read_texture, Gpu, GpuError, FORCE_WARP_ENV};
pub use warp::{MeshData, Uniforms, WarpLayer, WarpShader};
