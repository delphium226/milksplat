//! ms-render: wgpu device setup, the feedback chain, the warp and composite
//! passes, the shader prelude and user shader compilation, texture readback
//! and the golden-image test harness that every rendering pass is tested
//! against.

mod composite;
mod feedback;
pub mod golden;
mod gpu;
mod shader_scan;
mod shaders;
mod warp;

pub use composite::{CompShader, CompositeLayer};
pub use feedback::{FeedbackFormat, RenderConfig, Renderer};
pub use gpu::{read_texture, Gpu, GpuError, FORCE_WARP_ENV};
pub use shaders::{RendererShared, ShaderSet, Uniforms, PRELUDE_WGSL};
pub use warp::{MeshData, WarpLayer, WarpShader};
