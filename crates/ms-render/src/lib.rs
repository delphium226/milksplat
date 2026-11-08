//! ms-render: wgpu device setup, the feedback chain, the warp and composite
//! passes, the blur chain, noise, spectrum and user textures, the shader
//! prelude and user shader compilation, texture readback and the
//! golden-image test harness that every rendering pass is tested against.

mod blur;
mod composite;
mod feedback;
pub mod golden;
mod gpu;
mod noise;
mod shader_scan;
mod shaders;
mod textures;
mod warp;

pub use composite::{CompShader, CompositeLayer};
pub use feedback::{FeedbackFormat, RenderConfig, Renderer};
pub use gpu::{read_texture, Gpu, GpuError, FORCE_WARP_ENV};
pub use noise::{noise_textures, NoiseSet, NOISE_SEED};
pub use shaders::{RendererShared, ShaderSet, Uniforms, PRELUDE_WGSL};
pub use textures::{
    PresetPicks, TextureResolver, MAX_DECODE_BYTES, MAX_DECODE_DIM, MAX_FILE_BYTES,
    MAX_TEXTURE_DIM, SPECTRUM_BINS,
};
pub use warp::{MeshData, WarpLayer, WarpShader};
