//! ms-render: wgpu device setup, the feedback chain, the warp and composite
//! passes, the overlays (waves, shapes, borders, motion vectors), the blur
//! chain, noise, spectrum and user textures, the shader prelude and user
//! shader compilation, the on-screen text, device-loss recreation, texture
//! readback and the golden-image test harness that every rendering pass is
//! tested against.

mod blur;
mod borders;
mod composite;
mod draw;
mod feedback;
pub mod golden;
mod gpu;
mod motion;
mod noise;
mod shader_scan;
mod shaders;
mod shapes;
#[doc(hidden)]
pub mod test_support;
mod text;
mod textures;
mod warmup;
mod warp;
mod waves;

pub use borders::Borders;
pub use composite::{CompShader, CompositeLayer};
pub use draw::DrawList;
pub use feedback::{FeedbackFormat, RenderConfig, Renderer};
pub use gpu::{read_texture, Gpu, GpuError, FORCE_WARP_ENV};
pub use motion::MotionVectors;
pub use noise::{noise_textures, NoiseSet, NOISE_SEED};
pub use shaders::{RendererShared, ShaderSet, ShaderSources, Uniforms, PRELUDE_WGSL};
pub use shapes::ShapeDraw;
pub use text::TextItem;
pub use textures::{
    PresetPicks, TextureResolver, MAX_DECODE_BYTES, MAX_DECODE_DIM, MAX_FILE_BYTES,
    MAX_TEXTURE_DIM, SPECTRUM_BINS,
};
pub use warp::{MeshData, WarpLayer, WarpShader};
pub use waves::{main_wave_points, MainWave, WaveDraw};
