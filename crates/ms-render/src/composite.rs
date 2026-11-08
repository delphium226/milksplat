//! The composite pass: draws the latest feedback buffer to a display target
//! through one or more composite shaders (spec §6.3).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::feedback::Renderer;
use crate::gpu::Gpu;
use crate::shaders::{self, Uniforms};
use crate::warp::{self, LayerSlot};

/// Layer mixing: colour and alpha = src * k + dst * (1 - k), with k the blend
/// constant, set per draw to the layer's alpha (1 for the first layer). The
/// shader writes alpha 1 and the target is cleared to opaque black, so the
/// result stays opaque.
///
/// This is `SrcAlpha / OneMinusSrcAlpha` blending with the layer alpha
/// supplied as the blend constant instead of through the shader, which keeps
/// the fixed uniform block unchanged.
const COMP_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Constant,
        dst_factor: wgpu::BlendFactor::OneMinusConstant,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Constant,
        dst_factor: wgpu::BlendFactor::OneMinusConstant,
        operation: wgpu::BlendOperation::Add,
    },
};

/// A compiled composite shader (built-in or a preset's).
///
/// Belongs to the [`RendererShared`](crate::RendererShared) generation that
/// compiled it; after [`Renderer::recreate`] it is stale and
/// [`Renderer::composite`] draws the built-in composite in its place.
///
/// Pipelines are cached here per target format. Shader compilation (on the
/// loader thread) creates the one for the target format it is given; any
/// other format is created on first use in [`Renderer::composite`] (a safety
/// net, e.g. after the window's format changed, which logs a warning), and a
/// format whose pipeline cannot be created falls back to the built-in
/// composite. Cheap to clone; `Send + Sync`.
#[derive(Clone)]
pub struct CompShader {
    inner: Arc<CompInner>,
}

struct CompInner {
    label: String,
    /// The `RendererShared` generation that compiled this shader.
    generation: u64,
    module: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    /// The user texture bind group, if the shader set has user textures.
    group1: Option<wgpu::BindGroup>,
    /// Pipelines by target format; `None` = creation failed.
    pipelines: Mutex<Vec<(wgpu::TextureFormat, Option<wgpu::RenderPipeline>)>>,
    /// Pipelines created by `composite` rather than ahead of time.
    lazy_creations: AtomicUsize,
}

impl CompShader {
    pub(crate) fn new(
        module: wgpu::ShaderModule,
        layout: wgpu::PipelineLayout,
        group1: Option<wgpu::BindGroup>,
        label: &str,
        generation: u64,
    ) -> Self {
        CompShader {
            inner: Arc::new(CompInner {
                label: label.to_string(),
                generation,
                module,
                layout,
                group1,
                pipelines: Mutex::new(Vec::new()),
                lazy_creations: AtomicUsize::new(0),
            }),
        }
    }

    /// Creates (inside this thread's error scopes) and caches the pipeline
    /// for `format`, unless it is cached already.
    pub(crate) fn prepare(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> Result<(), String> {
        let mut cache = self.lock();
        if let Some((_, pipeline)) = cache.iter().find(|(f, _)| *f == format) {
            return match pipeline {
                Some(_) => Ok(()),
                None => Err(format!("the pipeline for {format:?} failed earlier")),
            };
        }
        let (pipeline, error) = shaders::scoped(device, || self.create_pipeline(device, format));
        match error {
            None => {
                cache.push((format, Some(pipeline)));
                Ok(())
            }
            Some(e) => {
                cache.push((format, None));
                Err(format!("cannot create the pipeline for {format:?}: {e}"))
            }
        }
    }

    /// The cached pipeline for `format`: `None` if not created yet,
    /// `Some(None)` if creation failed.
    fn cached(&self, format: wgpu::TextureFormat) -> Option<Option<wgpu::RenderPipeline>> {
        self.lock()
            .iter()
            .find(|(f, _)| *f == format)
            .map(|(_, p)| p.clone())
    }

    /// The pipeline for `format`, created now (with a warning) if shader
    /// compilation did not prepare it. `None` if it cannot be created.
    fn pipeline(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> Option<wgpu::RenderPipeline> {
        if let Some(pipeline) = self.cached(format) {
            return pipeline;
        }
        tracing::warn!(
            "{}: no composite pipeline prepared for {format:?}; creating it on the render thread",
            self.inner.label
        );
        self.inner.lazy_creations.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = self.prepare(device, format) {
            tracing::error!("{}: {e}", self.inner.label);
        }
        self.cached(format).flatten()
    }

    /// How many pipelines [`Renderer::composite`] had to create itself
    /// because shader compilation had not prepared them (test hook).
    #[doc(hidden)]
    pub fn render_thread_pipeline_creations(&self) -> usize {
        self.inner.lazy_creations.load(Ordering::Relaxed)
    }

    fn lock(
        &self,
    ) -> std::sync::MutexGuard<'_, Vec<(wgpu::TextureFormat, Option<wgpu::RenderPipeline>)>> {
        // The cache stays consistent even if a holder panicked.
        self.inner
            .pipelines
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn create_pipeline(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let inner = &self.inner;
        let clamp = if clamps_output(format) { 1.0 } else { 0.0 };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&inner.label),
            layout: Some(&inner.layout),
            vertex: wgpu::VertexState {
                module: &inner.module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &inner.module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("ms_clamp_output", clamp)],
                    ..Default::default()
                },
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(COMP_BLEND),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }
}

/// Whether composite output is clamped to [0, 1] for `format`: yes for
/// unorm (and every other non-float) format, no for float formats, which keep
/// values above 1.
fn clamps_output(format: wgpu::TextureFormat) -> bool {
    use wgpu::TextureFormat as F;
    !matches!(
        format,
        F::R16Float
            | F::Rg16Float
            | F::Rgba16Float
            | F::R32Float
            | F::Rg32Float
            | F::Rgba32Float
            | F::Rg11b10Ufloat
    )
}

/// One weighted composite of the latest feedback buffer.
pub struct CompositeLayer<'a> {
    pub shader: &'a CompShader,
    pub uniforms: &'a Uniforms,
    /// Mix weight over the layers before it, clamped to [0, 1] (non-finite
    /// counts as 0). Ignored for the first layer, which is drawn opaque.
    pub alpha: f32,
}

/// Per-layer uniform buffers and bind groups reused across composites.
#[derive(Default)]
pub(crate) struct CompositeState {
    slots: Vec<LayerSlot>,
}

impl CompositeState {
    /// Drops bind groups that reference feedback or auxiliary views.
    pub(crate) fn invalidate_bind_groups(&mut self) {
        for slot in &mut self.slots {
            slot.bind_groups = [None, None];
        }
    }
}

impl Renderer {
    /// Composites the latest feedback buffer into `target` (of
    /// `target_format`): clears it to opaque black, then draws a full-screen
    /// triangle per layer. The first layer is drawn opaque; each later layer
    /// is mixed over the result with its `alpha` (so layers with weights
    /// `1 - p` and `p` cross-fade, and a single layer always fills the
    /// target).
    ///
    /// A layer whose shader cannot be used for `target_format`, or is stale
    /// (compiled before the last [`Renderer::recreate`]), is drawn with the
    /// built-in composite instead. Pipelines for `target_format` should
    /// have been created by `compile_shaders(.., target_format, ..)`; any
    /// other format is created here, with a warning. Does not block.
    pub fn composite(
        &mut self,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
        layers: &[CompositeLayer<'_>],
    ) {
        let current = self.current;
        warp::ensure_slots(
            gpu,
            &mut self.comp_state.slots,
            layers.len(),
            "composite uniforms",
        );
        for (layer, slot) in layers.iter().zip(&mut self.comp_state.slots) {
            gpu.queue
                .write_buffer(&slot.uniforms, 0, bytemuck::bytes_of(layer.uniforms));
            if slot.bind_groups[current].is_none() {
                slot.bind_groups[current] = Some(shaders::group0_bind_group(
                    &self.shared,
                    &self.aux,
                    &slot.uniforms,
                    &self.views[current],
                    "composite group 0",
                ));
            }
        }

        let builtin = &self.shared.builtin_comp;
        let mut stale = false;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            for (i, (layer, slot)) in layers.iter().zip(&self.comp_state.slots).enumerate() {
                let Some(group0) = &slot.bind_groups[current] else {
                    continue;
                };
                // A stale shader's objects belong to an old device: never
                // touch them (not even to create a pipeline).
                let fresh = layer.shader.inner.generation == self.shared.generation;
                stale |= !fresh;
                let user = if fresh {
                    layer.shader.pipeline(&gpu.device, target_format)
                } else {
                    None
                };
                let (pipeline, group1) = match user {
                    Some(p) => (p, layer.shader.inner.group1.as_ref()),
                    None => match builtin.pipeline(&gpu.device, target_format) {
                        Some(p) => (p, None),
                        None => {
                            tracing::error!(
                                "composite: no pipeline for {target_format:?}; skipping layer {i}"
                            );
                            continue;
                        }
                    },
                };
                let alpha = if i == 0 {
                    1.0
                } else if layer.alpha.is_finite() {
                    f64::from(layer.alpha.clamp(0.0, 1.0))
                } else {
                    0.0
                };
                pass.set_pipeline(&pipeline);
                pass.set_blend_constant(wgpu::Color {
                    r: alpha,
                    g: alpha,
                    b: alpha,
                    a: alpha,
                });
                pass.set_bind_group(0, group0, &[]);
                if let Some(group1) = group1 {
                    pass.set_bind_group(1, group1, &[]);
                }
                pass.draw(0..3, 0..1);
            }
        }
        gpu.queue.submit([encoder.finish()]);
        if stale {
            self.report_stale("composite");
        }
    }
}
