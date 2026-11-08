//! The composite pass: draws the latest feedback buffer to a display target
//! through one or more composite shaders (spec §6.3).

use std::sync::{Arc, Mutex};

use crate::feedback::Renderer;
use crate::gpu::Gpu;
use crate::shaders::{self, Uniforms};
use crate::warp::{self, LayerSlot};

/// The target format used to create a composite shader's first pipeline at
/// compile time, so that pipeline errors are caught there.
const VALIDATION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

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
/// Pipelines are created per target format on first use and cached here; a
/// format whose pipeline cannot be created falls back to the built-in
/// composite. Cheap to clone; `Send + Sync`.
#[derive(Clone)]
pub struct CompShader {
    inner: Arc<CompInner>,
}

struct CompInner {
    label: String,
    module: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    /// The user texture bind group, if the shader set has user textures.
    group1: Option<wgpu::BindGroup>,
    /// Pipelines by target format; `None` = creation failed.
    pipelines: Mutex<Vec<(wgpu::TextureFormat, Option<wgpu::RenderPipeline>)>>,
}

impl CompShader {
    pub(crate) fn new(
        module: wgpu::ShaderModule,
        layout: wgpu::PipelineLayout,
        group1: Option<wgpu::BindGroup>,
        label: &str,
    ) -> Self {
        CompShader {
            inner: Arc::new(CompInner {
                label: label.to_string(),
                module,
                layout,
                group1,
                pipelines: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Creates and caches the pipeline for [`VALIDATION_FORMAT`] without an
    /// error scope of its own: the caller's scope sees any error.
    pub(crate) fn create_validation_pipeline(&self, device: &wgpu::Device) {
        let pipeline = self.create_pipeline(device, VALIDATION_FORMAT);
        self.lock().push((VALIDATION_FORMAT, Some(pipeline)));
    }

    /// The pipeline for `format`, created (inside error scopes) and cached on
    /// first use. `None` if it cannot be created.
    fn pipeline(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> Option<wgpu::RenderPipeline> {
        let mut cache = self.lock();
        if let Some((_, pipeline)) = cache.iter().find(|(f, _)| *f == format) {
            return pipeline.clone();
        }
        let (pipeline, error) = shaders::scoped(device, || self.create_pipeline(device, format));
        let pipeline = match error {
            None => Some(pipeline),
            Some(e) => {
                tracing::error!(
                    "{}: cannot create a composite pipeline for {format:?}: {e}",
                    self.inner.label
                );
                None
            }
        };
        cache.push((format, pipeline.clone()));
        pipeline
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
                compilation_options: Default::default(),
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
    /// A layer whose shader cannot be used for `target_format` is drawn with
    /// the built-in composite instead. Does not block.
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
                let (pipeline, group1) = match layer.shader.pipeline(&gpu.device, target_format) {
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
    }
}
