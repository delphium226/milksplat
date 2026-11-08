//! The feedback chain: a ping-pong pair of textures at internal resolution,
//! plus resizing and format conversion that keep the current image.

use image::RgbaImage;

use crate::composite::CompositeState;
use crate::gpu::Gpu;
use crate::shaders::{AuxViews, RendererShared};
use crate::warp::WarpState;

/// Smallest internal size per axis.
const MIN_DIM: u32 = 16;

/// Pixel format of the feedback buffers.
///
/// This is `ms-render`'s own enum; the engine maps the preset's format onto it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FeedbackFormat {
    /// `Rgba8Unorm` (the default, and what imported presets use).
    #[default]
    Rgba8,
    /// `Rgba16Float`; the warp pass clamps to [0, 64] and drops NaN/Inf.
    Fp16,
}

impl FeedbackFormat {
    /// The wgpu texture format for this feedback format.
    pub fn texture_format(self) -> wgpu::TextureFormat {
        match self {
            FeedbackFormat::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
            FeedbackFormat::Fp16 => wgpu::TextureFormat::Rgba16Float,
        }
    }

    /// Index into per-format pipeline arrays.
    pub(crate) fn index(self) -> usize {
        match self {
            FeedbackFormat::Rgba8 => 0,
            FeedbackFormat::Fp16 => 1,
        }
    }

    pub(crate) const ALL: [FeedbackFormat; 2] = [FeedbackFormat::Rgba8, FeedbackFormat::Fp16];
}

/// How the renderer is set up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderConfig {
    /// Feedback buffer size in pixels (clamped like [`Renderer::resize`]).
    pub internal_size: [u32; 2],
    pub format: FeedbackFormat,
}

/// GPU state for the visualizer: the feedback chain, the warp and composite
/// passes and the shared shader resources.
pub struct Renderer {
    pub(crate) format: FeedbackFormat,
    pub(crate) size: [u32; 2],
    /// Ping-pong pair; `textures[current]` is the latest image.
    pub(crate) textures: [wgpu::Texture; 2],
    pub(crate) views: [wgpu::TextureView; 2],
    pub(crate) current: usize,
    /// Layouts, samplers and built-in shaders (also handed to loader
    /// threads via [`Renderer::shared`]).
    pub(crate) shared: RendererShared,
    /// Group 0 bindings 2-10 (placeholders until the blur chain and noise
    /// textures replace them; call [`Renderer::invalidate_bind_groups`]
    /// after changing them).
    pub(crate) aux: AuxViews,
    pub(crate) warp_state: WarpState,
    pub(crate) comp_state: CompositeState,
    blit: Blit,
}

impl Renderer {
    /// Creates the feedback textures (cleared to black), the shared shader
    /// resources (layouts, samplers, built-in shaders, placeholder textures)
    /// and the blit resources.
    pub fn new(gpu: &Gpu, cfg: RenderConfig) -> Self {
        let size = clamp_size(gpu, cfg.internal_size);
        let format = cfg.format;
        let textures = [0, 1].map(|i| create_feedback_texture(gpu, size, format, i));
        let views = [0, 1].map(|i| textures[i].create_view(&Default::default()));
        clear_black(gpu, &views);

        let shared = RendererShared::new(gpu);
        let aux = AuxViews::placeholders(gpu);

        Renderer {
            format,
            size,
            textures,
            views,
            current: 0,
            shared,
            aux,
            warp_state: WarpState::default(),
            comp_state: CompositeState::default(),
            blit: Blit::new(gpu),
        }
    }

    /// The most recent feedback buffer (last written by the warp pass, or the
    /// initial one).
    pub fn feedback_texture(&self) -> &wgpu::Texture {
        &self.textures[self.current]
    }

    /// The current feedback size in pixels (after clamping).
    pub fn internal_size(&self) -> [u32; 2] {
        self.size
    }

    /// The current feedback format.
    pub fn format(&self) -> FeedbackFormat {
        self.format
    }

    /// Changes the internal resolution, clamped per axis to
    /// `[16, max_texture_dimension_2d]`. The latest image is kept by a
    /// bilinear copy into the new size.
    pub fn resize(&mut self, gpu: &Gpu, internal_size: [u32; 2]) {
        let size = clamp_size(gpu, internal_size);
        if size != self.size {
            self.rebuild(gpu, size, self.format);
        }
    }

    /// Changes the feedback format, converting the latest image.
    pub fn set_format(&mut self, gpu: &Gpu, f: FeedbackFormat) {
        if f != self.format {
            self.rebuild(gpu, self.size, f);
        }
    }

    /// Test seeding: uploads an RGBA8 image into the current buffer
    /// (converted to f16 in `Fp16`). The image must match the buffer size.
    #[doc(hidden)]
    pub fn write_feedback(&self, gpu: &Gpu, rgba8: &RgbaImage) {
        if rgba8.dimensions() != (self.size[0], self.size[1]) {
            tracing::error!(
                "write_feedback: image is {:?}, feedback is {:?}",
                rgba8.dimensions(),
                self.size
            );
            return;
        }
        match self.format {
            FeedbackFormat::Rgba8 => self.upload(gpu, rgba8.as_raw(), 4),
            FeedbackFormat::Fp16 => {
                let bytes: Vec<u8> = rgba8
                    .as_raw()
                    .iter()
                    .flat_map(|&v| half::f16::from_f32(f32::from(v) / 255.0).to_le_bytes())
                    .collect();
                self.upload(gpu, &bytes, 8);
            }
        }
    }

    /// Test seeding: uploads raw f16 texels (row-major) into the current
    /// buffer (clamped to [0, 1] and quantised in `Rgba8`, NaN becoming 0).
    #[doc(hidden)]
    pub fn write_feedback_f16(&self, gpu: &Gpu, data: &[[half::f16; 4]]) {
        let texels = self.size[0] as usize * self.size[1] as usize;
        if data.len() != texels {
            tracing::error!(
                "write_feedback_f16: {} texels given, feedback has {texels}",
                data.len()
            );
            return;
        }
        let bytes: Vec<u8> = match self.format {
            FeedbackFormat::Fp16 => data
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            FeedbackFormat::Rgba8 => data
                .iter()
                .flatten()
                // NaN saturates to 0 in the cast.
                .map(|v| (v.to_f32().clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect(),
        };
        let bpp = match self.format {
            FeedbackFormat::Rgba8 => 4,
            FeedbackFormat::Fp16 => 8,
        };
        self.upload(gpu, &bytes, bpp);
    }

    fn upload(&self, gpu: &Gpu, data: &[u8], bytes_per_pixel: u32) {
        let tex = self.feedback_texture();
        gpu.queue.write_texture(
            tex.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tex.width() * bytes_per_pixel),
                rows_per_image: Some(tex.height()),
            },
            tex.size(),
        );
    }

    /// Recreates both feedback textures at `size`/`format`, bilinearly
    /// copying the latest image into the new current buffer and clearing the
    /// other one.
    fn rebuild(&mut self, gpu: &Gpu, size: [u32; 2], format: FeedbackFormat) {
        let textures = [0, 1].map(|i| create_feedback_texture(gpu, size, format, i));
        let views = [0, 1].map(|i| textures[i].create_view(&Default::default()));
        clear_black(gpu, &views[1..]);
        self.blit.copy(
            gpu,
            &self.views[self.current],
            &self.shared.samplers.fc,
            &views[0],
            format,
        );
        self.textures = textures;
        self.views = views;
        self.current = 0;
        self.size = size;
        self.format = format;
        // Per-layer bind groups reference the old texture views.
        self.invalidate_bind_groups();
    }

    /// Drops the cached group 0 bind groups (they reference the feedback
    /// and auxiliary texture views).
    pub(crate) fn invalidate_bind_groups(&mut self) {
        self.warp_state.invalidate_bind_groups();
        self.comp_state.invalidate_bind_groups();
    }
}

fn clamp_size(gpu: &Gpu, [w, h]: [u32; 2]) -> [u32; 2] {
    // The device was created with the adapter's limits; use the device's own
    // value since that is what texture creation is validated against.
    let max = gpu.device.limits().max_texture_dimension_2d.max(MIN_DIM);
    [w.clamp(MIN_DIM, max), h.clamp(MIN_DIM, max)]
}

fn create_feedback_texture(
    gpu: &Gpu,
    [width, height]: [u32; 2],
    format: FeedbackFormat,
    index: usize,
) -> wgpu::Texture {
    let label = if index == 0 {
        "feedback 0"
    } else {
        "feedback 1"
    };
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format.texture_format(),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Clears each view to opaque black.
fn clear_black(gpu: &Gpu, views: &[wgpu::TextureView]) {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("feedback clear"),
        });
    for view in views {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("feedback clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
    }
    gpu.queue.submit([encoder.finish()]);
}

/// Full-target bilinear copy, one pipeline per feedback format.
struct Blit {
    bind_group_layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::RenderPipeline; 2],
}

impl Blit {
    fn new(gpu: &Gpu) -> Self {
        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("blit"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("blit"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("blit.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
            });
        let pipelines = FeedbackFormat::ALL.map(|f| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("blit"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: f.texture_format(),
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        });
        Blit {
            bind_group_layout,
            pipelines,
        }
    }

    /// Draws `src` over the whole of `dst` (format `dst_format`), sampled
    /// with `sampler`.
    fn copy(
        &self,
        gpu: &Gpu,
        src: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        dst: &wgpu::TextureView,
        dst_format: FeedbackFormat,
    ) {
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blit"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipelines[dst_format.index()]);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        gpu.queue.submit([encoder.finish()]);
    }
}
