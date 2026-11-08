//! The blur chain (spec §6.1): three levels, each half the size of the one
//! before (level 1 is half the feedback size), bound at group 0 bindings 2-4
//! as `sampler_blur1`..`3`.
//!
//! Level n is a separable 9-tap Gaussian of level n-1 (level 1 blurs the
//! current feedback buffer), in two passes: horizontal into a scratch
//! texture of the level's size, then vertical into the level. Taps are at
//! integer offsets in the target's texel space, sampled with linear filtering
//! and clamp addressing; the horizontal pass's bilinear reads of the
//! twice-as-large source also do the 2x2 downsample. Levels have the
//! feedback format and hold raw values (`GetBlurN` applies
//! `blur_min`/`blur_max`).

use wgpu::util::DeviceExt as _;

use crate::feedback::{clear_black, FeedbackFormat, Renderer};
use crate::gpu::Gpu;

/// Number of blur levels.
pub(crate) const LEVELS: usize = 3;

/// Gaussian standard deviation, in target texels.
const SIGMA: f64 = 2.0;

/// The 9-tap weights for offsets 0..=4 (offsets +-i share `w[i]`):
/// `exp(-i^2 / (2 sigma^2))` with sigma = 2, normalised so the nine taps sum
/// to 1. That is about `[0.204164, 0.180174, 0.123832, 0.066282, 0.027631]`.
pub(crate) fn gaussian_weights() -> [f32; 5] {
    let raw: [f64; 5] = std::array::from_fn(|i| (-((i * i) as f64) / (2.0 * SIGMA * SIGMA)).exp());
    let sum = raw[0] + 2.0 * raw[1..].iter().sum::<f64>();
    raw.map(|w| (w / sum) as f32)
}

/// The blur pipelines and the three levels' textures and bind groups.
pub(crate) struct Blur {
    layout: wgpu::BindGroupLayout,
    /// One pipeline per feedback format (both passes share it; the tap
    /// direction is a uniform).
    pipelines: [wgpu::RenderPipeline; 2],
    format: FeedbackFormat,
    levels: [Level; LEVELS],
}

struct Level {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    scratch: wgpu::TextureView,
    /// Horizontal pass, indexed by the current feedback buffer (only level
    /// 1 reads it; levels 2-3 hold the same bind group twice).
    horizontal: [wgpu::BindGroup; 2],
    vertical: wgpu::BindGroup,
}

impl Blur {
    /// Builds the pipelines and the levels for a feedback chain of `size`
    /// and `format` whose views are `feedback`.
    ///
    /// Deliberately not wrapped in error scopes (unlike preset shaders,
    /// spec §6.7): the WGSL and every descriptor here are fixed, internal
    /// and covered by the tests on hardware and WARP, so nothing here
    /// depends on user data. A failure would be a bug, reported through the
    /// device's uncaptured-error handler, which logs.
    pub(crate) fn new(
        gpu: &Gpu,
        sampler: &wgpu::Sampler,
        feedback: &[wgpu::TextureView; 2],
        size: [u32; 2],
        format: FeedbackFormat,
    ) -> Self {
        let device = &gpu.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blur"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let w = gaussian_weights();
        let source = format!(
            "const W = array<f32, 5>({:?}, {:?}, {:?}, {:?}, {:?});\n{}",
            w[0],
            w[1],
            w[2],
            w[3],
            w[4],
            include_str!("shaders/blur.wgsl")
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur.wgsl"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = FeedbackFormat::ALL.map(|f| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blur"),
                layout: Some(&pipeline_layout),
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
        let levels = create_levels(gpu, &layout, sampler, feedback, size, format);
        Blur {
            layout,
            pipelines,
            format,
            levels,
        }
    }

    /// Recreates the levels (cleared to black) for a new feedback chain.
    /// The caller must put the new [`Blur::views`] in group 0.
    pub(crate) fn recreate(
        &mut self,
        gpu: &Gpu,
        sampler: &wgpu::Sampler,
        feedback: &[wgpu::TextureView; 2],
        size: [u32; 2],
        format: FeedbackFormat,
    ) {
        self.levels = create_levels(gpu, &self.layout, sampler, feedback, size, format);
        self.format = format;
    }

    /// The level views, for group 0 bindings 2-4.
    pub(crate) fn views(&self) -> [wgpu::TextureView; LEVELS] {
        std::array::from_fn(|i| self.levels[i].view.clone())
    }

    /// The texture of level `level` (1-3).
    pub(crate) fn texture(&self, level: u8) -> Option<&wgpu::Texture> {
        let i = usize::from(level).checked_sub(1)?;
        self.levels.get(i).map(|l| &l.texture)
    }

    /// Computes levels `1..=levels` (clamped to 3) from feedback buffer
    /// `current`. Does not block.
    pub(crate) fn run(&self, gpu: &Gpu, current: usize, levels: u8) {
        let count = usize::from(levels).min(LEVELS);
        if count == 0 {
            return;
        }
        let pipeline = &self.pipelines[self.format.index()];
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blur"),
            });
        for level in &self.levels[..count] {
            for (target, bind_group) in [
                (&level.scratch, &level.horizontal[current]),
                (&level.view, &level.vertical),
            ] {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("blur"),
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
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        gpu.queue.submit([encoder.finish()]);
    }
}

impl Renderer {
    /// Computes blur levels `1..=levels` (clamped to 3; 0 does nothing) from
    /// the current feedback buffer. Higher levels keep their contents. Does
    /// not block.
    pub fn blur(&mut self, gpu: &Gpu, levels: u8) {
        self.blur.run(gpu, self.current, levels);
    }

    /// The texture of blur level `level` (1-3), or `None` for other values.
    /// Level 1 is half the internal size per axis (at least 1), and each
    /// further level half the one before, in the feedback format.
    pub fn blur_texture(&self, level: u8) -> Option<&wgpu::Texture> {
        self.blur.texture(level)
    }
}

/// Size of level `n` (1-based) for a feedback buffer of `size`: halved per
/// level, each axis at least 1.
fn level_size(size: [u32; 2], n: u32) -> [u32; 2] {
    size.map(|s| (s >> n).max(1))
}

fn create_levels(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    feedback: &[wgpu::TextureView; 2],
    size: [u32; 2],
    format: FeedbackFormat,
) -> [Level; LEVELS] {
    let device = &gpu.device;
    let new_texture = |label: &str, [width, height]: [u32; 2], usage| {
        device.create_texture(&wgpu::TextureDescriptor {
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
                | usage,
            view_formats: &[],
        })
    };
    let bind_group = |src: &wgpu::TextureView, step: [f32; 4]| {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur step"),
            contents: bytemuck::cast_slice(&step),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        })
    };

    // Level n reads level n - 1, or both feedback views for level 1.
    let level = |n: u32, previous: Option<&Level>| {
        let [w, h] = level_size(size, n);
        let label = ["blur1", "blur2", "blur3"][(n - 1) as usize];
        let texture = new_texture(label, [w, h], wgpu::TextureUsages::COPY_SRC);
        let view = texture.create_view(&Default::default());
        let scratch = new_texture("blur scratch", [w, h], wgpu::TextureUsages::empty())
            .create_view(&Default::default());
        let h_step = [1.0 / w as f32, 0.0, 0.0, 0.0];
        let horizontal = match previous {
            None => feedback.each_ref().map(|src| bind_group(src, h_step)),
            Some(previous) => {
                let g = bind_group(&previous.view, h_step);
                [g.clone(), g]
            }
        };
        let vertical = bind_group(&scratch, [0.0, 1.0 / h as f32, 0.0, 0.0]);
        Level {
            texture,
            view,
            scratch,
            horizontal,
            vertical,
        }
    };
    let l1 = level(1, None);
    let l2 = level(2, Some(&l1));
    let l3 = level(3, Some(&l2));
    let levels = [l1, l2, l3];
    let views = levels.each_ref().map(|l| l.view.clone());
    clear_black(gpu, &views);
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_are_a_normalised_gaussian() {
        let w = gaussian_weights();
        let sum = w[0] + 2.0 * w[1..].iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-6, "sum {sum}");
        let expected = [
            0.204_163_69,
            0.180_173_82,
            0.123_831_54,
            0.066_282_25,
            0.027_630_55,
        ];
        for (a, b) in w.iter().zip(expected) {
            assert!((a - b).abs() < 1e-7, "{w:?}");
        }
    }

    #[test]
    fn level_sizes_halve() {
        assert_eq!(level_size([512, 512], 1), [256, 256]);
        assert_eq!(level_size([512, 512], 3), [64, 64]);
        assert_eq!(level_size([100, 60], 3), [12, 7]);
        assert_eq!(level_size([16, 3], 3), [2, 1]);
    }
}
