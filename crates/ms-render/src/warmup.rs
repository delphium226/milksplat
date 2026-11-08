//! The hidden warm-up draw (spec §3.2): after a preset's shaders are
//! compiled on the loader thread, one warp pass and one composite are drawn
//! with them into a private 64x64 target, so any shader work the driver
//! defers to the first draw happens off the render thread.

use crate::feedback::FeedbackFormat;
use crate::gpu::poll_wait_on;
use crate::shaders::{self, scoped, AuxViews, RendererShared, ShaderSet, Uniforms};
use crate::warp::{self, Vertex};

/// Side of the private warm-up target, in pixels.
const SIZE: u32 = 64;

impl RendererShared {
    /// Draws `set` once, off screen: a warp pass (with the pipeline for
    /// `feedback`) over a 1x1 mesh into a private 64x64 feedback-format
    /// target, then a composite into a private 64x64 `comp_format` target
    /// (with the pipeline `compile_shaders(.., comp_format, ..)` prepared).
    /// Everything runs inside this thread's error scopes, then waits for the
    /// GPU (through the same lost-device-safe poll as `Gpu::poll_wait`).
    ///
    /// The targets and every other object are private and dropped
    /// afterwards; the renderer's own state is not touched. Safe to call
    /// from any thread. Returns `Err` (never panics) if `set` belongs to
    /// another renderer generation, if a GPU error was raised, or if the
    /// device is lost.
    pub fn warm_up(
        &self,
        set: &ShaderSet,
        feedback: FeedbackFormat,
        comp_format: wgpu::TextureFormat,
    ) -> Result<(), String> {
        if set.warp.generation != self.generation || set.comp.generation() != self.generation {
            return Err("the shader set belongs to another renderer generation".into());
        }
        let ((), error) = scoped(&self.device, || {
            self.warm_up_draw(set, feedback, comp_format)
        });
        if let Some(e) = error {
            return Err(format!("warm-up draw failed: {e}"));
        }
        poll_wait_on(&self.device, &self.lost).map_err(|e| e.to_string())
    }

    fn warm_up_draw(
        &self,
        set: &ShaderSet,
        feedback: FeedbackFormat,
        comp_format: wgpu::TextureFormat,
    ) {
        let device = &self.device;
        let target = |label, format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: SIZE,
                        height: SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let fb_format = feedback.texture_format();
        let src = target("warm-up feedback src", fb_format);
        let dst = target("warm-up feedback dst", fb_format);
        let out = target("warm-up composite", comp_format);
        let volume = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("warm-up volume"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let white = &self.white;
        let aux = AuxViews {
            blur: [white.clone(), white.clone(), white.clone()],
            noise: [white.clone(), white.clone(), white.clone()],
            noisevol: [volume.clone(), volume],
            spectrum: white.clone(),
        };

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warm-up uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniforms, 0, bytemuck::bytes_of(&Uniforms::default()));
        let warp_group0 =
            shaders::group0_bind_group(self, &aux, &uniforms, &src, "warm-up warp group 0");
        let comp_group0 =
            shaders::group0_bind_group(self, &aux, &uniforms, &dst, "warm-up comp group 0");

        // A 1x1 mesh: four corners with identity UVs and full weight.
        let vertices: Vec<Vertex> = [(0.0f32, 0.0f32), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
            .iter()
            .map(|&(x, y)| Vertex {
                pos: [x * 2.0 - 1.0, 1.0 - y * 2.0],
                uv: [x, y],
                color: [1.0; 4],
            })
            .collect();
        let mut indices = Vec::with_capacity(6);
        warp::push_indices(&mut indices, 1, 1);
        let buffer = |label, bytes: &[u8], usage| {
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&b, 0, bytes);
            b
        };
        let vb = buffer(
            "warm-up vertices",
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );
        let ib = buffer(
            "warm-up indices",
            bytemuck::cast_slice(&indices),
            wgpu::BufferUsages::INDEX,
        );

        let attachment = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("warm-up"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("warm-up warp"),
                color_attachments: &[attachment(&dst)],
                ..Default::default()
            });
            pass.set_pipeline(&set.warp.pipelines[feedback.index()]);
            pass.set_bind_group(0, &warp_group0, &[]);
            if let Some(group1) = &set.warp.group1 {
                pass.set_bind_group(1, group1, &[]);
            }
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }
        match set.comp.cached(comp_format).flatten() {
            Some(pipeline) => {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("warm-up composite"),
                    color_attachments: &[attachment(&out)],
                    ..Default::default()
                });
                pass.set_pipeline(&pipeline);
                pass.set_blend_constant(wgpu::Color::WHITE);
                pass.set_bind_group(0, &comp_group0, &[]);
                if let Some(group1) = set.comp.group1() {
                    pass.set_bind_group(1, group1, &[]);
                }
                pass.draw(0..3, 0..1);
            }
            None => tracing::warn!(
                "warm-up: no composite pipeline was prepared for {comp_format:?}; \
                 skipping the composite draw"
            ),
        }
        self.queue.submit([encoder.finish()]);
    }
}
