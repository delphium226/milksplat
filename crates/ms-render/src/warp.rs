//! The warp pass: redraws the previous feedback image through the warp mesh
//! into the other ping-pong buffer, blending one or more weighted layers.

use std::num::NonZeroU64;

use crate::feedback::{FeedbackFormat, Renderer};
use crate::gpu::Gpu;

/// Largest mesh accepted, in vertices per layer (the quality setting caps
/// real meshes at 193 * 145); bigger layers are skipped with an error.
const MAX_MESH_VERTICES: u64 = 1 << 20;

/// The uniform block shared by warp and composite shaders (std140 layout:
/// every row is 16 bytes).
///
/// Field order is part of the shader interface; the WGSL `Uniforms` struct
/// must match it exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub time: f32,
    pub fps: f32,
    pub frame: f32,
    pub progress: f32,

    pub bass: f32,
    pub mid: f32,
    pub treb: f32,
    pub beat: f32,

    pub bass_att: f32,
    pub mid_att: f32,
    pub treb_att: f32,
    pub beat_strength: f32,

    pub decay: f32,
    pub gamma: f32,
    pub echo_zoom: f32,
    pub echo_alpha: f32,

    pub echo_orient: f32,
    pub invert: f32,
    pub solarize: f32,
    pub brighten: f32,

    pub darken: f32,
    pub darken_center: f32,
    pub aspect_x: f32,
    pub aspect_y: f32,

    /// xy = size in pixels, zw = 1 / size.
    pub texsize: [f32; 4],
    /// New every tick.
    pub rand_frame: [f32; 4],
    /// Fixed per preset.
    pub rand_preset: [f32; 4],
    pub roam_cos: [f32; 4],
    pub roam_sin: [f32; 4],
    /// xyz used (blur levels 1-3).
    pub blur_min: [f32; 4],
    /// xyz used (blur levels 1-3).
    pub blur_max: [f32; 4],
    /// `q1`..`q32`, packed four per vec4.
    pub q: [[f32; 4]; 8],
}

impl Default for Uniforms {
    fn default() -> Self {
        Uniforms {
            decay: 0.98,
            gamma: 1.0,
            echo_zoom: 1.0,
            aspect_x: 1.0,
            aspect_y: 1.0,
            blur_max: [1.0; 4],
            ..bytemuck::Zeroable::zeroed()
        }
    }
}

/// The warp mesh for one layer: `(cols + 1) * (rows + 1)` vertices,
/// row-major, x left to right, y top to bottom.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshData {
    pub cols: u32,
    pub rows: u32,
    /// Where each vertex samples the previous frame, in texture space
    /// ([0, 1], v = 0 at the top).
    pub uv: Vec<[f32; 2]>,
    /// Per-vertex colour; rgb multiplies the warp result (alpha is unused:
    /// the layer weight takes its place).
    pub color: Vec<[f32; 4]>,
}

/// A compiled warp shader: one pipeline per feedback format, all using the
/// warp bind group layout (group 0: 0 `Uniforms`, 1 `sampler_main`,
/// 11 `s_fw`, 12 `s_fc`).
///
/// Only the built-in shader exists for now; user shaders plug into the same
/// [`WarpLayer::shader`] slot.
pub struct WarpShader {
    pipelines: [wgpu::RenderPipeline; 2],
}

impl WarpShader {
    /// The built-in warp: `sample(prev, uv).rgb * decay * colour.rgb`,
    /// sampled with linear filtering and wrap addressing.
    pub fn builtin(gpu: &Gpu, renderer: &Renderer) -> WarpShader {
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("warp_builtin.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/warp_builtin.wgsl").into()),
            });
        WarpShader::from_module(gpu, renderer, &module, "builtin warp")
    }

    /// Builds the per-format pipelines from a module with `vs_main` and
    /// `fs_main` entry points written against the warp bind group layout.
    pub(crate) fn from_module(
        gpu: &Gpu,
        renderer: &Renderer,
        module: &wgpu::ShaderModule,
        label: &str,
    ) -> WarpShader {
        let pipelines = FeedbackFormat::ALL.map(|format| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&renderer.warp_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[Some(Vertex::LAYOUT)],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: format.texture_format(),
                            blend: Some(LAYER_BLEND),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        });
        WarpShader { pipelines }
    }
}

/// Additive layer blending: colour = src * srcAlpha (the layer weight) +
/// dst; alpha keeps the target's value (cleared to 1).
const LAYER_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

/// One weighted contribution to the warp pass.
pub struct WarpLayer<'a> {
    pub mesh: &'a MeshData,
    /// Per-vertex weight (blend factor), one per mesh vertex; clamped to
    /// [0, 1], non-finite values count as 0.
    pub weight: &'a [f32],
    pub shader: &'a WarpShader,
    pub uniforms: &'a Uniforms,
}

/// Interleaved warp vertex: clip position, UV, colour (a = weight).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4];
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Vertex::ATTRIBUTES,
    };
}

/// Group 0 layout for every warp shader: 0 `Uniforms`, 1 `sampler_main`,
/// 11 `s_fw` (linear wrap), 12 `s_fc` (linear clamp).
pub(crate) fn create_bind_group_layout(gpu: &Gpu) -> wgpu::BindGroupLayout {
    let sampler = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    gpu.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("warp group 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                sampler(11),
                sampler(12),
            ],
        })
}

/// Mesh and per-layer GPU buffers reused across warp calls.
#[derive(Default)]
pub(crate) struct WarpState {
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    /// `(cols, rows)` of each drawn layer, in order, whose indices are
    /// currently in `index_buffer`.
    index_key: Vec<(u32, u32)>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    /// One uniform buffer (and bind groups) per layer position.
    slots: Vec<LayerSlot>,
}

struct LayerSlot {
    uniforms: wgpu::Buffer,
    /// Bind group sampling feedback texture 0 or 1.
    bind_groups: [Option<wgpu::BindGroup>; 2],
}

impl WarpState {
    /// Drops bind groups that reference feedback views (after the textures
    /// are recreated).
    pub(crate) fn invalidate_bind_groups(&mut self) {
        for slot in &mut self.slots {
            slot.bind_groups = [None, None];
        }
    }
}

/// A validated layer, ready to draw.
struct Draw<'l, 'a> {
    layer: &'l WarpLayer<'a>,
    base_vertex: i32,
    first_index: u32,
    index_count: u32,
}

/// Vertex count of a well-formed layer, or why it is malformed.
fn layer_vertex_count(layer: &WarpLayer<'_>) -> Result<u64, String> {
    let mesh = layer.mesh;
    if mesh.cols == 0 || mesh.rows == 0 {
        return Err(format!("empty mesh {}x{}", mesh.cols, mesh.rows));
    }
    let n = (u64::from(mesh.cols) + 1) * (u64::from(mesh.rows) + 1);
    if n > MAX_MESH_VERTICES {
        return Err(format!("mesh {}x{} is too large", mesh.cols, mesh.rows));
    }
    let lens = [mesh.uv.len(), mesh.color.len(), layer.weight.len()];
    if lens.iter().any(|&len| len as u64 != n) {
        return Err(format!(
            "mesh {}x{} needs {n} vertices, got uv {}, color {}, weight {}",
            mesh.cols, mesh.rows, lens[0], lens[1], lens[2]
        ));
    }
    Ok(n)
}

fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Makes sure `buf` holds at least `needed` bytes, reallocating (to the
/// next power of two) only when it is too small. Returns whether it was
/// reallocated.
fn ensure_capacity(
    gpu: &Gpu,
    buf: &mut Option<wgpu::Buffer>,
    needed: u64,
    usage: wgpu::BufferUsages,
    label: &str,
) -> bool {
    if buf.as_ref().is_some_and(|b| b.size() >= needed) {
        return false;
    }
    let max = gpu.device.limits().max_buffer_size & !(wgpu::COPY_BUFFER_ALIGNMENT - 1);
    let size = needed.next_power_of_two().min(max).max(needed);
    *buf = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    true
}

impl Renderer {
    /// Runs the warp pass: swaps the ping-pong buffers, clears the new target
    /// to black and draws each layer additively, weighted per vertex.
    ///
    /// Malformed layers (array lengths that do not match the mesh size, empty
    /// or oversized meshes) are skipped with an error log. Does not block.
    pub fn warp(&mut self, gpu: &Gpu, layers: &[WarpLayer<'_>]) {
        let src = self.current;
        let dst = 1 - src;
        let max_buffer = gpu.device.limits().max_buffer_size;
        let vertex_size = std::mem::size_of::<Vertex>() as u64;

        // Validate layers and lay them out in the shared buffers.
        let state = &mut self.warp_state;
        state.vertices.clear();
        let mut draws = Vec::with_capacity(layers.len());
        let mut key = Vec::with_capacity(layers.len());
        let (mut total_vertices, mut total_indices) = (0u64, 0u64);
        for (i, layer) in layers.iter().enumerate() {
            let n = match layer_vertex_count(layer) {
                Ok(n) => n,
                Err(why) => {
                    tracing::error!("warp: skipping layer {i}: {why}");
                    continue;
                }
            };
            let index_count = u64::from(layer.mesh.cols) * u64::from(layer.mesh.rows) * 6;
            let (end_vertex, end_index) = (total_vertices + n, total_indices + index_count);
            if i32::try_from(end_vertex).is_err()
                || u32::try_from(end_index).is_err()
                || end_vertex * vertex_size > max_buffer
                || end_index * 4 > max_buffer
            {
                tracing::error!("warp: skipping layer {i}: mesh buffers exceed the device limit");
                continue;
            }
            // The end offsets fit in i32/u32, so the start offsets and the
            // count do too.
            draws.push(Draw {
                layer,
                base_vertex: total_vertices as i32,
                first_index: total_indices as u32,
                index_count: index_count as u32,
            });
            key.push((layer.mesh.cols, layer.mesh.rows));
            total_vertices += n;
            total_indices += index_count;
            push_vertices(&mut state.vertices, layer);
        }

        // Upload mesh data.
        if !draws.is_empty() {
            ensure_capacity(
                gpu,
                &mut state.vertex_buffer,
                total_vertices * vertex_size,
                wgpu::BufferUsages::VERTEX,
                "warp vertices",
            );
            if ensure_capacity(
                gpu,
                &mut state.index_buffer,
                total_indices * 4,
                wgpu::BufferUsages::INDEX,
                "warp indices",
            ) {
                state.index_key.clear();
            }
            if let Some(vb) = &state.vertex_buffer {
                gpu.queue
                    .write_buffer(vb, 0, bytemuck::cast_slice(&state.vertices));
            }
            if state.index_key != key {
                state.indices.clear();
                for &(cols, rows) in &key {
                    push_indices(&mut state.indices, cols, rows);
                }
                if let Some(ib) = &state.index_buffer {
                    gpu.queue
                        .write_buffer(ib, 0, bytemuck::cast_slice(&state.indices));
                }
                state.index_key = key;
            }
        }

        // Per-layer uniforms and bind groups.
        while state.slots.len() < draws.len() {
            state.slots.push(LayerSlot {
                uniforms: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("warp uniforms"),
                    size: std::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                bind_groups: [None, None],
            });
        }
        for (draw, slot) in draws.iter().zip(&mut state.slots) {
            gpu.queue
                .write_buffer(&slot.uniforms, 0, bytemuck::bytes_of(draw.layer.uniforms));
            if slot.bind_groups[src].is_none() {
                slot.bind_groups[src] =
                    Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("warp group 0"),
                        layout: &self.warp_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: slot.uniforms.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&self.views[src]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 11,
                                resource: wgpu::BindingResource::Sampler(&self.sampler_wrap),
                            },
                            wgpu::BindGroupEntry {
                                binding: 12,
                                resource: wgpu::BindingResource::Sampler(&self.sampler_clamp),
                            },
                        ],
                    }));
            }
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("warp"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("warp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.views[dst],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Colour 0; alpha 1, which the blend state preserves.
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            if let (Some(vb), Some(ib)) = (&state.vertex_buffer, &state.index_buffer) {
                if !draws.is_empty() {
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                }
                for (draw, slot) in draws.iter().zip(&state.slots) {
                    let Some(bind_group) = &slot.bind_groups[src] else {
                        continue;
                    };
                    pass.set_pipeline(&draw.layer.shader.pipelines[self.format.index()]);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.draw_indexed(
                        draw.first_index..draw.first_index + draw.index_count,
                        draw.base_vertex,
                        0..1,
                    );
                }
            }
        }
        gpu.queue.submit([encoder.finish()]);
        self.current = dst;
    }
}

fn push_vertices(out: &mut Vec<Vertex>, layer: &WarpLayer<'_>) {
    let mesh = layer.mesh;
    let (cols, rows) = (mesh.cols as usize, mesh.rows as usize);
    for y in 0..=rows {
        for x in 0..=cols {
            let i = y * (cols + 1) + x;
            let [r, g, b, _] = mesh.color[i];
            let w = layer.weight[i];
            let weight = if w.is_finite() {
                w.clamp(0.0, 1.0)
            } else {
                0.0
            };
            out.push(Vertex {
                pos: [
                    x as f32 / cols as f32 * 2.0 - 1.0,
                    1.0 - y as f32 / rows as f32 * 2.0,
                ],
                uv: mesh.uv[i].map(finite_or_zero),
                color: [r, g, b, weight],
            });
        }
    }
}

/// Two triangles per cell, indices relative to the layer's first vertex.
fn push_indices(out: &mut Vec<u32>, cols: u32, rows: u32) {
    let stride = cols + 1;
    for y in 0..rows {
        for x in 0..cols {
            let v0 = y * stride + x;
            let (v1, v2, v3) = (v0 + 1, v0 + stride, v0 + stride + 1);
            out.extend_from_slice(&[v0, v2, v1, v1, v2, v3]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_layout() {
        assert_eq!(std::mem::size_of::<Uniforms>() % 16, 0);
        assert_eq!(std::mem::size_of::<Uniforms>(), 336);
        assert_eq!(std::mem::offset_of!(Uniforms, decay), 48);
        assert_eq!(std::mem::offset_of!(Uniforms, texsize), 96);
        assert_eq!(std::mem::offset_of!(Uniforms, blur_max), 192);
        assert_eq!(std::mem::offset_of!(Uniforms, q), 208);
    }

    #[test]
    fn uniforms_default() {
        let u = Uniforms::default();
        assert_eq!(u.decay, 0.98);
        assert_eq!((u.gamma, u.echo_zoom), (1.0, 1.0));
        assert_eq!((u.aspect_x, u.aspect_y), (1.0, 1.0));
        assert_eq!(u.blur_max, [1.0; 4]);
        assert_eq!(u.blur_min, [0.0; 4]);
        assert_eq!((u.time, u.echo_alpha, u.q), (0.0, 0.0, [[0.0; 4]; 8]));
    }

    #[test]
    fn indices_cover_every_cell() {
        let mut idx = Vec::new();
        push_indices(&mut idx, 2, 1);
        assert_eq!(idx, [0, 3, 1, 1, 3, 4, 1, 4, 2, 2, 4, 5]);
    }
}
