//! The warp pass: redraws the previous feedback image through the warp mesh
//! into the other ping-pong buffer, blending one or more weighted layers.

use crate::feedback::{FeedbackFormat, Renderer};
use crate::gpu::Gpu;
use crate::shaders::{self, Uniforms};

/// Largest mesh accepted, in vertices per layer (the quality setting caps
/// real meshes at 193 * 145); bigger layers are skipped with an error.
const MAX_MESH_VERTICES: u64 = 1 << 20;

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

/// A compiled warp shader (built-in or a preset's): one pipeline per
/// feedback format, plus the preset's user-texture bind group (group 1), if
/// any. Cheap to clone; `Send + Sync`.
///
/// Belongs to the [`RendererShared`](crate::RendererShared) generation that
/// compiled it; after [`Renderer::recreate`] it is stale and
/// [`Renderer::warp`] draws the built-in warp in its place.
#[derive(Clone)]
pub struct WarpShader {
    pub(crate) pipelines: [wgpu::RenderPipeline; 2],
    pub(crate) group1: Option<wgpu::BindGroup>,
    /// The `RendererShared` generation that compiled this shader.
    pub(crate) generation: u64,
}

impl WarpShader {
    /// The built-in warp: `GetMain(uv) * decay * colour.rgb` (linear
    /// filtering, wrap addressing). Shared, not recompiled. Like every
    /// shader it is stale after [`Renderer::recreate`]: fetch it again.
    pub fn builtin(_gpu: &Gpu, renderer: &Renderer) -> WarpShader {
        renderer.shared.builtin_warp.clone()
    }
}

/// Builds the per-format warp pipelines from a module with `vs_main` and
/// `fs_main` entry points (see `shaders/warp_stage.wgsl`).
pub(crate) fn create_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    label: &str,
) -> [wgpu::RenderPipeline; 2] {
    FeedbackFormat::ALL.map(|format| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
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
    })
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
pub(crate) struct Vertex {
    pub(crate) pos: [f32; 2],
    pub(crate) uv: [f32; 2],
    pub(crate) color: [f32; 4],
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

/// A layer position's uniform buffer and its group 0 bind groups (one per
/// feedback texture bound as `sampler_main`). Shared with the composite pass.
pub(crate) struct LayerSlot {
    pub(crate) uniforms: wgpu::Buffer,
    /// Bind group sampling feedback texture 0 or 1.
    pub(crate) bind_groups: [Option<wgpu::BindGroup>; 2],
}

/// Grows `slots` to at least `n` entries.
pub(crate) fn ensure_slots(gpu: &Gpu, slots: &mut Vec<LayerSlot>, n: usize, label: &str) {
    while slots.len() < n {
        slots.push(LayerSlot {
            uniforms: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bind_groups: [None, None],
        });
    }
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
pub(crate) fn ensure_capacity(
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
    /// or oversized meshes) are skipped with an error log. A stale shader
    /// (compiled before the last [`Renderer::recreate`]) is replaced by the
    /// built-in warp, with an error log. Does not block.
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
        ensure_slots(gpu, &mut state.slots, draws.len(), "warp uniforms");
        for (draw, slot) in draws.iter().zip(&mut state.slots) {
            gpu.queue
                .write_buffer(&slot.uniforms, 0, bytemuck::bytes_of(draw.layer.uniforms));
            if slot.bind_groups[src].is_none() {
                slot.bind_groups[src] = Some(shaders::group0_bind_group(
                    &self.shared,
                    &self.aux,
                    &slot.uniforms,
                    &self.views[src],
                    "warp group 0",
                ));
            }
        }

        let mut stale = false;
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
                    let mut shader = draw.layer.shader;
                    if shader.generation != self.shared.generation {
                        // Its objects belong to an old device.
                        stale = true;
                        shader = &self.shared.builtin_warp;
                    }
                    pass.set_pipeline(&shader.pipelines[self.format.index()]);
                    pass.set_bind_group(0, bind_group, &[]);
                    if let Some(group1) = &shader.group1 {
                        pass.set_bind_group(1, group1, &[]);
                    }
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
        if stale {
            self.report_stale("warp");
        }
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
pub(crate) fn push_indices(out: &mut Vec<u32>, cols: u32, rows: u32) {
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
    fn indices_cover_every_cell() {
        let mut idx = Vec::new();
        push_indices(&mut idx, 2, 1);
        assert_eq!(idx, [0, 3, 1, 1, 3, 4, 1, 4, 2, 2, 4, 5]);
    }
}
