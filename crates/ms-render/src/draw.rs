//! Overlay drawing into the feedback buffer (spec §6.2 step 3, §6.4): the
//! shared part. [`DrawList`] holds one tick's overlays; [`Renderer::draw`]
//! tessellates them on the CPU (`waves.rs`, `shapes.rs`, `borders.rs`,
//! `motion.rs`) into one reused vertex buffer and draws them in one render
//! pass that loads the current feedback image, in this order: motion
//! vectors, custom shapes, custom waves, the main waveform, the outer border,
//! the inner border.
//!
//! **Coordinates.** Every position is in MilkDrop's space: [0, 1] on both
//! axes with the origin at the bottom left. The vertex shader
//! (`shaders/overlay.wgsl`) maps it to clip space (`clip = 2p - 1`); texture
//! space (v = 0 at the top) is only used for the UVs of textured shapes.
//!
//! **Lines and dots** are quads (two triangles per segment). Widths are in
//! pixels of the feedback buffer, scaled with its height relative to 1080
//! rows and never below one pixel: thin lines 1 px, thick lines 2 px, dots
//! 2x2 px. Lines have no caps or joins.
//!
//! **Blending.** Normal draws are premultiplied-alpha "over" (`One`,
//! `OneMinusSrcAlpha`, the shader premultiplying), additive draws are
//! `SrcAlpha`, `One`. The alpha channel uses `Zero`, `One`, so the feedback
//! buffer's alpha stays 1, as after the warp pass.
//!
//! **Sanitising.** Every incoming float is checked: NaN and infinities
//! become 0, colours are clamped to [0, 1] and positions to [-10, 10];
//! counts are capped (see the constants). Bad data never panics.

use crate::borders::{push_borders, Borders};
use crate::feedback::{FeedbackFormat, Renderer};
use crate::gpu::Gpu;
use crate::motion::{push_motion, MotionVectors};
use crate::shapes::{push_shape, ShapeDraw};
use crate::warp::ensure_capacity;
use crate::waves::{push_main_wave, push_wave, MainWave, WaveDraw};

/// Most custom waves drawn per [`Renderer::draw`] (four per preset; a
/// transition draws each preset separately). Extra waves are ignored.
pub(crate) const MAX_WAVES: usize = 64;
/// Most points used per custom wave (spec §6.4).
pub(crate) const MAX_WAVE_POINTS: usize = 512;
/// Most shape instances drawn per [`Renderer::draw`]: four shapes of up to
/// 1024 instances each.
pub(crate) const MAX_SHAPES: usize = 4 * 1024;
/// Upper bound on the vertex buffer, whatever the device allows (the worst
/// case of the caps above is about 3.9 million vertices).
const MAX_VERTICES: u64 = 1 << 22;

/// The overlays for one tick, drawn by [`Renderer::draw`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DrawList {
    pub motion: Option<MotionVectors>,
    /// One entry per shape instance, in drawing order.
    pub shapes: Vec<ShapeDraw>,
    pub waves: Vec<WaveDraw>,
    pub main_wave: Option<MainWave>,
    pub borders: Borders,
}

/// How a run of triangles is blended into the feedback buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Blend {
    /// Premultiplied-alpha over.
    Normal,
    /// `src * srcAlpha + dst`.
    Additive,
}

impl Blend {
    pub(crate) fn from_additive(additive: bool) -> Blend {
        if additive {
            Blend::Additive
        } else {
            Blend::Normal
        }
    }

    fn index(self) -> usize {
        match self {
            Blend::Normal => 0,
            Blend::Additive => 1,
        }
    }

    const ALL: [Blend; 2] = [Blend::Normal, Blend::Additive];
}

/// One overlay vertex. `color` is straight (not premultiplied) alpha;
/// `textured` is 1 for textured shapes, whose colour multiplies the previous
/// feedback image sampled at `uv` (texture space), and 0 otherwise.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    pub(crate) pos: [f32; 2],
    pub(crate) uv: [f32; 2],
    pub(crate) color: [f32; 4],
    pub(crate) textured: f32,
}

impl Vertex {
    /// An untextured vertex.
    pub(crate) fn flat(pos: [f32; 2], color: [f32; 4]) -> Vertex {
        Vertex {
            pos,
            uv: [0.0; 2],
            color,
            textured: 0.0,
        }
    }

    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Float32
    ];
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Vertex::ATTRIBUTES,
    };
}

/// A run of consecutive vertices drawn with one blend mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Batch {
    blend: Blend,
    start: u32,
    end: u32,
}

/// CPU-side tessellation target: triangles plus the blend-mode runs, and the
/// target's pixel metrics.
pub(crate) struct Geometry {
    vertices: Vec<Vertex>,
    batches: Vec<Batch>,
    /// Feedback size in pixels.
    size: [f32; 2],
    /// `[aspect_x, aspect_y]`, see [`aspect_of`].
    pub(crate) aspect: [f32; 2],
    /// Global alpha multiplier, in [0, 1].
    pub(crate) alpha: f32,
    /// Line widths and dot size, in pixels.
    thin: f32,
    thick: f32,
    dot: f32,
    max_vertices: usize,
    /// Set when triangles were dropped because the buffer limit was hit.
    overflowed: bool,
}

impl Geometry {
    fn new() -> Geometry {
        Geometry {
            vertices: Vec::new(),
            batches: Vec::new(),
            size: [1.0; 2],
            aspect: [1.0; 2],
            alpha: 1.0,
            thin: 1.0,
            thick: 1.0,
            dot: 1.0,
            max_vertices: 0,
            overflowed: false,
        }
    }

    /// Empties the geometry for a `size` target (pixels).
    fn reset(&mut self, size: [u32; 2], alpha: f32, max_vertices: usize) {
        self.vertices.clear();
        self.batches.clear();
        self.size = [size[0].max(1) as f32, size[1].max(1) as f32];
        self.aspect = aspect_of(size);
        self.alpha = alpha;
        let scale = self.size[1] / 1080.0;
        self.thin = scale.max(1.0);
        self.thick = (2.0 * scale).max(1.0);
        self.dot = (2.0 * scale).max(1.0);
        self.max_vertices = max_vertices;
        self.overflowed = false;
    }

    /// An empty geometry for a `size` target at alpha 1 (unit tests).
    #[cfg(test)]
    pub(crate) fn for_tests(size: [u32; 2]) -> Geometry {
        let mut g = Geometry::new();
        g.reset(size, 1.0, 1 << 20);
        g
    }

    #[cfg(test)]
    pub(crate) fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// A sanitised colour with its alpha scaled by the global alpha.
    pub(crate) fn color(&self, c: [f32; 4]) -> [f32; 4] {
        let [r, g, b, a] = c.map(unit);
        [r, g, b, a * self.alpha]
    }

    /// Appends one triangle.
    pub(crate) fn tri(&mut self, blend: Blend, v: [Vertex; 3]) {
        if self.vertices.len() + 3 > self.max_vertices {
            self.overflowed = true;
            return;
        }
        let start = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&v);
        let end = start + 3;
        match self.batches.last_mut() {
            Some(last) if last.blend == blend && last.end == start => last.end = end,
            _ => self.batches.push(Batch { blend, start, end }),
        }
    }

    /// Appends a quad `a b c d` (in order around its edge) as two triangles.
    pub(crate) fn quad(&mut self, blend: Blend, [a, b, c, d]: [Vertex; 4]) {
        self.tri(blend, [a, b, c]);
        self.tri(blend, [a, c, d]);
    }

    /// Axis-aligned rectangle from `lo` to `hi` (MilkDrop coordinates).
    pub(crate) fn rect(&mut self, blend: Blend, lo: [f32; 2], hi: [f32; 2], color: [f32; 4]) {
        self.quad(
            blend,
            [
                Vertex::flat(lo, color),
                Vertex::flat([hi[0], lo[1]], color),
                Vertex::flat(hi, color),
                Vertex::flat([lo[0], hi[1]], color),
            ],
        );
    }

    /// A line segment from `p0` (colour `c0`) to `p1` (colour `c1`), as a
    /// quad `width` pixels wide. Zero-length and fully transparent segments
    /// are skipped. Colours must already be sanitised.
    pub(crate) fn segment(
        &mut self,
        blend: Blend,
        (p0, c0): ([f32; 2], [f32; 4]),
        (p1, c1): ([f32; 2], [f32; 4]),
        thick: bool,
    ) {
        if c0[3] <= 0.0 && c1[3] <= 0.0 {
            return;
        }
        let width = if thick { self.thick } else { self.thin };
        let [w, h] = self.size;
        // Direction in pixels, so the width is the same in x and y.
        let d = [(p1[0] - p0[0]) * w, (p1[1] - p0[1]) * h];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if !len.is_finite() || len <= 1e-4 {
            return;
        }
        let k = 0.5 * width / len;
        let n = [-d[1] * k / w, d[0] * k / h];
        let v = |p: [f32; 2], s: f32, c| Vertex::flat([p[0] + s * n[0], p[1] + s * n[1]], c);
        self.quad(
            blend,
            [
                v(p0, 1.0, c0),
                v(p0, -1.0, c0),
                v(p1, -1.0, c1),
                v(p1, 1.0, c1),
            ],
        );
    }

    /// Lines through `points` in order.
    pub(crate) fn polyline(&mut self, blend: Blend, points: &[([f32; 2], [f32; 4])], thick: bool) {
        for pair in points.windows(2) {
            self.segment(blend, pair[0], pair[1], thick);
        }
    }

    /// A square dot centred on `p`.
    pub(crate) fn dot(&mut self, blend: Blend, p: [f32; 2], color: [f32; 4]) {
        if color[3] <= 0.0 {
            return;
        }
        let half = [0.5 * self.dot / self.size[0], 0.5 * self.dot / self.size[1]];
        self.rect(
            blend,
            [p[0] - half[0], p[1] - half[1]],
            [p[0] + half[0], p[1] + half[1]],
            color,
        );
    }
}

/// `[aspect_x, aspect_y]` for a `[width, height]` target: `aspect_x` is
/// `width / height` when the target is wider than tall (else 1) and
/// `aspect_y` is `height / width` when it is taller than wide (else 1).
/// Radial overlays divide their x offsets by `aspect_x` and their y offsets
/// by `aspect_y`, so circles stay round on screen.
pub(crate) fn aspect_of([w, h]: [u32; 2]) -> [f32; 2] {
    let (w, h) = (w.max(1) as f32, h.max(1) as f32);
    if w >= h {
        [w / h, 1.0]
    } else {
        [1.0, h / w]
    }
}

/// `v`, or 0 if it is NaN or infinite.
pub(crate) fn finite(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// A colour channel or alpha: finite and in [0, 1].
pub(crate) fn unit(v: f32) -> f32 {
    finite(v).clamp(0.0, 1.0)
}

/// A position or offset: finite and in [-10, 10].
pub(crate) fn coord(v: f32) -> f32 {
    finite(v).clamp(-10.0, 10.0)
}

/// A sanitised point.
pub(crate) fn point(p: [f32; 2]) -> [f32; 2] {
    p.map(coord)
}

/// The overlay pipelines, the reused vertex buffer and the cached bind
/// groups.
pub(crate) struct OverlayState {
    layout: wgpu::BindGroupLayout,
    /// `[feedback format][blend]`.
    pipelines: [[wgpu::RenderPipeline; 2]; 2],
    vertex_buffer: Option<wgpu::Buffer>,
    /// Bind group sampling feedback texture 0 or 1 (the previous frame).
    bind_groups: [Option<wgpu::BindGroup>; 2],
    geometry: Geometry,
}

impl OverlayState {
    /// Builds the pipelines. Like the other internal passes, not error
    /// scoped: the WGSL and the descriptors are fixed, so a failure would
    /// be a bug, reported (and logged) through the uncaptured-error
    /// handler.
    pub(crate) fn new(gpu: &Gpu) -> OverlayState {
        let device = &gpu.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay"),
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlay.wgsl").into()),
        });
        let pipelines = FeedbackFormat::ALL.map(|format| {
            Blend::ALL.map(|blend| {
                let (color, premultiply) = match blend {
                    Blend::Normal => (
                        wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        1.0,
                    ),
                    Blend::Additive => (
                        wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        0.0,
                    ),
                };
                let constants = [("ms_premultiply", premultiply)];
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("overlay"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[Some(Vertex::LAYOUT)],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &constants,
                            ..Default::default()
                        },
                        targets: &[Some(wgpu::ColorTargetState {
                            format: format.texture_format(),
                            blend: Some(wgpu::BlendState {
                                color,
                                // Keep the target's alpha (1).
                                alpha: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::Zero,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            })
        });
        OverlayState {
            layout,
            pipelines,
            vertex_buffer: None,
            bind_groups: [None, None],
            geometry: Geometry::new(),
        }
    }

    /// Drops the bind groups that reference the feedback views.
    pub(crate) fn invalidate_bind_groups(&mut self) {
        self.bind_groups = [None, None];
    }
}

/// Tessellates `list` into `g`, in the drawing order of spec §6.2.
fn tessellate(g: &mut Geometry, list: &DrawList) {
    if let Some(mv) = &list.motion {
        push_motion(g, mv);
    }
    for shape in list.shapes.iter().take(MAX_SHAPES) {
        push_shape(g, shape);
    }
    for wave in list.waves.iter().take(MAX_WAVES) {
        push_wave(g, wave);
    }
    if let Some(main) = &list.main_wave {
        push_main_wave(g, main);
    }
    push_borders(g, &list.borders);
}

impl Renderer {
    /// Draws `list` into the current feedback buffer (the one the warp pass
    /// just wrote), on top of its contents, in the order of spec §6.2:
    /// motion vectors, shapes, custom waves, the main waveform, the outer
    /// border, the inner border. Every colour's alpha is multiplied by
    /// `alpha` (clamped to [0, 1]; NaN counts as 0), which is how a
    /// transition fades a preset's overlays.
    ///
    /// Textured shapes sample the previous feedback buffer (the one the
    /// last warp read from), never the buffer being drawn into.
    ///
    /// Inputs are sanitised (see the module docs); nothing here panics.
    /// One vertex upload, one render pass, one submit; does not block.
    pub fn draw(&mut self, gpu: &Gpu, list: &DrawList, alpha: f32) {
        let alpha = unit(alpha);
        if alpha <= 0.0 {
            return;
        }
        let vertex_size = std::mem::size_of::<Vertex>() as u64;
        let max_vertices = (gpu.device.limits().max_buffer_size / vertex_size).min(MAX_VERTICES);
        let state = &mut self.overlay;
        let g = &mut state.geometry;
        g.reset(self.size, alpha, max_vertices as usize);
        tessellate(g, list);
        if g.overflowed {
            tracing::warn!(
                "draw: overlay geometry exceeds {max_vertices} vertices; the rest is not drawn"
            );
        }
        if g.vertices.is_empty() {
            return;
        }

        ensure_capacity(
            gpu,
            &mut state.vertex_buffer,
            g.vertices.len() as u64 * vertex_size,
            wgpu::BufferUsages::VERTEX,
            "overlay vertices",
        );
        let Some(vertex_buffer) = &state.vertex_buffer else {
            return;
        };
        gpu.queue
            .write_buffer(vertex_buffer, 0, bytemuck::cast_slice(&g.vertices));

        let previous = 1 - self.current;
        let bind_group = state.bind_groups[previous].get_or_insert_with(|| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("overlay"),
                layout: &state.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.views[previous]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.shared.samplers.fw),
                    },
                ],
            })
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("overlay"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.views[self.current],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_bind_group(0, &*bind_group, &[]);
            let pipelines = &state.pipelines[self.format.index()];
            for batch in &g.batches {
                pass.set_pipeline(&pipelines[batch.blend.index()]);
                pass.draw(batch.start..batch.end, 0..1);
            }
        }
        gpu.queue.submit([encoder.finish()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(size: [u32; 2]) -> Geometry {
        Geometry::for_tests(size)
    }

    #[test]
    fn vertex_layout() {
        assert_eq!(std::mem::size_of::<Vertex>(), 36);
    }

    #[test]
    fn aspect_convention() {
        assert_eq!(aspect_of([1920, 1080]), [1920.0 / 1080.0, 1.0]);
        assert_eq!(aspect_of([100, 200]), [1.0, 2.0]);
        assert_eq!(aspect_of([64, 64]), [1.0, 1.0]);
    }

    #[test]
    fn batches_merge_runs_of_one_blend() {
        let mut g = geometry([64, 64]);
        let v = Vertex::flat([0.0; 2], [1.0; 4]);
        g.tri(Blend::Normal, [v; 3]);
        g.tri(Blend::Normal, [v; 3]);
        g.tri(Blend::Additive, [v; 3]);
        g.tri(Blend::Normal, [v; 3]);
        let runs: Vec<(Blend, u32, u32)> = g
            .batches
            .iter()
            .map(|b| (b.blend, b.start, b.end))
            .collect();
        assert_eq!(
            runs,
            [
                (Blend::Normal, 0, 6),
                (Blend::Additive, 6, 9),
                (Blend::Normal, 9, 12)
            ]
        );
    }

    #[test]
    fn widths_scale_with_height() {
        let g = geometry([64, 1080]);
        assert_eq!((g.thin, g.thick, g.dot), (1.0, 2.0, 2.0));
        let g = geometry([3840, 2160]);
        assert_eq!((g.thin, g.thick, g.dot), (2.0, 4.0, 4.0));
        let g = geometry([128, 128]);
        assert_eq!((g.thin, g.thick, g.dot), (1.0, 1.0, 1.0));
    }

    #[test]
    fn segment_quad_is_width_pixels_wide() {
        let mut g = geometry([100, 1080]);
        let c = [1.0; 4];
        g.segment(Blend::Normal, ([0.0, 0.5], c), ([1.0, 0.5], c), true);
        let ys: Vec<f32> = g.vertices.iter().map(|v| v.pos[1] * 1080.0).collect();
        let (lo, hi) = ys
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &y| (lo.min(y), hi.max(y)));
        assert!((hi - lo - 2.0).abs() < 1e-3, "{lo}..{hi}");
        // Degenerate and transparent segments add nothing.
        let n = g.vertices.len();
        g.segment(Blend::Normal, ([0.5, 0.5], c), ([0.5, 0.5], c), true);
        let clear = [1.0, 1.0, 1.0, 0.0];
        g.segment(
            Blend::Normal,
            ([0.0, 0.0], clear),
            ([1.0, 1.0], clear),
            true,
        );
        assert_eq!(g.vertices.len(), n);
    }

    #[test]
    fn overflow_drops_triangles() {
        let mut g = Geometry::new();
        g.reset([64, 64], 1.0, 4);
        let v = Vertex::flat([0.0; 2], [1.0; 4]);
        g.tri(Blend::Normal, [v; 3]);
        g.tri(Blend::Normal, [v; 3]);
        assert_eq!(g.vertices.len(), 3);
        assert!(g.overflowed);
    }

    #[test]
    fn sanitisers() {
        assert_eq!(unit(f32::NAN), 0.0);
        assert_eq!(unit(2.0), 1.0);
        assert_eq!(coord(f32::NEG_INFINITY), 0.0);
        assert_eq!(coord(1e9), 10.0);
        assert_eq!(point([f32::NAN, -20.0]), [0.0, -10.0]);
    }
}
