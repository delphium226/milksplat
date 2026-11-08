//! On-screen text (spec §6.3), drawn onto the display target after the
//! composite: `font8x8`'s basic Latin glyphs (`BASIC_LEGACY`, U+0020 to
//! U+007E) with a one-pixel black shadow. Any other character, including
//! control characters such as `\n`, is drawn as `?`.
//!
//! The 95 glyphs live in one `R8Unorm` atlas (16 x 6 cells of 8 x 8 texels,
//! 128 x 48), built once. Each visible glyph is a quad of `8 * scale` pixels
//! whose top-left corner is on a whole pixel, sampled with nearest
//! filtering, so every glyph pixel becomes an exact `scale` x `scale` block.
//! For each item the shadow (black, the item's alpha, offset by
//! `(+scale, +scale)`) is drawn first, then the text.
//!
//! Pipelines are cached per target format, created the first time a format
//! is seen (this is fixed, built-in WGSL, so the one-off creation on the
//! render thread is cheap; it is error-scoped and logged at debug). The
//! vertex buffer grows as needed and is reused; after warm-up a call
//! creates no GPU objects.

use wgpu::util::DeviceExt as _;

use crate::feedback::Renderer;
use crate::gpu::Gpu;
use crate::shaders::scoped;
use crate::warp::ensure_capacity;

/// Glyph cell size in atlas texels (and in pixels at scale 1).
const GLYPH: u32 = 8;
const ATLAS_COLS: u32 = 16;
const ATLAS_ROWS: u32 = 6;
const ATLAS_WIDTH: u32 = ATLAS_COLS * GLYPH;
const ATLAS_HEIGHT: u32 = ATLAS_ROWS * GLYPH;
const FIRST_CHAR: char = ' ';
const LAST_CHAR: char = '~';
const UNKNOWN_CHAR: char = '?';
/// Largest scale used; bigger values are clamped to it.
pub(crate) const MAX_SCALE: u32 = 64;
/// Most quads (shadows plus glyphs) drawn per call; the rest is dropped
/// with a warning.
const MAX_QUADS: usize = 1 << 16;

/// One line of on-screen text.
#[derive(Clone, Debug, PartialEq)]
pub struct TextItem {
    pub text: String,
    /// Top-left corner, in pixels from the target's top-left corner
    /// (rounded to whole pixels; an item with a non-finite position is
    /// skipped).
    pub pos: [f32; 2],
    /// Integer magnification: each glyph is `8 * scale` pixels square. 0 is
    /// treated as 1; values above 64 are clamped to 64.
    pub scale: u32,
    /// Straight-alpha RGBA, each channel clamped to [0, 1] (non-finite
    /// counts as 0). The shadow uses the same alpha.
    pub color: [f32; 4],
}

/// The atlas cell of `c`: its index in the basic Latin range, `?` for any
/// other character.
fn glyph_index(c: char) -> u32 {
    let c = if (FIRST_CHAR..=LAST_CHAR).contains(&c) {
        c
    } else {
        UNKNOWN_CHAR
    };
    c as u32 - FIRST_CHAR as u32
}

/// The atlas texels, row-major: 255 where a glyph pixel is set. In
/// `font8x8`, byte `r` of a glyph is row `r` and bit `b` is column `b`
/// (bit 0 is the leftmost pixel).
fn atlas_texels() -> Vec<u8> {
    let mut texels = vec![0u8; (ATLAS_WIDTH * ATLAS_HEIGHT) as usize];
    for (index, code) in (FIRST_CHAR as usize..=LAST_CHAR as usize).enumerate() {
        let glyph = font8x8::legacy::BASIC_LEGACY[code];
        let (cell_x, cell_y) = (index as u32 % ATLAS_COLS, index as u32 / ATLAS_COLS);
        for (row, bits) in (0u32..).zip(glyph) {
            for col in 0..GLYPH {
                if bits & (1 << col) != 0 {
                    let x = cell_x * GLYPH + col;
                    let y = cell_y * GLYPH + row;
                    texels[(y * ATLAS_WIDTH + x) as usize] = 255;
                }
            }
        }
    }
    texels
}

/// Text vertex: clip position, atlas UV, straight-alpha colour.
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

/// Colour blending onto the target: straight colour over the target
/// (`SrcAlpha`, `OneMinusSrcAlpha`), alpha composited the same way.
const TEXT_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

/// The atlas, its bind group, the shader, the per-format pipelines and the
/// reused vertex buffer.
pub(crate) struct TextState {
    /// Atlas view and nearest/clamp sampler (holds the atlas texture).
    bind_group: wgpu::BindGroup,
    pipeline_layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    /// Pipelines by target format; `None` = creation failed (not retried).
    pipelines: Vec<(wgpu::TextureFormat, Option<wgpu::RenderPipeline>)>,
    vertex_buffer: Option<wgpu::Buffer>,
    vertices: Vec<Vertex>,
    /// Pipelines created so far (test hook).
    pipeline_creations: usize,
}

impl TextState {
    /// Uploads the atlas and builds the bind group and shader module. Not
    /// error scoped, like the other internal passes: everything is fixed.
    pub(crate) fn new(gpu: &Gpu, sampler: &wgpu::Sampler) -> TextState {
        let device = &gpu.device;
        let atlas = device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("text atlas"),
                size: wgpu::Extent3d {
                    width: ATLAS_WIDTH,
                    height: ATLAS_HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &atlas_texels(),
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("text"),
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
        let view = atlas.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("text"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("text"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("text.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/text.wgsl").into()),
        });
        TextState {
            bind_group,
            pipeline_layout,
            module,
            pipelines: Vec::new(),
            vertex_buffer: None,
            vertices: Vec::new(),
            pipeline_creations: 0,
        }
    }

    /// The pipeline for `format`, created (inside error scopes) the first
    /// time the format is seen. `None` if it cannot be created (e.g. the
    /// format is not a blendable colour format); that is logged once.
    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> Option<wgpu::RenderPipeline> {
        if let Some((_, pipeline)) = self.pipelines.iter().find(|(f, _)| *f == format) {
            return pipeline.clone();
        }
        tracing::debug!("draw_text: creating the text pipeline for {format:?}");
        self.pipeline_creations += 1;
        let (pipeline, error) = scoped(device, || {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("text"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(Vertex::LAYOUT)],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(TEXT_BLEND),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        let pipeline = match error {
            None => Some(pipeline),
            Some(e) => {
                tracing::error!("draw_text: cannot draw text onto {format:?}: {e}");
                None
            }
        };
        self.pipelines.push((format, pipeline.clone()));
        pipeline
    }
}

/// Clamps to [0, 1]; non-finite counts as 0.
fn unit(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Builds glyph quads in target pixels and converts them to clip space.
struct Quads<'a> {
    out: &'a mut Vec<Vertex>,
    size: [f32; 2],
    count: usize,
    overflowed: bool,
}

impl Quads<'_> {
    /// One glyph cell, `cell` pixels square, at `[x, y]` (top-left).
    fn push(&mut self, [x, y]: [f32; 2], cell: f32, glyph: u32, color: [f32; 4]) {
        if self.count >= MAX_QUADS {
            self.overflowed = true;
            return;
        }
        self.count += 1;
        let (gx, gy) = (glyph % ATLAS_COLS, glyph / ATLAS_COLS);
        let u0 = (gx * GLYPH) as f32 / ATLAS_WIDTH as f32;
        let v0 = (gy * GLYPH) as f32 / ATLAS_HEIGHT as f32;
        let u1 = ((gx + 1) * GLYPH) as f32 / ATLAS_WIDTH as f32;
        let v1 = ((gy + 1) * GLYPH) as f32 / ATLAS_HEIGHT as f32;
        let [w, h] = self.size;
        let clip = |px: f32, py: f32| [px / w * 2.0 - 1.0, 1.0 - py / h * 2.0];
        let vertex = |px, py, u, v| Vertex {
            pos: clip(px, py),
            uv: [u, v],
            color,
        };
        let (x1, y1) = (x + cell, y + cell);
        let a = vertex(x, y, u0, v0);
        let b = vertex(x1, y, u1, v0);
        let c = vertex(x, y1, u0, v1);
        let d = vertex(x1, y1, u1, v1);
        self.out.extend_from_slice(&[a, c, b, b, c, d]);
    }

    /// The shadow, then the glyphs, of one item. Spaces and cells wholly
    /// outside the target add nothing.
    fn item(&mut self, item: &TextItem) {
        let [px, py] = item.pos;
        if !px.is_finite() || !py.is_finite() {
            tracing::debug!(
                "draw_text: skipping {:?} at a non-finite position",
                item.text
            );
            return;
        }
        let color = item.color.map(unit);
        if color[3] <= 0.0 {
            return;
        }
        let scale = item.scale.clamp(1, MAX_SCALE) as f32;
        let cell = GLYPH as f32 * scale;
        let origin = [px.round(), py.round()];
        let [w, h] = self.size;
        let shadow = [0.0, 0.0, 0.0, color[3]];
        for (offset, color) in [(scale, shadow), (0.0, color)] {
            let y = origin[1] + offset;
            if y >= h || y + cell <= 0.0 {
                continue;
            }
            for (i, c) in item.text.chars().enumerate() {
                let x = origin[0] + offset + i as f32 * cell;
                if x >= w || self.overflowed {
                    break;
                }
                if c == ' ' || x + cell <= 0.0 {
                    continue;
                }
                self.push([x, y], cell, glyph_index(c), color);
            }
        }
    }
}

impl Renderer {
    /// Draws `items` onto `target` (`target_size` pixels), over its contents,
    /// in order: for each item its one-pixel (times `scale`) black shadow,
    /// then its text. See [`TextItem`] for positions, scale and colour.
    ///
    /// The pipeline matches the format of `target`'s texture
    /// (`target.texture().format()`), so the view must not reinterpret the
    /// texture as another format. A zero-size target, or items with nothing
    /// visible, return at once without submitting any work. Never panics;
    /// does not block.
    pub fn draw_text(
        &mut self,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        target_size: [u32; 2],
        items: &[TextItem],
    ) {
        let [w, h] = target_size;
        if w == 0 || h == 0 || items.is_empty() {
            return;
        }
        let state = &mut self.text;
        state.vertices.clear();
        let mut quads = Quads {
            out: &mut state.vertices,
            size: [w as f32, h as f32],
            count: 0,
            overflowed: false,
        };
        for item in items {
            quads.item(item);
        }
        if quads.overflowed {
            tracing::warn!("draw_text: more than {MAX_QUADS} glyph quads; the rest is not drawn");
        }
        if state.vertices.is_empty() {
            return;
        }

        let Some(pipeline) = state.pipeline(&gpu.device, target.texture().format()) else {
            return;
        };
        let vertex_size = std::mem::size_of::<Vertex>() as u64;
        ensure_capacity(
            gpu,
            &mut state.vertex_buffer,
            state.vertices.len() as u64 * vertex_size,
            wgpu::BufferUsages::VERTEX,
            "text vertices",
        );
        let Some(vertex_buffer) = &state.vertex_buffer else {
            return;
        };
        gpu.queue
            .write_buffer(vertex_buffer, 0, bytemuck::cast_slice(&state.vertices));

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("text"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("text"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &state.bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            // At most MAX_QUADS * 6 vertices, which fits in u32.
            pass.draw(0..state.vertices.len() as u32, 0..1);
        }
        gpu.queue.submit([encoder.finish()]);
    }

    /// How many text pipelines have been created (one per target format
    /// seen; test hook).
    #[doc(hidden)]
    pub fn text_pipeline_creations(&self) -> usize {
        self.text.pipeline_creations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_characters_map_to_question_mark() {
        assert_eq!(glyph_index(' '), 0);
        assert_eq!(glyph_index('~'), 94);
        let question = glyph_index('?');
        for c in ['\u{2014}', '\n', '\u{0}', '\u{7f}', 'é', '\u{1F600}'] {
            assert_eq!(glyph_index(c), question, "{c:?}");
        }
    }

    #[test]
    fn atlas_holds_the_legacy_glyphs() {
        let texels = atlas_texels();
        assert_eq!(texels.len(), 128 * 48);
        // `A` (0x41) is cell 33: column 1, row 2 of the grid.
        let a = font8x8::legacy::BASIC_LEGACY[0x41];
        let index = glyph_index('A');
        assert_eq!(index, 33);
        let (cx, cy) = (index % ATLAS_COLS * GLYPH, index / ATLAS_COLS * GLYPH);
        for row in 0..8 {
            for col in 0..8 {
                let set = a[row as usize] & (1 << col) != 0;
                let texel = texels[((cy + row) * ATLAS_WIDTH + cx + col) as usize];
                assert_eq!(texel, if set { 255 } else { 0 }, "({col},{row})");
            }
        }
        // The space cell is empty.
        assert!((0..8).all(|row| texels[(row * ATLAS_WIDTH) as usize..][..8]
            .iter()
            .all(|&t| t == 0)));
    }
}
