//! The shader interface (spec §6.5): the uniform block, the prelude, the
//! shared group 0 layout, and compilation of preset warp/composite shaders
//! with fallback to the built-ins.
//!
//! A preset shader is a WGSL function *body*. It is compiled as
//! `prelude + group 1 declarations + stage wrapper + fn user_X(..) { body }`
//! (see `shaders/prelude.wgsl`). Any failure (naga parse or validation, or a
//! wgpu error while creating the pipeline) makes that stage fall back to the
//! built-in shader and adds a message to [`ShaderSet::errors`]; nothing
//! reachable from shader source or texture images panics.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

use image::RgbaImage;

use crate::composite::CompShader;
use crate::feedback::Renderer;
use crate::gpu::Gpu;
use crate::shader_scan::{self, SamplerKind, Scan, MAX_USER_TEXTURES};
use crate::warp::{self, WarpShader};

/// The shader prelude: the uniform block, the group 0 bindings and the helper
/// functions every warp and composite shader can use.
pub const PRELUDE_WGSL: &str = include_str!("shaders/prelude.wgsl");

const WARP_STAGE_WGSL: &str = include_str!("shaders/warp_stage.wgsl");
const COMP_STAGE_WGSL: &str = include_str!("shaders/comp_stage.wgsl");
const WARP_BUILTIN_BODY: &str = include_str!("shaders/warp_builtin.wgsl");
const COMP_BUILTIN_WGSL: &str = include_str!("shaders/comp_builtin.wgsl");
const COMP_BUILTIN_BODY: &str = "return ms_comp_builtin(uv);";

/// The uniform block shared by warp and composite shaders (std140 layout:
/// every row is 16 bytes).
///
/// Field order is part of the shader interface; the WGSL `Uniforms` struct
/// in the prelude must match it exactly (checked by the `uniforms_layout`
/// test with naga's layouter).
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

/// The result of compiling a preset's shaders. Each stage is the preset's
/// shader or, if it had none or it failed, the built-in one.
pub struct ShaderSet {
    pub warp: WarpShader,
    pub comp: CompShader,
    /// One message per problem (compile errors with line numbers relative
    /// to the preset's shader body, missing textures, too many textures).
    pub errors: Vec<String>,
    /// Highest blur level (0-3) either stage references.
    pub blur_levels: u8,
    /// User texture binding names (e.g. `fc_checker`) in group 1 slot order.
    pub user_textures: Vec<String>,
    /// The warp body this set was compiled from.
    pub warp_source: Option<String>,
    /// The composite body this set was compiled from.
    pub comp_source: Option<String>,
    /// The user texture images that were found, by file name (the binding
    /// name without its prefix), so the set can be compiled again (e.g.
    /// after a device loss) without reloading files.
    pub texture_images: HashMap<String, Arc<RgbaImage>>,
}

/// The four sampler kinds, shared by group 0 (`s_fw`..`s_pc`) and the user
/// textures in group 1.
#[derive(Clone)]
pub(crate) struct Samplers {
    pub(crate) fw: wgpu::Sampler,
    pub(crate) fc: wgpu::Sampler,
    pub(crate) pw: wgpu::Sampler,
    pub(crate) pc: wgpu::Sampler,
}

impl Samplers {
    fn new(device: &wgpu::Device) -> Self {
        let sampler = |label, filter, address_mode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: address_mode,
                address_mode_v: address_mode,
                address_mode_w: address_mode,
                mag_filter: filter,
                min_filter: filter,
                ..Default::default()
            })
        };
        use wgpu::{AddressMode as A, FilterMode as F};
        Samplers {
            fw: sampler("s_fw", F::Linear, A::Repeat),
            fc: sampler("s_fc", F::Linear, A::ClampToEdge),
            pw: sampler("s_pw", F::Nearest, A::Repeat),
            pc: sampler("s_pc", F::Nearest, A::ClampToEdge),
        }
    }

    fn get(&self, kind: SamplerKind) -> &wgpu::Sampler {
        match kind {
            SamplerKind::LinearWrap => &self.fw,
            SamplerKind::LinearClamp => &self.fc,
            SamplerKind::PointWrap => &self.pw,
            SamplerKind::PointClamp => &self.pc,
        }
    }
}

/// The textures bound at group 0 bindings 2-10: the blur levels, the noise
/// textures and the spectrum.
///
/// [`Renderer::new`] fills them from the blur chain, the noise set and the
/// spectrum texture; whoever replaces a view must call
/// `Renderer::invalidate_bind_groups` (the cached group 0 bind groups
/// reference these views). Every view must be a filterable float texture:
/// 2D, except `noisevol` (3D).
#[derive(Clone)]
pub(crate) struct AuxViews {
    pub(crate) blur: [wgpu::TextureView; 3],
    pub(crate) noise: [wgpu::TextureView; 3],
    pub(crate) noisevol: [wgpu::TextureView; 2],
    pub(crate) spectrum: wgpu::TextureView,
}

/// A 1x1 `Rgba8Unorm` texture holding `rgba`.
fn placeholder(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    rgba: [u8; 4],
) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        tex.as_image_copy(),
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        tex.size(),
    );
    tex.create_view(&Default::default())
}

/// Everything shader compilation needs, without the feedback textures:
/// cheap to clone, `Send + Sync`, so a loader thread can compile presets
/// while the render thread keeps the [`Renderer`].
///
/// wgpu handles (device, queue, layouts, samplers, views, pipelines) are
/// reference-counted internally, so every field is an `Arc`-like clone.
#[derive(Clone)]
pub struct RendererShared {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// Group 0 (the prelude's bindings 0-14), shared by every warp and
    /// composite shader.
    pub(crate) group0_layout: wgpu::BindGroupLayout,
    /// `[group 0]`, for shaders without user textures.
    pub(crate) base_pipeline_layout: wgpu::PipelineLayout,
    pub(crate) samplers: Samplers,
    /// 1x1 white, bound for user textures whose file was not found.
    white: wgpu::TextureView,
    pub(crate) builtin_warp: WarpShader,
    pub(crate) builtin_comp: CompShader,
}

/// The group 1 objects for one shader set's user textures.
struct Group1 {
    pipeline_layout: wgpu::PipelineLayout,
    bind_group: wgpu::BindGroup,
}

/// Which shader a body is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    Warp,
    Comp,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Warp => "warp",
            Stage::Comp => "comp",
        }
    }

    fn wrapper(self) -> &'static str {
        match self {
            Stage::Warp => WARP_STAGE_WGSL,
            Stage::Comp => COMP_STAGE_WGSL,
        }
    }

    /// The opening line of the user function; the body follows on the next
    /// line.
    fn signature(self) -> &'static str {
        match self {
            Stage::Warp => {
                "fn user_warp(uv: vec2f, uv_orig: vec2f, rad: f32, ang: f32, hue_shader: vec3f) -> vec3f {\n"
            }
            Stage::Comp => "fn user_comp(uv: vec2f, rad: f32, ang: f32, hue_shader: vec3f) -> vec3f {\n",
        }
    }
}

/// A complete shader module source built around a body.
pub(crate) struct Assembled {
    pub(crate) source: String,
    /// Lines before the body: line `n` of the module is line
    /// `n - prefix_lines` of the body.
    prefix_lines: usize,
    body_lines: usize,
}

/// Builds `prelude + group 1 declarations + extra + stage wrapper +
/// fn user_X(..) {\n body \n}`.
pub(crate) fn assemble(
    stage: Stage,
    extra: &str,
    user_textures: &[String],
    body: &str,
) -> Assembled {
    let mut source = String::with_capacity(
        PRELUDE_WGSL.len() + extra.len() + stage.wrapper().len() + body.len() + 1024,
    );
    let mut push = |text: &str| {
        source.push_str(text);
        if !text.ends_with('\n') {
            source.push('\n');
        }
    };
    push(PRELUDE_WGSL);
    let mut decls = String::new();
    for (i, name) in user_textures.iter().enumerate() {
        // Writing to a String cannot fail.
        let _ = writeln!(
            decls,
            "@group(1) @binding({}) var sampler_{name}: texture_2d<f32>;\n\
             @group(1) @binding({}) var sampler_{name}_s: sampler;",
            2 * i,
            2 * i + 1
        );
    }
    push(&decls);
    push(extra);
    push(stage.wrapper());
    source.push_str(stage.signature());
    let prefix_lines = source.matches('\n').count();
    source.push_str(body);
    source.push_str("\n}\n");
    Assembled {
        source,
        prefix_lines,
        body_lines: body.matches('\n').count() + 1,
    }
}

impl Assembled {
    /// The body-relative (line, column) of a byte offset in the module, or
    /// `None` if it lies outside the body (in generated code).
    fn body_position(&self, offset: usize) -> Option<(usize, usize)> {
        let prefix = self.source.get(..offset)?;
        let line = prefix.matches('\n').count() + 1;
        let line_start = prefix.rfind('\n').map_or(0, |i| i + 1);
        let body_line = line.checked_sub(self.prefix_lines)?;
        (1..=self.body_lines)
            .contains(&body_line)
            .then_some((body_line, offset - line_start + 1))
    }

    fn describe_position(&self, range: &Option<Range<usize>>) -> String {
        match range.as_ref().and_then(|r| self.body_position(r.start)) {
            Some((line, column)) => format!("line {line}, column {column}"),
            None => "generated code".to_string(),
        }
    }

    fn in_body(&self, range: &Option<Range<usize>>) -> bool {
        range
            .as_ref()
            .is_some_and(|r| self.body_position(r.start).is_some())
    }
}

/// Formats a naga error as `"<stage> shader, line L, column C: message
/// (line L, column C: label) ...; note: ..."`.
///
/// Positions are relative to the user's body: naga reports module line `n`,
/// and the body starts after [`Assembled::prefix_lines`] generated lines, so
/// the body line is `n - prefix_lines` (columns are unchanged, in bytes).
/// The headline position is the first label inside the body (naga's first
/// label is often the enclosing function, in generated code). Labels naga
/// uses only to name IR handles (`naga::ir::...`) are left out.
fn describe(
    stage: Stage,
    a: &Assembled,
    message: &str,
    labels: &[(Option<Range<usize>>, String)],
    notes: &[String],
) -> String {
    let mut out = format!("{} shader", stage.name());
    let primary = labels
        .iter()
        .position(|(range, _)| a.in_body(range))
        .or(if labels.is_empty() { None } else { Some(0) });
    if let Some(i) = primary {
        let _ = write!(out, ", {}", a.describe_position(&labels[i].0));
    }
    let _ = write!(out, ": {message}");
    for (range, label) in labels {
        if !label.is_empty() && !label.starts_with("naga::") {
            let _ = write!(out, " ({}: {label})", a.describe_position(range));
        }
    }
    for note in notes {
        let _ = write!(out, "; note: {note}");
    }
    out
}

/// Parses and validates `a` with naga, describing the first error.
fn naga_check(stage: Stage, a: &Assembled) -> Result<(), String> {
    let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let module = naga::front::wgsl::parse_str(&a.source).map_err(|e| {
            let labels: Vec<_> = e
                .labels()
                .map(|(span, label)| (span.to_range(), label.to_string()))
                .collect();
            let notes: Vec<_> = e.notes().map(str::to_string).collect();
            describe(stage, a, e.message(), &labels, &notes)
        })?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .map_err(|e| {
            let mut message = e.as_inner().to_string();
            let mut source = std::error::Error::source(e.as_inner());
            while let Some(next) = source {
                let _ = write!(message, ": {next}");
                source = next.source();
            }
            let labels: Vec<_> = e
                .spans()
                .map(|(span, label)| (span.to_range(), label.clone()))
                .collect();
            describe(stage, a, &message, &labels, &[])
        })?;
        Ok(())
    }));
    checked.unwrap_or_else(|_| {
        Err(format!(
            "{} shader: the shader compiler panicked",
            stage.name()
        ))
    })
}

/// Front-end checks for one stage: every user texture it uses has a slot,
/// and the assembled module (with `slots` declared) passes naga.
fn check_stage(
    stage: Stage,
    body: &str,
    scan: &Scan,
    slots: &[String],
    overflow: &[String],
) -> Result<(), String> {
    let unbound: Vec<&str> = scan
        .textures
        .iter()
        .filter(|t| overflow.contains(t))
        .map(String::as_str)
        .collect();
    if !unbound.is_empty() {
        return Err(format!(
            "{} shader: more than {MAX_USER_TEXTURES} user textures; no slot for {}",
            stage.name(),
            unbound.join(", ")
        ));
    }
    naga_check(stage, &assemble(stage, "", slots, body))
}

/// Runs `f` inside this thread's Validation, OutOfMemory and Internal error
/// scopes and returns its result with the first error captured, if any.
///
/// wgpu 30 error scopes are thread-local (with its default `std` feature),
/// so this only sees errors from work done on the calling thread, and the
/// scopes are always popped in reverse order (a mismatched pop panics).
pub(crate) fn scoped<T>(device: &wgpu::Device, f: impl FnOnce() -> T) -> (T, Option<String>) {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let value = f();
    let internal = pollster::block_on(internal.pop());
    let oom = pollster::block_on(oom.pop());
    let validation = pollster::block_on(validation.pop());
    let error = [validation, oom, internal]
        .into_iter()
        .flatten()
        .next()
        .map(|e| match e {
            // `Display` of these is only "Validation Error"/"Internal Error";
            // the description carries the detail.
            wgpu::Error::Validation { description, .. }
            | wgpu::Error::Internal { description, .. } => description,
            other @ wgpu::Error::OutOfMemory { .. } => other.to_string(),
        });
    (value, error)
}

impl RendererShared {
    pub(crate) fn new(gpu: &Gpu) -> Self {
        let device = gpu.device.clone();
        let samplers = Samplers::new(&device);
        let group0_layout = create_group0_layout(&device);
        let base_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shader pipeline layout"),
            bind_group_layouts: &[Some(&group0_layout)],
            immediate_size: 0,
        });
        let white = placeholder(&device, &gpu.queue, "missing user texture", [255; 4]);
        let builtin_warp = {
            let a = assemble(Stage::Warp, "", &[], WARP_BUILTIN_BODY);
            let module = create_module(&device, &a, "builtin warp");
            WarpShader {
                pipelines: warp::create_pipelines(
                    &device,
                    &base_pipeline_layout,
                    &module,
                    "builtin warp",
                ),
                group1: None,
            }
        };
        let builtin_comp = {
            let a = assemble(Stage::Comp, COMP_BUILTIN_WGSL, &[], COMP_BUILTIN_BODY);
            let module = create_module(&device, &a, "builtin comp");
            CompShader::new(module, base_pipeline_layout.clone(), None, "builtin comp")
        };
        RendererShared {
            device,
            queue: gpu.queue.clone(),
            group0_layout,
            base_pipeline_layout,
            samplers,
            white,
            builtin_warp,
            builtin_comp,
        }
    }

    /// Compiles a preset's warp and composite bodies (`None` = built-in),
    /// creating the composite pipeline for `comp_format` (the display
    /// target's format) so that `Renderer::composite` into that format does
    /// no shader compilation.
    ///
    /// User textures (`sampler_X` identifiers the prelude does not declare)
    /// are loaded with `tex(file)`, where `file` is `X` without its `fw_`,
    /// `fc_`, `pw_` or `pc_` prefix. A texture `tex` cannot supply is bound
    /// as 1x1 white and reported in `errors`; a stage that references more
    /// than 16 distinct user textures (counting warp then comp) falls back.
    /// Only stages that pass naga validation load textures, and only stages
    /// that compile count towards `blur_levels`.
    ///
    /// Safe to call from any thread: errors are caught in this thread's
    /// error scopes. Never panics on bad shader source or images.
    pub fn compile_shaders(
        &self,
        warp: Option<&str>,
        comp: Option<&str>,
        comp_format: wgpu::TextureFormat,
        tex: &dyn Fn(&str) -> Option<RgbaImage>,
    ) -> ShaderSet {
        let mut errors = Vec::new();
        let bodies = [(Stage::Warp, warp), (Stage::Comp, comp)];
        let scans = bodies.map(|(_, body)| body.map(shader_scan::scan).unwrap_or_default());
        for ((stage, _), scan) in bodies.iter().zip(&scans) {
            for e in &scan.errors {
                errors.push(format!("{} shader: {e}", stage.name()));
            }
        }
        let (slots, overflow) = shader_scan::allocate_slots(&scans);
        if !overflow.is_empty() {
            errors.push(format!(
                "too many user textures (at most {MAX_USER_TEXTURES} per preset); not bound: {}",
                overflow.join(", ")
            ));
        }
        let fail = |errors: &mut Vec<String>, stage: Stage, e: String| {
            tracing::warn!("{e}; using the built-in {} shader", stage.name());
            errors.push(e);
        };

        // 1. Slot and naga checks; nothing is uploaded for a stage that fails.
        let mut ok = [false; 2];
        for (i, &(stage, body)) in bodies.iter().enumerate() {
            if let Some(body) = body {
                match check_stage(stage, body, &scans[i], &slots, &overflow) {
                    Ok(()) => ok[i] = true,
                    Err(e) => fail(&mut errors, stage, e),
                }
            }
        }

        // 2. User textures of the stages that passed (a subset of `slots`,
        //    reallocated so the bound slots are contiguous).
        let (slots, _) =
            shader_scan::allocate_slots(scans.iter().zip(ok).filter_map(|(s, ok)| ok.then_some(s)));
        let mut texture_images = HashMap::new();
        let group1 = if slots.is_empty() {
            None
        } else {
            self.create_group1(&slots, tex, &mut texture_images, &mut errors)
        };

        // 3. Modules and pipelines.
        let mut compiled = [None, None];
        for (i, &(stage, body)) in bodies.iter().enumerate() {
            let Some(body) = body.filter(|_| ok[i]) else {
                continue;
            };
            match self.build_stage(stage, body, &scans[i], &slots, group1.as_ref(), comp_format) {
                Ok(c) => compiled[i] = Some(c),
                Err(e) => {
                    ok[i] = false;
                    fail(&mut errors, stage, e);
                }
            }
        }
        let [warp_compiled, comp_compiled] = compiled;
        let warp_shader = match warp_compiled {
            Some(Compiled::Warp(w)) => w,
            _ => self.builtin_warp.clone(),
        };
        let comp_shader = match comp_compiled {
            Some(Compiled::Comp(c)) => c,
            _ => {
                if let Err(e) = self.builtin_comp.prepare(&self.device, comp_format) {
                    errors.push(format!("built-in comp shader: {e}"));
                }
                self.builtin_comp.clone()
            }
        };

        ShaderSet {
            warp: warp_shader,
            comp: comp_shader,
            errors,
            blur_levels: scans
                .iter()
                .zip(ok)
                .filter(|(_, ok)| *ok)
                .map(|(s, _)| s.blur_levels)
                .max()
                .unwrap_or(0),
            user_textures: slots,
            warp_source: warp.map(str::to_string),
            comp_source: comp.map(str::to_string),
            texture_images,
        }
    }

    /// Builds a stage that passed [`check_stage`]: the module and pipelines
    /// (for a composite, the pipeline for `comp_format`), inside error scopes.
    fn build_stage(
        &self,
        stage: Stage,
        body: &str,
        scan: &Scan,
        slots: &[String],
        group1: Option<&Group1>,
        comp_format: wgpu::TextureFormat,
    ) -> Result<Compiled, String> {
        let name = stage.name();
        let (decls, layout, bind_group) = match group1 {
            Some(g) => (slots, &g.pipeline_layout, Some(g.bind_group.clone())),
            None if scan.textures.is_empty() => (&[][..], &self.base_pipeline_layout, None),
            None => {
                return Err(format!(
                    "{name} shader: its user textures could not be bound"
                ));
            }
        };
        // Same body as checked; only the group 1 declarations may have shrunk
        // to the textures of the stages that passed, which still covers every
        // texture this stage uses. wgpu validates the module again below.
        let a = assemble(stage, "", decls, body);
        let label = format!("user {name}");
        let device = &self.device;
        let (compiled, error) = scoped(device, || {
            let module = create_module(device, &a, &label);
            match stage {
                Stage::Warp => Compiled::Warp(WarpShader {
                    pipelines: warp::create_pipelines(device, layout, &module, &label),
                    group1: bind_group,
                }),
                Stage::Comp => {
                    Compiled::Comp(CompShader::new(module, layout.clone(), bind_group, &label))
                }
            }
        });
        if let Some(e) = error {
            return Err(format!("{name} shader: pipeline creation failed: {e}"));
        }
        if let Compiled::Comp(comp) = &compiled {
            comp.prepare(device, comp_format)
                .map_err(|e| format!("{name} shader: pipeline creation failed: {e}"))?;
        }
        Ok(compiled)
    }

    /// Loads the user textures and builds the group 1 layout and bind group.
    /// Returns `None` (with an error) only if the GPU objects could not be
    /// created.
    fn create_group1(
        &self,
        slots: &[String],
        tex: &dyn Fn(&str) -> Option<RgbaImage>,
        images: &mut HashMap<String, Arc<RgbaImage>>,
        errors: &mut Vec<String>,
    ) -> Option<Group1> {
        let mut views = Vec::with_capacity(slots.len());
        for name in slots {
            let (_, file) = shader_scan::split_prefix(name);
            let image = match images.get(file) {
                Some(image) => Some(Arc::clone(image)),
                None => tex(file).map(Arc::new),
            };
            let view = match image {
                None => {
                    errors.push(format!(
                        "user texture `{name}`: no texture `{file}`; bound as white"
                    ));
                    self.white.clone()
                }
                Some(image) => match self.upload(name, &image) {
                    Ok(view) => {
                        images.insert(file.to_string(), image);
                        view
                    }
                    Err(e) => {
                        errors.push(format!("user texture `{name}`: {e}; bound as white"));
                        self.white.clone()
                    }
                },
            };
            views.push(view);
        }

        let (group1, error) = scoped(&self.device, || {
            let mut layout_entries = Vec::with_capacity(slots.len() * 2);
            let mut entries = Vec::with_capacity(slots.len() * 2);
            for (i, (name, view)) in slots.iter().zip(&views).enumerate() {
                let binding = 2 * i as u32;
                let (kind, _) = shader_scan::split_prefix(name);
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: texture_2d_entry(),
                    count: None,
                });
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: binding + 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                });
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::TextureView(view),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: binding + 1,
                    resource: wgpu::BindingResource::Sampler(self.samplers.get(kind)),
                });
            }
            let layout = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("user textures"),
                    entries: &layout_entries,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("user textures"),
                layout: &layout,
                entries: &entries,
            });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("shader pipeline layout (user textures)"),
                        bind_group_layouts: &[Some(&self.group0_layout), Some(&layout)],
                        immediate_size: 0,
                    });
            Group1 {
                pipeline_layout,
                bind_group,
            }
        });
        match error {
            None => Some(group1),
            Some(e) => {
                errors.push(format!("user textures could not be bound: {e}"));
                None
            }
        }
    }

    /// Uploads an image as an `Rgba8Unorm` texture.
    fn upload(&self, name: &str, image: &RgbaImage) -> Result<wgpu::TextureView, String> {
        let (width, height) = image.dimensions();
        let max = self.device.limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > max || height > max {
            return Err(format!(
                "image is {width}x{height} (allowed 1..={max} per side)"
            ));
        }
        let (view, error) = scoped(&self.device, || {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(name),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                tex.as_image_copy(),
                image.as_raw(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                tex.size(),
            );
            tex.create_view(&Default::default())
        });
        match error {
            None => Ok(view),
            Some(e) => Err(format!("upload failed: {e}")),
        }
    }
}

enum Compiled {
    Warp(WarpShader),
    Comp(CompShader),
}

fn create_module(device: &wgpu::Device, a: &Assembled, label: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(a.source.as_str().into()),
    })
}

fn texture_2d_entry() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

/// Group 0, as declared by the prelude: 0 `Uniforms`, 1 `sampler_main`,
/// 2-4 blur levels, 5-7 2D noise, 8-9 3D noise, 10 spectrum, 11-14 samplers
/// `s_fw`, `s_fc`, `s_pw`, `s_pc`.
fn create_group0_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let fragment = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    };
    let volume = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D3,
        multisampled: false,
    };
    let sampler = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
        },
        count: None,
    }];
    entries.extend((1..=7).map(|b| fragment(b, texture_2d_entry())));
    entries.extend((8..=9).map(|b| fragment(b, volume)));
    entries.push(fragment(10, texture_2d_entry()));
    entries.extend((11..=14).map(|b| fragment(b, sampler)));
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shader group 0"),
        entries: &entries,
    })
}

/// Builds a group 0 bind group: `uniforms`, `main` as `sampler_main`, the
/// auxiliary textures and the four samplers.
pub(crate) fn group0_bind_group(
    shared: &RendererShared,
    aux: &AuxViews,
    uniforms: &wgpu::Buffer,
    main: &wgpu::TextureView,
    label: &str,
) -> wgpu::BindGroup {
    use wgpu::BindingResource::{Sampler, TextureView};
    let views = [
        main,
        &aux.blur[0],
        &aux.blur[1],
        &aux.blur[2],
        &aux.noise[0],
        &aux.noise[1],
        &aux.noise[2],
        &aux.noisevol[0],
        &aux.noisevol[1],
        &aux.spectrum,
    ];
    let s = &shared.samplers;
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: uniforms.as_entire_binding(),
    }];
    entries.extend(
        (1u32..)
            .zip(views)
            .map(|(binding, view)| wgpu::BindGroupEntry {
                binding,
                resource: TextureView(view),
            }),
    );
    entries.extend(
        (11u32..)
            .zip([&s.fw, &s.fc, &s.pw, &s.pc])
            .map(|(binding, sampler)| wgpu::BindGroupEntry {
                binding,
                resource: Sampler(sampler),
            }),
    );
    shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &shared.group0_layout,
        entries: &entries,
    })
}

impl Renderer {
    /// A cheap, `Send + Sync` handle for compiling shaders on another
    /// thread (see [`RendererShared::compile_shaders`]).
    pub fn shared(&self) -> RendererShared {
        self.shared.clone()
    }

    /// Compiles a preset's shaders; see [`RendererShared::compile_shaders`].
    /// Safe to call from any thread.
    pub fn compile_shaders(
        &self,
        _gpu: &Gpu,
        warp: Option<&str>,
        comp: Option<&str>,
        comp_format: wgpu::TextureFormat,
        tex: &dyn Fn(&str) -> Option<RgbaImage>,
    ) -> ShaderSet {
        self.shared.compile_shaders(warp, comp, comp_format, tex)
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
    fn shared_handles_are_send_sync() {
        fn check<T: Send + Sync>() {}
        check::<RendererShared>();
        check::<ShaderSet>();
    }

    #[test]
    fn body_positions_are_relative_to_the_body() {
        let a = assemble(
            Stage::Comp,
            "",
            &["t".to_string()],
            "let a = 1.0;\nreturn x;",
        );
        let body_start = a.source.find("let a").unwrap();
        assert_eq!(a.body_position(body_start), Some((1, 1)));
        let ret = a.source.find("return x").unwrap();
        assert_eq!(a.body_position(ret + 7), Some((2, 8)));
        // The prelude and the closing brace are generated code.
        assert_eq!(a.body_position(0), None);
        assert_eq!(a.body_position(a.source.len() - 2), None);
        assert_eq!(a.body_position(a.source.len() + 10), None);
    }

    #[test]
    fn builtin_sources_validate() {
        let warp = assemble(Stage::Warp, "", &[], WARP_BUILTIN_BODY);
        assert_eq!(naga_check(Stage::Warp, &warp), Ok(()));
        let comp = assemble(Stage::Comp, COMP_BUILTIN_WGSL, &[], COMP_BUILTIN_BODY);
        assert_eq!(naga_check(Stage::Comp, &comp), Ok(()));
    }
}
