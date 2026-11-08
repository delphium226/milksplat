//! The director (spec §6.7, §7): decides which preset is on screen, when to
//! blend or hard-cut to another, and drives the per-tick rendering of one
//! preset or of a blend between two.
//!
//! # Use from the render thread
//!
//! - Each simulation tick: [`Director::update`] (commands, loader results,
//!   auto-advance, hard cuts, requests), then [`Director::tick_and_render`]
//!   (ticks the preset or both blending presets and runs the warp, overlay
//!   and blur passes; it also switches the renderer's feedback format to the
//!   one [`Director::frame_plan`] wants).
//! - Each display frame: [`Director::composite`].
//! - Input: [`Director::command`] (applied by the next `update`).
//! - Files: [`Director::reload_current`] (hot reload of the preset on
//!   screen) and [`Director::set_library`] (presets added or removed).
//! - Device loss: [`Director::on_device_lost`] (returns an id for the
//!   persistent blocklist), then, once the renderer and the loader have been
//!   recreated, [`Director::on_renderer_recreated`].
//!
//! # State machine
//!
//! - **Showing** one preset, or **Blending** from the current preset to an
//!   incoming one (for `blend_s` seconds, with a random [`BlendPattern`]).
//!   The incoming preset *becomes current* when its blend starts: its
//!   display time, its auto-advance deadline and the hard-cut gap all count
//!   from then.
//! - As soon as a preset becomes current, the next one is picked (the
//!   [`Selector`]) and requested from the [`Loader`]; it waits as the *ready
//!   next* preset. A switch only ever happens to a preset that is ready.
//! - **Auto-advance:** once `next_duration(auto_advance_s)` has passed since
//!   the current preset became current, and a next preset is ready, blend to
//!   it. **Hard cut:** a beat with `beat_strength > hard_cut_strength`, at
//!   least `hard_cut_min_gap_s` after the last change, switches to the ready
//!   next preset instantly. Neither happens while locked, blending, or
//!   waiting for a manually requested preset.
//! - **Commands:** `Next`/`HardCut` take the ready next preset (or wait for
//!   the one being loaded); `Random`, `Load(id)` and `Prev` request a
//!   specific preset (`Prev` walks back past entries that are now banned,
//!   blocklisted or failed). The current preset stays on screen until the
//!   requested one is ready. A newer request supersedes an older pending
//!   one; answers to superseded requests are ignored (matched by ticket).
//!   A switch requested during a blend waits for the blend to finish.
//! - **Requests** remember the loader they went to
//!   ([`Loader::loader_id`]): if `update` is given a different loader (the
//!   app replaced it), the pending request is sent again to that one. A
//!   request with no answer after 30 s (of `now`) is logged, and sent again
//!   only if the loader reports it is idle (it lost the job); a busy loader
//!   is not sent duplicates.
//! - **Failures:** a preset that fails to load is logged and skipped for the
//!   rest of the session, and another is requested. After 10 consecutive
//!   failures the director stays on the current preset and tries again 10 s
//!   later. A load that failed because the GPU device was lost is neither
//!   skipped nor counted: it is held until [`Director::on_device_lost`]
//!   decides (it blames the preset being loaded, so that preset is
//!   blocklisted and never sent again), or, if that is not called, it is
//!   sent again to the next loader.
//! - The built-in default preset is shown at startup and whenever nothing
//!   else can be (empty library, everything banned, blocklisted or failed).

mod active;
mod switching;

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use ms_analysis::AudioFrame;
use ms_expr::Rng;
use ms_preset::{compile_preset, parse, CompiledPreset, Library, Prefs, Selector};
use ms_render::{
    CompositeLayer, FeedbackFormat, Gpu, Renderer, RendererShared, TextureResolver, WarpLayer,
};

use crate::blend::{composite_layers, render_blend, BlendPattern, BlendScratch};
use crate::env::Globals;
use crate::loader::Loader;
use active::{compile_default, Active, Blend, Loaded, Shaders};
use switching::{Goal, Hist, Pending, Queued};

/// The id of the built-in default preset (never a library id: library ids
/// end in `.splat`).
pub const DEFAULT_PRESET_ID: &str = "<default>";

/// The built-in default preset's source.
pub const DEFAULT_PRESET_SOURCE: &str = include_str!("../../../presets/default.splat");

/// Consecutive load failures before the director pauses its requests.
const MAX_FAILURES: u32 = 10;
/// How long requests pause after [`MAX_FAILURES`] failures, or after
/// finding nothing to request, in seconds.
const RETRY_S: f64 = 10.0;
/// A device loss this soon after a preset became current blocklists it.
const BLOCKLIST_WINDOW_S: f64 = 2.0;
/// Presets remembered for [`Command::Prev`].
const HISTORY_CAP: usize = 50;
/// Default mesh cap (spec §6.6: at most 192x144).
const DEFAULT_MESH_CAP: [u32; 2] = [192, 144];
/// A request with no answer after this long (seconds of `now`) is sent
/// again.
const REQUEST_TIMEOUT_S: f64 = 30.0;

/// A user or app command, applied by the next [`Director::update`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Blend to the next preset in shuffle order.
    Next,
    /// Blend back to the preset shown before the current one.
    Prev,
    /// Blend to a uniformly random preset (not the current one).
    Random,
    /// Stop (or resume) auto-advance and hard cuts.
    ToggleLock,
    /// Cut to the next preset in shuffle order, without a blend.
    HardCut,
    /// Blend to the library preset with this id (banned and blocklisted
    /// ids are refused).
    Load(String),
}

/// Director settings (spec §7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectorConfig {
    /// Base display time, varied by ±25%. Zero, negative or non-finite:
    /// no auto-advance.
    pub auto_advance_s: f64,
    /// Blend duration. Zero, negative or non-finite: switches are instant.
    pub blend_s: f64,
    pub hard_cuts: bool,
    pub hard_cut_strength: f32,
    pub hard_cut_min_gap_s: f64,
}

impl Default for DirectorConfig {
    fn default() -> Self {
        DirectorConfig {
            auto_advance_s: 20.0,
            blend_s: 2.7,
            hard_cuts: true,
            hard_cut_strength: 2.0,
            hard_cut_min_gap_s: 10.0,
        }
    }
}

/// What is on screen.
#[derive(Debug, Clone, PartialEq)]
pub enum State {
    Showing {
        cur: String,
    },
    /// A blend from `from` to `to`, started at `start` (seconds, the
    /// `now` of `update`), lasting `dur` seconds.
    Blending {
        from: String,
        to: String,
        start: f64,
        dur: f64,
        pattern: BlendPattern,
    },
}

/// The per-tick inputs the app supplies; the director adds each preset's
/// own time, frame count and progress.
#[derive(Debug, Clone, Copy)]
pub struct TickInputBase<'a> {
    pub fps: f64,
    pub audio: &'a AudioFrame,
    /// See [`TickInput::aspect`].
    pub aspect: [f64; 2],
    /// Internal (feedback) resolution.
    pub pixels: [u32; 2],
}

/// What the next tick and frame will render.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FramePlan {
    /// The feedback format the current preset wants (during a blend: fp16
    /// if either preset wants it). [`Director::tick_and_render`] applies it
    /// with `Renderer::set_format`.
    pub feedback_format: FeedbackFormat,
    /// Blur levels to compute (during a blend: the larger of the two).
    pub blur_levels: u8,
    /// Blend progress in [0, 1], or `None` when one preset is showing.
    pub blend_progress: Option<f32>,
    /// Simulation ticks per second (spec §3.3, §4): the current preset's
    /// `sim_rate`; during a blend, the outgoing preset's, until the blend
    /// completes. The app drives its fixed clock at this rate
    /// (`FixedClock::advance_at`) and passes it as the tick's `fps`.
    pub sim_rate: f64,
}

/// Decides what is shown and renders it (see the module docs).
pub struct Director {
    cfg: DirectorConfig,
    lib: Library,
    prefs: Prefs,
    /// Ids that failed to load this session.
    skip: HashSet<String>,
    /// `prefs` with `skip` added to the blocklist: what `Selector` sees.
    pick_prefs: Prefs,
    selector: Selector,
    rng: Rng,
    mesh_cap: [u32; 2],
    default_preset: Arc<CompiledPreset>,
    default_shaders: Shaders,
    cur: Active,
    blend: Option<Blend>,
    ready: Option<Loaded>,
    pending: Option<Pending>,
    queued: Option<Queued>,
    commands: VecDeque<Command>,
    locked: bool,
    /// Library presets shown, oldest first, and the index of the current.
    history: Vec<String>,
    cursor: Option<usize>,
    failures: u32,
    /// No requests before this time (after failures, or nothing to pick).
    retry_at: f64,
    started: bool,
    scratch: BlendScratch,
    /// Per-vertex weight 1, for single-preset warps.
    ones: Vec<f32>,
    /// See [`Director::set_load_hook`].
    load_hook: Option<LoadHook>,
    /// The hook was last told `Some(id)`.
    load_notified: bool,
}

type LoadHook = Box<dyn FnMut(Option<&str>) + Send>;

impl Director {
    /// Starts on the built-in default preset ([`DEFAULT_PRESET_SOURCE`]),
    /// which is parsed and compiled here, on the calling thread: it is tiny
    /// and trusted, so the default stack is plenty. `shared` and
    /// `comp_format` (the display target's format) compile its shaders,
    /// which are the built-in ones. `seed` seeds the selection, the display
    /// times, the blend patterns and each preset instance's seed.
    ///
    /// Nothing is requested until the first [`update`](Self::update), whose
    /// `now` is when the default preset becomes current.
    pub fn new(
        cfg: DirectorConfig,
        lib: Library,
        prefs: Prefs,
        seed: u64,
        shared: &RendererShared,
        comp_format: wgpu::TextureFormat,
    ) -> Self {
        let preset = match parse(DEFAULT_PRESET_SOURCE.as_bytes()) {
            Ok((p, _)) => p,
            Err(e) => {
                tracing::error!("the built-in default preset does not parse: {e}");
                ms_preset::Preset::default()
            }
        };
        let default_preset = Arc::new(compile_preset(preset));
        let default_shaders =
            Shaders::Compiled(compile_default(shared, comp_format, &default_preset));
        let mut rng = Rng::seeded(seed);
        let selector = Selector::new(rng.next_u64());
        let loaded = Loaded {
            id: DEFAULT_PRESET_ID.to_string(),
            preset: Arc::clone(&default_preset),
            shaders: default_shaders.duplicate(),
            seed: rng.next_u64(),
        };
        let mut d = Director {
            cfg,
            lib,
            pick_prefs: prefs.clone(),
            prefs,
            skip: HashSet::new(),
            selector,
            rng,
            mesh_cap: DEFAULT_MESH_CAP,
            default_preset,
            default_shaders,
            cur: Active::new(loaded, 0.0, f64::INFINITY, DEFAULT_MESH_CAP),
            blend: None,
            ready: None,
            pending: None,
            queued: None,
            commands: VecDeque::new(),
            locked: false,
            history: Vec::new(),
            cursor: None,
            failures: 0,
            retry_at: f64::NEG_INFINITY,
            started: false,
            scratch: BlendScratch::default(),
            ones: Vec::new(),
            load_hook: None,
            load_notified: false,
        };
        d.cur.duration = d.display_time();
        d
    }

    /// Queues a command for the next [`update`](Self::update).
    pub fn command(&mut self, c: Command) {
        self.commands.push_back(c);
    }

    /// Advances the state machine to `now` (seconds, monotonic): applies
    /// the queued commands, takes the loader's answers, finishes a blend
    /// that is done, auto-advances or hard-cuts, and makes the requests
    /// that are due. Call it once per tick, before
    /// [`tick_and_render`](Self::tick_and_render). `loader` may be replaced
    /// at any time (see the module docs). A non-finite `now` is ignored
    /// (the call does nothing). Never blocks.
    pub fn update(&mut self, now: f64, audio: &AudioFrame, loader: &Loader) {
        if !now.is_finite() {
            tracing::warn!("director: ignoring an update with a non-finite time ({now})");
            return;
        }
        if !self.started {
            self.started = true;
            self.cur.since = now;
        }
        while let Some(c) = self.commands.pop_front() {
            self.apply(c, now);
        }
        while let Some(r) = loader.poll() {
            self.on_result(r, now);
        }
        if let Some(p) = &mut self.pending {
            if p.loader != loader.loader_id() && (p.req.is_some() || p.parked) {
                // Sent to (or held for) another loader, which will never
                // answer here: send it to this one.
                p.req = None;
                p.parked = false;
            } else if p.req.is_some() && now - p.sent_at >= REQUEST_TIMEOUT_S {
                if loader.idle() {
                    // The loader has no job, so this one was lost: send it
                    // again.
                    tracing::warn!(
                        "loading {} got no answer in {REQUEST_TIMEOUT_S} s and the loader is idle; requesting it again",
                        p.id
                    );
                    p.req = None;
                } else if !p.timeout_logged {
                    // Still being worked on (or queued): a duplicate would
                    // only queue the same work again.
                    p.timeout_logged = true;
                    tracing::warn!(
                        "loading {} got no answer in {REQUEST_TIMEOUT_S} s; still waiting for the loader",
                        p.id
                    );
                }
            }
        }
        self.finish_blend(now);
        let switch_pending = self
            .pending
            .as_ref()
            .is_some_and(|p| matches!(p.goal, Goal::Switch { .. }));
        if self.blend.is_none() && !self.locked && !switch_pending && self.ready.is_some() {
            let since = self.cur.since;
            let cut = self.cfg.hard_cuts
                && audio.beat_strength > self.cfg.hard_cut_strength
                && now - since >= self.cfg.hard_cut_min_gap_s;
            let due = now >= since + self.cur.duration;
            if cut || due {
                if let Some(next) = self.ready.take() {
                    self.switch(next, cut, Hist::Push, now);
                }
            }
        }
        self.prefetch(now);
        if let Some(p) = &mut self.pending {
            if p.req.is_none() && !p.parked {
                match self.lib.entries.iter().find(|e| e.id == p.id) {
                    Some(e) => {
                        // Before the loader can start on it (see `set_load_hook`).
                        if let Some(hook) = &mut self.load_hook {
                            hook(Some(&p.id));
                            self.load_notified = true;
                        }
                        p.req = Some(loader.request(p.id.clone(), e.path.clone(), p.seed));
                        p.loader = loader.loader_id();
                        p.sent_at = now;
                        p.timeout_logged = false;
                    }
                    None => {
                        tracing::warn!("preset {} is no longer in the library", p.id);
                        self.pending = None;
                    }
                }
            }
        }
        // Nothing with the loader any more: the answer was taken, or the
        // request was dropped.
        if self.loading().is_none() {
            self.notify_not_loading();
        }
    }

    /// The preset the loader is working on for this director: a request in
    /// flight, or one whose load failed because the device was lost (held
    /// for [`on_device_lost`](Self::on_device_lost)). The load hook's
    /// `Some(id)` lasts exactly as long as this is `Some(id)`.
    pub fn loading(&self) -> Option<&str> {
        self.pending
            .as_ref()
            .filter(|p| p.req.is_some() || p.parked)
            .map(|p| p.id.as_str())
    }

    /// Tells the load hook that nothing is loading, if it was told otherwise.
    fn notify_not_loading(&mut self) {
        if self.load_notified {
            self.load_notified = false;
            if let Some(hook) = &mut self.load_hook {
                hook(None);
            }
        }
    }

    /// Sets a hook told which preset the loader is working on: `Some(id)`
    /// just *before* each request is sent (so the app can record it before
    /// any of the preset's work starts, e.g. for crash detection), and
    /// `None` once no request is outstanding (its answer was taken by
    /// `update`, or it was dropped); see [`loading`](Self::loading). Called
    /// on the thread calling `update` (or `on_device_lost`).
    pub fn set_load_hook(&mut self, hook: impl FnMut(Option<&str>) + Send + 'static) {
        self.load_hook = Some(Box::new(hook));
    }

    /// The id of the preset on screen (during a blend, the incoming one),
    /// without allocating.
    pub fn shown(&self) -> &str {
        self.shown_id()
    }

    /// What is on screen.
    pub fn current(&self) -> State {
        match &self.blend {
            None => State::Showing {
                cur: self.cur.id.clone(),
            },
            Some(b) => State::Blending {
                from: self.cur.id.clone(),
                to: b.to.id.clone(),
                start: b.start,
                dur: b.dur,
                pattern: b.pattern,
            },
        }
    }

    /// Whether auto-advance and hard cuts are stopped.
    pub fn locked(&self) -> bool {
        self.locked
    }

    /// The id of the next preset, if it is loaded and ready.
    pub fn next_ready(&self) -> Option<&str> {
        self.ready.as_ref().map(|l| l.id.as_str())
    }

    /// The feedback format, blur levels, blend progress and simulation
    /// rate for `now`.
    pub fn frame_plan(&self, now: f64) -> FramePlan {
        let cur = &self.cur;
        match &self.blend {
            None => FramePlan {
                feedback_format: cur.feedback_format(),
                blur_levels: cur.shaders.blur_levels(),
                blend_progress: None,
                sim_rate: cur.sim_rate(),
            },
            Some(b) => {
                let fp16 =
                    [cur.feedback_format(), b.to.feedback_format()].contains(&FeedbackFormat::Fp16);
                FramePlan {
                    feedback_format: if fp16 {
                        FeedbackFormat::Fp16
                    } else {
                        FeedbackFormat::Rgba8
                    },
                    blur_levels: cur.shaders.blur_levels().max(b.to.shaders.blur_levels()),
                    blend_progress: Some(b.progress(now)),
                    sim_rate: cur.sim_rate(),
                }
            }
        }
    }

    /// Runs one simulation tick at `now`: sets the renderer's feedback
    /// format to [`frame_plan`](Self::frame_plan)'s if it differs, then
    /// either ticks the current preset and runs its warp pass (weight 1),
    /// its overlays (alpha 1) and its blur levels, or, during a blend, ticks
    /// both presets and runs `render_blend` and the larger blur level. Each
    /// preset's `time` counts from when it became current. Does not block.
    ///
    /// The app still sends the spectrum (`Renderer::set_spectrum`) and
    /// handles resizes itself.
    pub fn tick_and_render(
        &mut self,
        now: f64,
        base: &TickInputBase,
        globals: &mut Globals,
        renderer: &mut Renderer,
        gpu: &Gpu,
    ) {
        if self.on_screen_released() {
            tracing::debug!("tick_and_render after release_gpu: skipped");
            return;
        }
        let plan = self.frame_plan(now);
        if renderer.format() != plan.feedback_format {
            renderer.set_format(gpu, plan.feedback_format);
        }
        match &mut self.blend {
            None => {
                let (out, shaders) = self.cur.tick(now, base, globals);
                let Some(shaders) = shaders.compiled() else {
                    return;
                };
                let n = out.mesh.uv.len();
                if self.ones.len() != n {
                    self.ones.clear();
                    self.ones.resize(n, 1.0);
                }
                renderer.warp(
                    gpu,
                    &[WarpLayer {
                        mesh: &out.mesh,
                        weight: &self.ones,
                        shader: &shaders.warp,
                        uniforms: &out.uniforms,
                    }],
                );
                renderer.draw(gpu, &out.draw, 1.0);
            }
            Some(b) => {
                let p = b.progress(now);
                let from = self.cur.tick(now, base, globals);
                let to = b.to.tick(now, base, globals);
                let (Some(from_set), Some(to_set)) = (from.1.compiled(), to.1.compiled()) else {
                    return;
                };
                render_blend(
                    renderer,
                    gpu,
                    (from.0, &from_set.warp),
                    (to.0, &to_set.warp),
                    &b.pattern,
                    p,
                    &mut self.scratch,
                );
            }
        }
        renderer.blur(gpu, plan.blur_levels);
    }

    /// Composites the latest feedback image into `target` (of
    /// `target_format`, the format the loader compiled for), at the display
    /// rate: the current preset's composite shader, or, during a blend,
    /// both presets' cross-faded by the blend progress at `now`. Uniform
    /// `time` is exact for `now`. Does not block.
    pub fn composite(
        &self,
        now: f64,
        renderer: &mut Renderer,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) {
        let Some(cur_set) = self.cur.shaders.compiled() else {
            tracing::debug!("composite after release_gpu: skipped");
            return;
        };
        let cur_u = self.cur.display_uniforms(now);
        match &self.blend {
            None => renderer.composite(
                gpu,
                target,
                target_format,
                &[CompositeLayer {
                    shader: &cur_set.comp,
                    uniforms: &cur_u,
                    alpha: 1.0,
                }],
            ),
            Some(b) => {
                let Some(to_set) = b.to.shaders.compiled() else {
                    tracing::debug!("composite after release_gpu: skipped");
                    return;
                };
                let to_u = b.to.display_uniforms(now);
                let layers = composite_layers(
                    (&cur_set.comp, &cur_u),
                    (&to_set.comp, &to_u),
                    b.progress(now),
                );
                renderer.composite(gpu, target, target_format, &layers);
            }
        }
    }

    /// Call when the GPU device is lost, before recreating it. Blames one
    /// preset, if any:
    ///
    /// - the preset the loader was working on ([`loading`](Self::loading):
    ///   a request in flight, or one that already failed on the lost
    ///   device), since its shader compile or warm-up draw is the likeliest
    ///   cause; the request is dropped and never sent again;
    /// - otherwise the preset on screen (during a blend, the incoming one),
    ///   if it became current less than 2 s before `now`.
    ///
    /// The blamed preset is added to this director's blocklist, dropped
    /// from the ready and queued slots, and, if it is on screen, taken off
    /// it (a blend is abandoned and the outgoing preset stays; otherwise
    /// the default preset is shown); its id is returned for the app to add
    /// to the persistent blocklist. The default preset is never blamed.
    /// Otherwise returns `None`.
    ///
    /// A switch queued behind an abandoned blend is applied at once if its
    /// preset is still allowed, and dropped otherwise. A non-finite `now`
    /// blames nothing.
    pub fn on_device_lost(&mut self, now: f64) -> Option<String> {
        if !now.is_finite() {
            return None;
        }
        let id = match self.loading().filter(|id| *id != DEFAULT_PRESET_ID) {
            Some(id) => {
                let id = id.to_string();
                tracing::warn!("device lost while {id} was loading; blocklisting it");
                id
            }
            None => {
                let active = self.blend.as_ref().map_or(&self.cur, |b| &b.to);
                if active.id == DEFAULT_PRESET_ID || now - active.since >= BLOCKLIST_WINDOW_S {
                    return None;
                }
                tracing::warn!(
                    "device lost {:.2} s after {} started; blocklisting it",
                    now - active.since,
                    active.id
                );
                active.id.clone()
            }
        };
        self.prefs.blocklist.insert(id.clone());
        self.pick_prefs.blocklist.insert(id.clone());
        if self.pending.as_ref().is_some_and(|p| p.id == id) {
            self.pending = None;
            self.notify_not_loading();
        }
        if self.ready.as_ref().is_some_and(|l| l.id == id) {
            self.ready = None;
        }
        if self.shown_id() != id {
            if self.queued.as_ref().is_some_and(|q| q.loaded.id == id) {
                tracing::info!("dropping the queued switch to {id}");
                self.queued = None;
            }
            return Some(id);
        }
        if self.blend.take().is_none() {
            let loaded = Loaded {
                id: DEFAULT_PRESET_ID.to_string(),
                preset: Arc::clone(&self.default_preset),
                shaders: self.default_shaders.duplicate(),
                seed: self.rng.next_u64(),
            };
            let duration = self.display_time();
            self.cur = Active::new(loaded, now, duration, self.mesh_cap);
        }
        if self.ready.as_ref().is_some_and(|l| l.id == id) {
            self.ready = None;
        }
        // Only a blend can have a queued switch, and the blend is gone.
        if let Some(q) = self.queued.take() {
            if self.allowed(&q.loaded.id) {
                self.switch(q.loaded, q.cut, q.hist, now);
            } else {
                tracing::info!("dropping the queued switch to {}", q.loaded.id);
            }
        }
        Some(id)
    }

    /// Call when the GPU device is lost, after
    /// [`on_device_lost`](Self::on_device_lost) and before the app drops
    /// the old device: drops the GPU objects of every shader set the
    /// director holds (current, incoming, ready next, queued, default),
    /// keeping only their sources and texture images. Afterwards the
    /// director holds nothing of the old device, so the driver can hand
    /// out a fresh one (D3D12 returns the same, removed, device while
    /// references to it remain). Until
    /// [`on_renderer_recreated`](Self::on_renderer_recreated),
    /// [`tick_and_render`](Self::tick_and_render) and
    /// [`composite`](Self::composite) do nothing, and a loader answer that
    /// arrives is kept as sources only.
    pub fn release_gpu(&mut self) {
        for shaders in self.shader_slots() {
            shaders.release();
        }
    }

    /// Call after a device loss, once the app has a renderer on the new
    /// device (`Renderer::new` or `Renderer::recreate`) and has spawned a
    /// new [`Loader`] with the new `shared`: compiles, with `shared` and
    /// for `comp_format`, the shaders of every preset the director holds
    /// (current, incoming, ready next, queued, default), from the old sets
    /// or, after [`release_gpu`](Self::release_gpu), from their kept
    /// sources, taking user textures from the old images (or `resolver`).
    /// Preset instances keep running, so the show continues where it was.
    /// A request the old loader had not answered is sent again to the new
    /// loader by the next `update`.
    pub fn on_renderer_recreated(
        &mut self,
        shared: &RendererShared,
        resolver: &TextureResolver,
        comp_format: wgpu::TextureFormat,
    ) {
        let seed = self.cur.seed;
        self.cur
            .shaders
            .rebuild(shared, resolver, comp_format, seed);
        if let Some(b) = &mut self.blend {
            b.to.shaders
                .rebuild(shared, resolver, comp_format, b.to.seed);
        }
        if let Some(l) = &mut self.ready {
            l.shaders.rebuild(shared, resolver, comp_format, l.seed);
        }
        if let Some(q) = &mut self.queued {
            let seed = q.loaded.seed;
            q.loaded
                .shaders
                .rebuild(shared, resolver, comp_format, seed);
        }
        self.default_shaders
            .rebuild(shared, resolver, comp_format, 0);
        if let Some(p) = &mut self.pending {
            p.req = None;
            p.parked = false;
        }
    }

    /// Caps the mesh of presets that become current from now on (the
    /// quality setting; default 192x144).
    pub fn set_mesh_cap(&mut self, cap: [u32; 2]) {
        self.mesh_cap = cap;
    }

    /// The favourites, bans and blocklist in effect, including the ids
    /// [`on_device_lost`](Self::on_device_lost) has blocklisted.
    pub fn prefs(&self) -> &Prefs {
        &self.prefs
    }

    /// Replaces the favourites, bans and blocklist (the app calls this when
    /// the user favourites or bans a preset; start from
    /// [`prefs`](Self::prefs)). A ready next preset or a prefetch that is
    /// no longer allowed is dropped and another is picked; the preset on
    /// screen stays until the next switch.
    pub fn set_prefs(&mut self, prefs: Prefs) {
        let mut pick = prefs.clone();
        pick.blocklist.extend(self.skip.iter().cloned());
        self.prefs = prefs;
        self.pick_prefs = pick;
        if self.ready.as_ref().is_some_and(|l| !self.allowed(&l.id)) {
            self.ready = None;
        }
        if self
            .pending
            .as_ref()
            .is_some_and(|p| matches!(p.goal, Goal::Prefetch) && !self.allowed(&p.id))
        {
            self.pending = None;
        }
    }

    /// Hot reload: loads the file of the preset on screen (during a blend,
    /// the incoming one) again and, once it is ready, cuts to it without a
    /// blend (a fresh instance with the same seed; the feedback image
    /// carries over, the display time restarts). If the load fails, the
    /// running version stays and the preset is not skipped. Returns false
    /// (and does nothing) for the built-in default preset, or while a
    /// switch the user asked for is still loading. A newer request (a
    /// command) supersedes the reload.
    pub fn reload_current(&mut self) -> bool {
        let active = self.blend.as_ref().map_or(&self.cur, |b| &b.to);
        if active.id == DEFAULT_PRESET_ID
            || self
                .pending
                .as_ref()
                .is_some_and(|p| matches!(p.goal, Goal::Switch { .. }))
        {
            return false;
        }
        self.pending = Some(Pending::new(active.id.clone(), active.seed, Goal::Reload));
        true
    }

    /// Replaces the library (presets added or removed at runtime). The
    /// preset on screen stays. A ready next preset that is no longer in
    /// the library is dropped; a pending request for one is dropped by the
    /// next `update`. A pause in requests (after finding nothing to pick)
    /// ends, so new presets are picked up at once.
    pub fn set_library(&mut self, lib: Library) {
        self.lib = lib;
        let gone = |id: &str, lib: &Library| !lib.entries.iter().any(|e| e.id == id);
        if self.ready.as_ref().is_some_and(|l| gone(&l.id, &self.lib)) {
            self.ready = None;
        }
        self.retry_at = f64::NEG_INFINITY;
    }

    /// Every shader set the director holds.
    fn shader_slots(&mut self) -> Vec<&mut Shaders> {
        let mut slots = vec![&mut self.cur.shaders, &mut self.default_shaders];
        if let Some(b) = &mut self.blend {
            slots.push(&mut b.to.shaders);
        }
        if let Some(l) = &mut self.ready {
            slots.push(&mut l.shaders);
        }
        if let Some(q) = &mut self.queued {
            slots.push(&mut q.loaded.shaders);
        }
        slots
    }

    /// Whether a preset on screen has only its sources (after
    /// [`release_gpu`](Self::release_gpu)).
    fn on_screen_released(&self) -> bool {
        self.cur.shaders.is_released()
            || self
                .blend
                .as_ref()
                .is_some_and(|b| b.to.shaders.is_released())
    }

    /// Test hook: how many shader sets the director holds, and how many of
    /// them still hold GPU objects.
    #[doc(hidden)]
    pub fn shader_sets(&mut self) -> (usize, usize) {
        let slots = self.shader_slots();
        let compiled = slots.iter().filter(|s| !s.is_released()).count();
        (slots.len(), compiled)
    }

    /// Test hook: load failures since the last success (or retry pause).
    #[doc(hidden)]
    pub fn consecutive_failures(&self) -> u32 {
        self.failures
    }

    /// Test hook: the preset waiting for the current blend to finish.
    #[doc(hidden)]
    pub fn queued(&self) -> Option<&str> {
        self.queued.as_ref().map(|q| q.loaded.id.as_str())
    }
}
