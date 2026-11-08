//! The render thread (spec §3.2, §3.3, §6.7): owns the GPU device, the
//! renderer, the director and loader, the audio analysis, the fixed-tick
//! clock and the quality governor, and runs the frame loop.
//!
//! **Frame loop** (one display frame):
//! 1. Drain the message channel (resizes, hotkey actions, suspend, quit).
//! 2. While the window has no area (minimised) or is suspended: pause the
//!    simulation, keep draining the audio ring and the channel, sleep 50 ms.
//! 3. Acquire the surface texture. This happens before any GPU work so that
//!    the wait for a free back buffer (vsync) is not counted as GPU time.
//! 4. Drain the audio ring into the history and pad silence up to now.
//! 5. Advance the fixed 60 Hz clock by the frame's wall-clock delta. For
//!    each due tick: analysis, spectrum upload, `director.update`, then
//!    `director.tick_and_render` (preset code, warp, overlays, blur).
//! 6. Composite into the surface texture at the exact (not tick) time.
//! 7. Draw the text overlay.
//! 8. Present.
//! 9. Feed the frame's GPU time to the governor.
//!
//! Mailbox presentation does not wait for vsync, so with Mailbox the loop
//! paces itself to the monitor's refresh rate (sleeping before step 3).
//!
//! **Time:** the [`AppClock`] (monotonic seconds) stamps audio packets and
//! drives analysis. The simulation has its own time, advanced by the
//! frame deltas while frames are rendered: it stops while minimised and the
//! first frame after resuming has a zero delta, so nothing catches up in a
//! burst.

mod actions;
mod audio;
mod gpu_state;
mod gpu_timer;
mod overlay;
mod recovery;

pub use gpu_state::pick_format;
pub use overlay::display_name;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, TryRecvError};
use ms_analysis::{Analyzer, AudioHistory};
use ms_engine::{Director, DirectorConfig, FixedClock, Globals, Governor, TickInputBase};
use ms_preset::{Library, Prefs};
use ms_render::TextureResolver;
use winit::window::Window;

use crate::config::{Config, Quality};
use crate::hotkeys::Action;
use crate::paths::Paths;
use audio::AudioInput;
use gpu_state::GpuState;
use overlay::Overlay;

/// The render thread's stack: the default preset compiles on it (R15).
pub const RENDER_STACK_BYTES: usize = 16 * 1024 * 1024;
/// The simulation tick rate (§3.3).
pub const SIM_HZ: f64 = 60.0;
/// Frame deltas are capped at this, so a stall does not jump the
/// simulation ahead.
const MAX_DT: f64 = 0.25;
/// Sleep while minimised or suspended.
const IDLE_SLEEP: Duration = Duration::from_millis(50);
/// Delay between attempts to recreate a lost GPU device.
const GPU_RETRY_S: f64 = 1.0;
/// Audio history length: the 2048-sample analysis window plus the largest
/// A/V offset, with room to spare.
const HISTORY_SECONDS: f64 = 3.0;
/// The A/V offset range (ms). The analysis window cannot end in the future.
const MAX_AV_OFFSET_MS: i32 = 1000;
/// The history's sample rate until the capture reports its own.
const DEFAULT_SAMPLE_RATE: u32 = 48_000;
/// How often the monitor's refresh rate (the governor's budget) is re-read.
const BUDGET_CHECK_S: f64 = 2.0;

/// Messages from the main thread.
#[derive(Debug)]
pub enum ToRender {
    /// The window's inner size in physical pixels, and its DPI scale.
    Resize {
        size: [u32; 2],
        scale: f64,
    },
    /// A hotkey action (the main thread handles fullscreen, monitors and Esc).
    Action(Action),
    /// Pause rendering (`true`), e.g. while the window is occluded.
    Suspend(bool),
    Quit,
    /// A new window surface, created on the main thread (winit only hands
    /// out the window handle there) from the render thread's instance, in
    /// answer to `RenderOptions::request_surface` after a device loss.
    NewSurface(wgpu::Surface<'static>),
}

/// How the render thread ended.
#[derive(Debug, Clone, PartialEq)]
pub enum RenderExit {
    Clean,
    /// A fatal error (no usable GPU) or a caught panic; already logged.
    Failed(String),
}

/// Monotonic app time in seconds, shared with the audio capture clock.
#[derive(Debug, Clone, Copy)]
pub struct AppClock {
    start: Instant,
}

impl AppClock {
    pub fn new() -> AppClock {
        AppClock {
            start: Instant::now(),
        }
    }

    /// Seconds since the clock was created.
    pub fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

impl Default for AppClock {
    fn default() -> Self {
        AppClock::new()
    }
}

/// Everything the render thread needs besides the GPU objects and config.
pub struct RenderOptions {
    pub paths: Paths,
    /// The window's inner size (physical pixels) and DPI scale at startup.
    pub initial_size: [u32; 2],
    pub initial_scale: f64,
    /// `--exit-after`: stop after this many seconds of rendering.
    pub exit_after: Option<f64>,
    /// `--log-fps`: print a stats line to stderr once per second.
    pub log_fps: bool,
    /// `--debug-lose-device`: destroy the GPU device after this many seconds
    /// of rendering, to exercise device-loss recovery.
    pub lose_device_after: Option<f64>,
    /// Called once when the thread ends, however it ends.
    pub on_exit: Box<dyn FnOnce(RenderExit) + Send>,
    /// Asks the main thread for a new surface (sent back as
    /// [`ToRender::NewSurface`]); used after a device loss.
    pub request_surface: Box<dyn Fn() + Send>,
}

/// Starts the render thread ("ms-render", 16 MB stack). `instance` must be
/// the instance that created `surface`. A panic on the thread is caught and
/// logged, and reported through `on_exit` like any other ending.
pub fn run_render_thread(
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    window: Arc<Window>,
    rx: Receiver<ToRender>,
    cfg: Config,
    mut opts: RenderOptions,
) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("ms-render".into())
        .stack_size(RENDER_STACK_BYTES)
        .spawn(move || {
            let on_exit = std::mem::replace(&mut opts.on_exit, Box::new(|_| {}));
            let result = catch_unwind(AssertUnwindSafe(|| {
                run(instance, surface, window, rx, cfg, opts)
            }));
            let exit = match result {
                Ok(Ok(())) => RenderExit::Clean,
                Ok(Err(e)) => {
                    tracing::error!("render thread: {e}");
                    RenderExit::Failed(e)
                }
                Err(payload) => {
                    let msg = payload
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| payload.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic");
                    tracing::error!("render thread panicked: {msg}");
                    RenderExit::Failed(format!("panic: {msg}"))
                }
            };
            on_exit(exit);
        })
}

fn run(
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    window: Arc<Window>,
    rx: Receiver<ToRender>,
    cfg: Config,
    opts: RenderOptions,
) -> Result<(), String> {
    let clock = AppClock::new();
    let resolver = TextureResolver::new(opts.paths.texture_roots());
    let roots = opts.paths.preset_roots();
    let lib = Library::scan(&roots);
    tracing::info!("{} presets found in {roots:?}", lib.entries.len());

    let size = opts.initial_size;
    let internal = internal_size(size, 1.0);
    let gs = GpuState::new(&instance, surface, &resolver, internal)?;

    let prefs = Prefs {
        favourites: cfg.favourites.iter().cloned().collect(),
        bans: cfg.bans.iter().cloned().collect(),
        blocklist: cfg.blocklist.iter().cloned().collect(),
    };
    let dcfg = DirectorConfig {
        auto_advance_s: cfg.auto_advance_s,
        blend_s: cfg.blend_s,
        hard_cuts: cfg.hard_cuts,
        hard_cut_strength: cfg.hard_cut_strength,
        ..DirectorConfig::default()
    };
    let mut director = Director::new(dcfg, lib, prefs, seed(), &gs.renderer.shared(), gs.format);
    director.set_mesh_cap(cfg.quality.mesh_cap());
    let audio = AudioInput::start(cfg.audio_device.clone(), Arc::new(move || clock.now()));

    let mut rt = RenderThread {
        director,
        gs: Some(gs),
        new_surface: None,
        instance,
        audio,
        window,
        rx,
        resolver,
        clock,
        exit_after: opts.exit_after,
        log_fps: opts.log_fps,
        lose_device_after: opts.lose_device_after,
        sim: FixedClock::new(SIM_HZ),
        globals: Globals::default(),
        analyzer: Analyzer::new(DEFAULT_SAMPLE_RATE),
        hist: AudioHistory::new(DEFAULT_SAMPLE_RATE, HISTORY_SECONDS),
        sim_time: 0.0,
        tick_time: 0.0,
        last_wall: None,
        size,
        dpi_scale: opts.initial_scale,
        suspended: false,
        surface_dirty: true,
        lost_streak: 0,
        internal,
        governor: Governor::new(),
        quality: cfg.quality,
        budget_ms: 1000.0 / 60.0,
        next_frame_due: None,
        next_budget_check: 0.0,
        av_offset_ms: 0,
        overlay: Overlay::default(),
        stats: FrameStats::default(),
        gpu_ms: None,
        timings: Vec::new(),
        loss_handled: false,
        gpu_retry_at: 0.0,
        request_surface: opts.request_surface,
        surface_request_at: 0.0,
        first_frame: None,
        quit: false,
        cfg,
    };
    let result = rt.run_loop();
    rt.shutdown();
    result
}

/// A seed for preset selection, different each run.
fn seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    nanos ^ (u64::from(std::process::id()) << 32)
}

/// The internal (feedback) resolution for a surface of `size` at `scale`
/// (the renderer clamps it further to `[16, max]` per side).
pub fn internal_size(size: [u32; 2], scale: f32) -> [u32; 2] {
    let scale = if scale.is_finite() && scale > 0.0 {
        f64::from(scale)
    } else {
        1.0
    };
    size.map(|v| ((f64::from(v) * scale).round() as u32).max(1))
}

/// The aspect multipliers for the preset's vertex `rad`/`ang` and its
/// `aspectx`/`aspecty`: the longer side spans 1 and the shorter is
/// scaled down (MilkDrop's grid convention), e.g. `[1.0, 0.5625]` at 16:9.
pub fn aspect([w, h]: [u32; 2]) -> [f64; 2] {
    let (w, h) = (f64::from(w.max(1)), f64::from(h.max(1)));
    if w >= h {
        [1.0, h / w]
    } else {
        [w / h, 1.0]
    }
}

/// Frames per second over one-second windows.
#[derive(Debug, Default)]
struct FrameStats {
    window_start: Option<f64>,
    frames: u32,
    fps: f64,
    frame_ms: f64,
}

impl FrameStats {
    /// Counts a frame at `now`; true when a window just closed.
    fn frame(&mut self, now: f64) -> bool {
        let start = *self.window_start.get_or_insert(now);
        self.frames += 1;
        let elapsed = now - start;
        if elapsed >= 1.0 {
            self.fps = f64::from(self.frames) / elapsed;
            self.frame_ms = elapsed * 1000.0 / f64::from(self.frames);
            self.window_start = Some(now);
            self.frames = 0;
            return true;
        }
        false
    }

    /// Restarts the window (after a pause, which is not a slow frame).
    fn reset(&mut self) {
        self.window_start = None;
        self.frames = 0;
    }
}

/// The render thread's state. Fields drop in declaration order: the
/// director, then the GPU state (loader, timer, renderer, surface, device),
/// then a spare surface and the instance, then the audio service (bounded
/// by its own 2 s).
struct RenderThread {
    director: Director,
    /// `None` only after a device loss, until a new device is built.
    gs: Option<GpuState>,
    /// A surface from the main thread for the next recovery attempt.
    new_surface: Option<wgpu::Surface<'static>>,
    instance: wgpu::Instance,
    audio: AudioInput,
    window: Arc<Window>,
    rx: Receiver<ToRender>,
    resolver: TextureResolver,
    clock: AppClock,
    exit_after: Option<f64>,
    log_fps: bool,
    lose_device_after: Option<f64>,

    // Simulation.
    sim: FixedClock,
    globals: Globals,
    analyzer: Analyzer,
    hist: AudioHistory,
    /// Simulation seconds (frozen while paused).
    sim_time: f64,
    /// Simulation time of the last tick run.
    tick_time: f64,
    /// App-clock time of the last rendered frame; `None` after a pause.
    last_wall: Option<f64>,

    // Display.
    size: [u32; 2],
    dpi_scale: f64,
    suspended: bool,
    surface_dirty: bool,
    /// Consecutive `Lost` surface results (only the first is logged).
    lost_streak: u32,
    /// The internal size last requested from the renderer.
    internal: [u32; 2],
    governor: Governor,
    quality: Quality,
    /// The governor's frame budget and the Mailbox pacing period: one
    /// refresh period of the window's monitor.
    budget_ms: f64,
    /// When the next frame may start (Mailbox pacing).
    next_frame_due: Option<f64>,
    next_budget_check: f64,
    /// In memory until Task 24 stores it per device.
    av_offset_ms: i32,
    overlay: Overlay,
    stats: FrameStats,
    gpu_ms: Option<f64>,
    timings: Vec<f64>,

    // Device loss.
    loss_handled: bool,
    gpu_retry_at: f64,
    request_surface: Box<dyn Fn() + Send>,
    /// When to ask the main thread for a surface (again).
    surface_request_at: f64,

    first_frame: Option<f64>,
    quit: bool,
    /// The config with this session's in-memory changes (favourites, bans,
    /// blocklist, quality); Task 24 saves it.
    cfg: Config,
}

impl RenderThread {
    fn run_loop(&mut self) -> Result<(), String> {
        loop {
            self.drain_messages();
            if self.quit {
                return Ok(());
            }
            let wall = self.clock.now();
            if let (Some(limit), Some(start)) = (self.exit_after, self.first_frame) {
                if wall - start >= limit {
                    tracing::info!("--exit-after: {limit} s of rendering done");
                    return Ok(());
                }
            }
            self.maybe_lose_device(wall);
            if self.gs.as_ref().is_none_or(GpuState::is_lost) {
                self.recover(wall);
                continue;
            }
            if self.size[0] == 0 || self.size[1] == 0 || self.suspended {
                self.pause();
                continue;
            }
            if let Some(wait) = self.pace(wall) {
                std::thread::sleep(wait);
                continue;
            }
            self.frame(wall);
        }
    }

    /// With Mailbox presentation nothing blocks on vsync, so frames are
    /// paced to one refresh period of the monitor (the governor's budget)
    /// instead of rendering thousands per second. `Some` is how long to
    /// wait before the next frame.
    fn pace(&self, wall: f64) -> Option<Duration> {
        let mailbox = self.gs.as_ref().is_some_and(GpuState::mailbox);
        let due = self.next_frame_due.filter(|_| mailbox)?;
        (wall < due).then(|| Duration::from_secs_f64(due - wall))
    }

    /// Drops everything in order: director, GPU state, instance, audio.
    fn shutdown(self) {
        let RenderThread {
            director,
            gs,
            new_surface,
            instance,
            audio,
            ..
        } = self;
        drop(director);
        drop(gs);
        drop(new_surface);
        drop(instance);
        drop(audio);
        tracing::info!("render thread stopped");
    }

    fn handle(&mut self, msg: ToRender) {
        match msg {
            ToRender::Resize { size, scale } => {
                if size != self.size {
                    self.size = size;
                    self.surface_dirty = true;
                }
                if scale.is_finite() && scale > 0.0 {
                    self.dpi_scale = scale;
                }
            }
            ToRender::Action(a) => self.action(a),
            ToRender::Suspend(s) => self.suspended = s,
            ToRender::Quit => self.quit = true,
            // Only wanted while recovering from a device loss.
            ToRender::NewSurface(surface) => {
                if self.gs.is_none() {
                    self.new_surface = Some(surface);
                }
            }
        }
    }

    fn drain_messages(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(msg) => self.handle(msg),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.quit = true;
                    return;
                }
            }
        }
    }

    /// Waits up to 50 ms for a message.
    fn idle_wait(&mut self) {
        match self.rx.recv_timeout(IDLE_SLEEP) {
            Ok(msg) => self.handle(msg),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => self.quit = true,
        }
    }

    /// Minimised or suspended: no configure, no rendering, no simulation.
    fn pause(&mut self) {
        self.last_wall = None;
        self.next_frame_due = None;
        self.stats.reset();
        self.audio.drain(&mut self.hist);
        self.idle_wait();
    }

    /// Resizes the feedback buffers when the surface size or the scale
    /// (quality setting or governor) changes.
    fn apply_internal_size(&mut self, gs: &mut GpuState) {
        let scale = self.render_scale();
        let want = internal_size(self.size, scale);
        if want != self.internal {
            gs.renderer.resize(&gs.gpu, want);
            self.internal = want;
        }
    }

    fn render_scale(&self) -> f32 {
        self.quality
            .fixed_scale()
            .unwrap_or_else(|| self.governor.scale())
    }

    /// Acquires the next surface texture, handling the failures; `None`
    /// skips this frame.
    fn acquire(&mut self, gs: &GpuState) -> Option<wgpu::SurfaceTexture> {
        if self.surface_dirty {
            gs.configure(self.size);
            self.surface_dirty = false;
        }
        match gs.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => {
                self.lost_streak = 0;
                Some(t)
            }
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                self.surface_dirty = true;
                Some(t)
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                tracing::debug!("surface acquire timed out; skipping a frame");
                None
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                self.last_wall = None;
                self.idle_wait();
                None
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface_dirty = true;
                None
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                // A lost device is noticed by the loop; otherwise reconfigure.
                if self.lost_streak == 0 && !gs.is_lost() {
                    tracing::warn!("the surface was lost; reconfiguring it");
                }
                self.lost_streak = self.lost_streak.saturating_add(1);
                self.surface_dirty = true;
                None
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                tracing::warn!("surface acquire failed validation; reconfiguring");
                self.surface_dirty = true;
                None
            }
        }
    }

    /// Renders one frame on the current GPU state (taken out of `self` for
    /// the frame, so the borrow checker sees it apart from the rest).
    fn frame(&mut self, wall: f64) {
        let Some(mut gs) = self.gs.take() else {
            return;
        };
        self.frame_on(&mut gs, wall);
        self.gs = Some(gs);
    }

    fn frame_on(&mut self, gs: &mut GpuState, wall: f64) {
        let Some(frame) = self.acquire(gs) else {
            return;
        };
        // Keep the cadence; after falling behind, start again from now.
        let period = self.budget_ms / 1000.0;
        self.next_frame_due =
            Some(self.next_frame_due.map_or(wall, |d| d.max(wall - period)) + period);
        let cpu_start = Instant::now();
        if let Some(t) = &mut gs.timer {
            t.begin(&gs.gpu);
        }

        self.audio.drain(&mut self.hist);
        self.hist.fill_silence_until(wall);

        let dt = self
            .last_wall
            .map_or(0.0, |last| (wall - last).clamp(0.0, MAX_DT));
        self.last_wall = Some(wall);
        self.sim_time += dt;
        let ticks = self.sim.advance(dt);
        self.apply_internal_size(gs);

        let period = 1.0 / SIM_HZ;
        let aspect = aspect(self.size);
        let av_offset = f64::from(self.av_offset_ms) / 1000.0;
        for _ in 0..ticks.run {
            self.tick_time += period;
            // The tick's moment on the app clock (ticks trail the frame).
            let audio_time = (wall - (self.sim_time - self.tick_time)).min(wall);
            let audio = self
                .analyzer
                .tick(&self.hist, audio_time, av_offset, period);
            gs.renderer.set_spectrum(&gs.gpu, &audio.spectrum);
            self.director.update(self.tick_time, audio, &gs.loader);
            let base = TickInputBase {
                fps: SIM_HZ,
                audio,
                aspect,
                pixels: gs.renderer.internal_size(),
            };
            self.director.tick_and_render(
                self.tick_time,
                &base,
                &mut self.globals,
                &mut gs.renderer,
                &gs.gpu,
            );
        }
        if ticks.dropped > 0 {
            tracing::debug!("dropped {} simulation ticks", ticks.dropped);
            self.tick_time += f64::from(ticks.dropped) * period;
        }

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.director
            .composite(self.sim_time, &mut gs.renderer, &gs.gpu, &view, gs.format);

        let shown = self.shown_id();
        self.overlay.set_shown(&shown, wall);
        let target = [frame.texture.width(), frame.texture.height()];
        let debug = if self.overlay.show_debug {
            self.debug_line(gs, &shown)
        } else {
            String::new()
        };
        let text_scale = self.dpi_scale.round().clamp(1.0, 4.0) as u32;
        let items = self.overlay.items(wall, target, text_scale, &debug);
        if !items.is_empty() {
            gs.renderer.draw_text(&gs.gpu, &view, target, &items);
        }
        if let Some(t) = &mut gs.timer {
            t.end(&gs.gpu);
        }
        drop(view);
        gs.gpu.queue.present(frame);
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        self.feed_governor(gs, wall, cpu_ms);
        self.first_frame.get_or_insert(wall);
        if self.stats.frame(wall) && self.log_fps {
            eprintln!(
                "fps={:.1} frame_ms={:.2} scale={:.3} preset={shown}",
                self.stats.fps,
                self.stats.frame_ms,
                self.render_scale()
            );
        }
    }

    /// Feeds the governor the GPU frame times that have arrived (or, without
    /// timestamp queries, this frame's CPU time), against one refresh period
    /// of the window's monitor.
    fn feed_governor(&mut self, gs: &mut GpuState, wall: f64, cpu_ms: f64) {
        if wall >= self.next_budget_check {
            self.next_budget_check = wall + BUDGET_CHECK_S;
            // A plain Win32 query on Windows; it does not wait on the main thread.
            if let Some(mhz) = self
                .window
                .current_monitor()
                .and_then(|m| m.refresh_rate_millihertz())
            {
                self.budget_ms = (1.0e6 / f64::from(mhz)).clamp(1000.0 / 360.0, 1000.0 / 24.0);
            }
        }
        let mut timings = std::mem::take(&mut self.timings);
        timings.clear();
        match &mut gs.timer {
            Some(t) => t.collect(&gs.gpu, &mut timings),
            None => timings.push(cpu_ms),
        }
        for &ms in &timings {
            self.gpu_ms = Some(ms);
            if let Some(scale) = self.governor.record(wall, ms, self.budget_ms) {
                tracing::info!("quality governor: render scale {scale}");
            }
        }
        self.timings = timings;
    }

    fn debug_line(&self, gs: &GpuState, shown: &str) -> String {
        let [w, h] = gs.renderer.internal_size();
        let gpu = self
            .gpu_ms
            .map_or_else(|| "-".to_string(), |ms| format!("{ms:.1}"));
        format!(
            "{:.0} fps  {:.1} ms  gpu {gpu} ms  {w}x{h}  scale {:.3}  {}  {}{}",
            self.stats.fps,
            self.stats.frame_ms,
            self.render_scale(),
            self.quality.name(),
            display_name(shown),
            if self.director.locked() {
                "  [locked]"
            } else {
                ""
            },
        )
    }
}
