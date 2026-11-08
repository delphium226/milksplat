//! The main thread (spec §3.2, §9.1): command line, logging, the window and
//! the winit event loop. It never renders; it forwards input and window
//! events to the render thread and handles fullscreen, monitors and quit.
//!
//! Startup order: arguments, paths (creating the directories), logging,
//! the config, the panic hook, the run marker, the config saver, then the
//! event loop. On exit: the render thread is joined, the window state is
//! saved, the saver finishes, the marker is removed (only if nothing
//! crashed), and after a crash the message box is shown and the exit code
//! is 1. Keep-awake requests are made here, on the main thread.

use std::ffi::OsString;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use ms_render::Gpu;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::ModifiersState;
use winit::monitor::MonitorHandle;
use winit::window::{Fullscreen, Window, WindowId};

use crate::config::{load_config, Config, ConfigStore, WindowState};
use crate::crash::{install_panic_hook, report_escaped_panic, show_crash_box, RunMarker};
use crate::hotkeys::{map_key, Action};
use crate::keepawake::KeepAwake;
use crate::paths::Paths;
use crate::render_thread::{run_render_thread, RenderExit, RenderOptions, ToRender};

/// How long the main thread waits for the render thread on quit.
const JOIN_TIMEOUT: Duration = Duration::from_secs(3);
/// Two left clicks within this make a double-click.
pub const DOUBLE_CLICK: Duration = Duration::from_millis(400);

const USAGE: &str = "\
Usage: milksplat [options]

Options:
  --presets <dir>        bundled preset directory (default: presets\\ next to the exe)
  --exit-after <secs>    quit after this many seconds of rendering
  --log-fps              print fps, frame time, scale and preset to stderr every second
  --safe                 safe mode: the built-in preset only, no user shaders,
                         the default audio device
  -h, --help             show this help";

/// Command-line options.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Args {
    pub presets: Option<PathBuf>,
    pub exit_after: Option<f64>,
    pub log_fps: bool,
    /// `--safe`.
    pub safe: bool,
    /// Hidden `--debug-lose-device <secs>`: destroy the GPU device after
    /// this many seconds of rendering, to exercise recovery.
    pub debug_lose_device: Option<f64>,
}

impl Args {
    /// Parses the arguments (without the program name). `Ok(None)` means
    /// help was asked for.
    pub fn parse<I: IntoIterator<Item = OsString>>(args: I) -> Result<Option<Args>, String> {
        let mut out = Args::default();
        let mut it = args.into_iter();
        while let Some(arg) = it.next() {
            let name = arg.to_string_lossy().into_owned();
            match name.as_str() {
                "-h" | "--help" => return Ok(None),
                "--log-fps" => out.log_fps = true,
                "--safe" => out.safe = true,
                "--presets" => {
                    let dir = it.next().ok_or("--presets needs a directory")?;
                    out.presets = Some(PathBuf::from(dir));
                }
                "--exit-after" => out.exit_after = Some(seconds(&name, it.next())?),
                "--debug-lose-device" => out.debug_lose_device = Some(seconds(&name, it.next())?),
                _ => return Err(format!("unknown argument: {name}")),
            }
        }
        Ok(Some(out))
    }
}

fn seconds(flag: &str, value: Option<OsString>) -> Result<f64, String> {
    let value = value.ok_or_else(|| format!("{flag} needs a number of seconds"))?;
    let text = value.to_string_lossy();
    match text.trim().parse::<f64>() {
        Ok(s) if s.is_finite() && s > 0.0 => Ok(s),
        _ => Err(format!("{flag}: not a positive number of seconds: {text}")),
    }
}

/// Recognises double-clicks: two presses within [`DOUBLE_CLICK`].
#[derive(Debug, Default)]
pub struct DoubleClick {
    last: Option<Instant>,
}

impl DoubleClick {
    /// Records a press at `now`; true if it completes a double-click (the
    /// next press then starts a new pair).
    pub fn press(&mut self, now: Instant) -> bool {
        match self.last.take() {
            Some(prev) if now.saturating_duration_since(prev) <= DOUBLE_CLICK => true,
            _ => {
                self.last = Some(now);
                false
            }
        }
    }
}

/// The app entry point (see `main.rs`).
pub fn main() -> ExitCode {
    let args = match Args::parse(std::env::args_os().skip(1)) {
        Ok(Some(args)) => args,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("milksplat: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let mut paths = Paths::discover(args.presets.clone());
    let dir_warnings = paths.create_dirs();
    // Flushes the log file when dropped, at the end of `main`.
    let _log_guard = crate::logging::init_logging(&paths.logs);
    tracing::info!("milksplat {} starting", env!("CARGO_PKG_VERSION"));
    for w in dir_warnings {
        tracing::warn!("{w}");
    }
    tracing::info!(
        "config {}, logs {}",
        paths.config.display(),
        paths.logs.display()
    );
    let (cfg, warnings) = load_config(&paths.config);
    for w in warnings {
        tracing::warn!("{w}");
    }
    let current_preset = Arc::new(Mutex::new(None));
    install_panic_hook(&paths, Arc::clone(&current_preset));
    let (marker, crashed_preset) = RunMarker::begin(&paths.logs);
    let (store, saver) = ConfigStore::start(paths.config.clone(), cfg.clone());

    let event_loop = match EventLoop::<UserEvent>::with_user_event().build() {
        Ok(event_loop) => event_loop,
        Err(e) => {
            tracing::error!("cannot create the event loop: {e}");
            marker.end();
            return ExitCode::FAILURE;
        }
    };
    let marker = Arc::new(marker);
    let mut app = App::new(cfg, paths, args, event_loop.create_proxy());
    app.store = Some(store);
    app.marker = Some(Arc::clone(&marker));
    app.current_preset = current_preset;
    app.crashed_preset = crashed_preset;
    let app_ref = &mut app;
    // winit hands a panic in an event handler back from `run_app`.
    match catch_unwind(AssertUnwindSafe(move || event_loop.run_app(app_ref))) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::error!("the event loop failed: {e}");
            app.failed = true;
        }
        Err(_) => {
            tracing::error!("the main thread panicked");
            app.crash = Some(report_escaped_panic("the main thread panicked", false));
            app.send(ToRender::Quit);
        }
    }
    let code = app.finish();
    saver.finish();
    // The marker stays after a crash, so the next run skips the preset.
    // The render thread (joined by `finish`, unless it hung) and `app` hold
    // the other references.
    let crash = app.crash.take();
    drop(app.marker.take());
    match (&crash, Arc::try_unwrap(marker)) {
        (None, Ok(marker)) => marker.end(),
        (None, Err(_)) => tracing::warn!(
            "the render thread did not stop, so the run marker stays; the next run will skip the \
             preset it names"
        ),
        (Some(_), _) => {}
    }
    if let Some(report) = crash {
        show_crash_box(report.as_deref());
        return ExitCode::FAILURE;
    }
    code
}

/// Events the render thread sends the main thread.
#[derive(Debug)]
pub enum UserEvent {
    /// The render thread has ended.
    RenderExited(RenderExit),
    /// After a device loss the render thread needs a new window surface;
    /// winit only gives out the window handle on the main thread.
    NeedSurface,
    /// 10 s of silence started (`true`) or ended: keep-awake follows.
    Silence(bool),
}

struct App {
    cfg: Config,
    paths: Paths,
    args: Args,
    proxy: EventLoopProxy<UserEvent>,
    tx: Sender<ToRender>,
    /// Handed to the render thread when it starts.
    rx: Option<Receiver<ToRender>>,
    window: Option<Arc<Window>>,
    /// The render thread's wgpu instance, for surfaces after a device loss.
    instance: Option<wgpu::Instance>,
    render: Option<std::thread::JoinHandle<()>>,
    mods: ModifiersState,
    clicks: DoubleClick,
    /// Set once quitting has begun: when to stop waiting for the render thread.
    quit_deadline: Option<Instant>,
    failed: bool,
    /// Set when a panic escaped (render or main thread): the crash report's
    /// path, if it was written.
    crash: Option<Option<PathBuf>>,
    store: Option<ConfigStore>,
    marker: Option<Arc<RunMarker>>,
    current_preset: Arc<Mutex<Option<String>>>,
    crashed_preset: Option<String>,
    /// The last windowed (not fullscreen, not minimised) size and
    /// position, saved on exit.
    windowed: WindowState,
    /// Whether the render thread reports 10 s of silence.
    silent: bool,
    keep_awake: KeepAwake,
}

impl App {
    fn new(cfg: Config, paths: Paths, args: Args, proxy: EventLoopProxy<UserEvent>) -> App {
        let (tx, rx) = crossbeam_channel::unbounded();
        App {
            windowed: cfg.window.clone(),
            cfg,
            paths,
            args,
            proxy,
            tx,
            rx: Some(rx),
            window: None,
            instance: None,
            render: None,
            mods: ModifiersState::empty(),
            clicks: DoubleClick::default(),
            quit_deadline: None,
            failed: false,
            crash: None,
            store: None,
            marker: None,
            current_preset: Arc::new(Mutex::new(None)),
            crashed_preset: None,
            silent: false,
            keep_awake: KeepAwake::default(),
        }
    }

    /// Creates the window and its surface and starts the render thread.
    fn start(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let ws = &self.cfg.window;
        let mut attrs = Window::default_attributes()
            .with_title("Milksplat")
            .with_inner_size(LogicalSize::new(ws.size[0].max(1), ws.size[1].max(1)))
            .with_min_inner_size(LogicalSize::new(160, 120));
        // A saved position is only used if it is on a connected monitor.
        if let Some([x, y]) = ws.position.filter(|&p| on_a_monitor(event_loop, p)) {
            attrs = attrs.with_position(PhysicalPosition::new(x, y));
        }
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .map_err(|e| format!("cannot create the window: {e}"))?,
        );
        let instance = Gpu::create_instance();
        let surface = instance
            .create_surface(Arc::clone(&window))
            .map_err(|e| format!("cannot create the window surface: {e}"))?;
        let rx = self
            .rx
            .take()
            .ok_or_else(|| "the render thread was already started".to_string())?;
        let size = window.inner_size();
        let proxy = self.proxy.clone();
        let surface_proxy = self.proxy.clone();
        let silence_proxy = self.proxy.clone();
        let store = self
            .store
            .clone()
            .ok_or_else(|| "the config store is missing".to_string())?;
        let opts = RenderOptions {
            paths: self.paths.clone(),
            initial_size: [size.width, size.height],
            initial_scale: window.scale_factor(),
            exit_after: self.args.exit_after,
            log_fps: self.args.log_fps,
            lose_device_after: self.args.debug_lose_device,
            on_exit: Box::new(move |exit| {
                // The loop may already be gone if the main thread quit first.
                let _ = proxy.send_event(UserEvent::RenderExited(exit));
            }),
            request_surface: Box::new(move || {
                let _ = surface_proxy.send_event(UserEvent::NeedSurface);
            }),
            on_silence: Box::new(move |silent| {
                let _ = silence_proxy.send_event(UserEvent::Silence(silent));
            }),
            safe: self.args.safe,
            store,
            marker: self.marker.clone(),
            current_preset: Arc::clone(&self.current_preset),
            crashed_preset: self.crashed_preset.take(),
        };
        self.instance = Some(instance.clone());
        let handle = run_render_thread(
            instance,
            surface,
            Arc::clone(&window),
            rx,
            self.cfg.clone(),
            opts,
        )
        .map_err(|e| format!("cannot start the render thread: {e}"))?;
        self.render = Some(handle);
        self.window = Some(window);
        Ok(())
    }

    /// Creates a surface for the window from the render thread's instance
    /// and sends it over. On failure the render thread asks again later.
    fn send_new_surface(&self) {
        let (Some(window), Some(instance)) = (&self.window, &self.instance) else {
            return;
        };
        match instance.create_surface(Arc::clone(window)) {
            Ok(surface) => self.send(ToRender::NewSurface(surface)),
            Err(e) => tracing::error!("cannot create a new window surface: {e}"),
        }
    }

    fn send(&self, msg: ToRender) {
        // A closed channel means the render thread has ended; its exit
        // event closes the app.
        let _ = self.tx.send(msg);
    }

    /// Asks the render thread to stop; the loop exits when it has (or after
    /// 3 s).
    fn begin_quit(&mut self, event_loop: &ActiveEventLoop) {
        if self.render.is_none() {
            event_loop.exit();
            return;
        }
        if self.quit_deadline.is_none() {
            self.send(ToRender::Quit);
            let deadline = Instant::now() + JOIN_TIMEOUT;
            self.quit_deadline = Some(deadline);
            event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
        }
    }

    fn action(&mut self, event_loop: &ActiveEventLoop, a: Action) {
        match a {
            Action::ToggleFullscreen => self.toggle_fullscreen(event_loop),
            Action::NextMonitor => self.next_monitor(event_loop),
            Action::EscapeKey => match &self.window {
                Some(w) if w.fullscreen().is_some() => w.set_fullscreen(None),
                _ => self.begin_quit(event_loop),
            },
            other => self.send(ToRender::Action(other)),
        }
    }

    fn toggle_fullscreen(&mut self, event_loop: &ActiveEventLoop) {
        let Some(w) = &self.window else {
            return;
        };
        if w.fullscreen().is_some() {
            w.set_fullscreen(None);
        } else {
            let monitor = self.fullscreen_monitor(event_loop, w);
            w.set_fullscreen(Some(Fullscreen::Borderless(monitor)));
        }
    }

    /// The configured fullscreen monitor if it is connected, else the
    /// window's.
    fn fullscreen_monitor(
        &self,
        event_loop: &ActiveEventLoop,
        w: &Window,
    ) -> Option<MonitorHandle> {
        self.cfg
            .fullscreen_monitor
            .as_ref()
            .and_then(|name| {
                event_loop
                    .available_monitors()
                    .find(|m| m.name().as_ref() == Some(name))
            })
            .or_else(|| w.current_monitor())
    }

    /// `M`: moves to the next monitor, fullscreen or centred there, and
    /// remembers it as the fullscreen monitor (saved in the config).
    fn next_monitor(&mut self, event_loop: &ActiveEventLoop) {
        let Some(w) = self.window.clone() else {
            return;
        };
        let monitors: Vec<MonitorHandle> = event_loop.available_monitors().collect();
        if monitors.len() < 2 {
            return;
        }
        let cur = w.current_monitor();
        let i = monitors
            .iter()
            .position(|m| Some(m) == cur.as_ref())
            .unwrap_or(0);
        let next = monitors[(i + 1) % monitors.len()].clone();
        self.cfg.fullscreen_monitor = next.name();
        if let Some(store) = &self.store {
            let monitor = self.cfg.fullscreen_monitor.clone();
            store.update(move |c| c.fullscreen_monitor = monitor);
        }
        if w.fullscreen().is_some() {
            w.set_fullscreen(Some(Fullscreen::Borderless(Some(next))));
        } else {
            let pos = next.position();
            let area = next.size();
            let outer = w.outer_size();
            let centre = |origin: i32, avail: u32, size: u32| {
                let gap = (i64::from(avail) - i64::from(size)).max(0) / 2;
                i32::try_from(i64::from(origin) + gap).unwrap_or(origin)
            };
            w.set_outer_position(PhysicalPosition::new(
                centre(pos.x, area.width, outer.width),
                centre(pos.y, area.height, outer.height),
            ));
        }
    }

    /// Remembers the window's size and position while it is windowed (not
    /// fullscreen or minimised), to save on exit.
    fn track_window(&mut self) {
        let Some(w) = &self.window else {
            return;
        };
        if w.fullscreen().is_some() || w.is_minimized() == Some(true) {
            return;
        }
        let size = w.inner_size();
        if size.width > 0 && size.height > 0 {
            let logical = size.to_logical::<f64>(w.scale_factor());
            self.windowed.size = [
                logical.width.round().max(1.0) as u32,
                logical.height.round().max(1.0) as u32,
            ];
        }
        // Windows parks minimised windows at -32000.
        if let Ok(pos) = w.outer_position().map(|p| [p.x, p.y]) {
            if pos.iter().all(|&v| v > -30_000) {
                self.windowed.position = Some(pos);
            }
        }
        if let Some(name) = w.current_monitor().and_then(|m| m.name()) {
            self.windowed.monitor = Some(name);
        }
    }

    /// Keeps the display awake while fullscreen with audio playing.
    fn update_keep_awake(&mut self) {
        let fullscreen = self
            .window
            .as_ref()
            .is_some_and(|w| w.fullscreen().is_some());
        let on = fullscreen && !self.silent && self.quit_deadline.is_none();
        self.keep_awake.update(on);
    }

    /// After the event loop: waits (bounded) for the render thread, saves
    /// the window state and gives the exit code.
    fn finish(&mut self) -> ExitCode {
        self.keep_awake.update(false);
        if let Some(handle) = self.render.take() {
            let deadline = self
                .quit_deadline
                .unwrap_or_else(|| Instant::now() + JOIN_TIMEOUT);
            while !handle.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            if handle.is_finished() {
                if handle.join().is_err() {
                    self.failed = true;
                }
            } else {
                tracing::error!(
                    "the render thread did not stop within {JOIN_TIMEOUT:?}; exiting anyway"
                );
                self.failed = true;
            }
        }
        self.track_window();
        if let Some(store) = &self.store {
            let window = self.windowed.clone();
            let monitor = self.cfg.fullscreen_monitor.clone();
            store.update(move |c| {
                c.window = window;
                c.fullscreen_monitor = monitor;
            });
        }
        self.instance = None;
        self.window = None;
        if self.failed || self.crash.is_some() {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() || self.failed {
            return;
        }
        if let Err(e) = self.start(event_loop) {
            tracing::error!("{e}");
            self.failed = true;
            event_loop.exit();
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::RenderExited(exit) => {
                match exit {
                    RenderExit::Clean => {}
                    RenderExit::Failed(_) => self.failed = true,
                    RenderExit::Crashed { report } => self.crash = Some(report),
                }
                event_loop.exit();
            }
            UserEvent::NeedSurface => self.send_new_surface(),
            UserEvent::Silence(silent) => self.silent = silent,
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.begin_quit(event_loop),
            WindowEvent::Resized(size) => {
                let scale = self.window.as_ref().map_or(1.0, |w| w.scale_factor());
                self.send(ToRender::Resize {
                    size: [size.width, size.height],
                    scale,
                });
                self.track_window();
            }
            WindowEvent::Moved(_) => self.track_window(),
            WindowEvent::Occluded(occluded) => self.send(ToRender::Suspend(occluded)),
            WindowEvent::ModifiersChanged(m) => self.mods = m.state(),
            WindowEvent::KeyboardInput {
                event,
                is_synthetic: false,
                ..
            } if event.state == ElementState::Pressed => {
                if let Some(a) = map_key(&event.logical_key, self.mods) {
                    if !event.repeat || a.repeats() {
                        self.action(event_loop, a);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if self.clicks.press(Instant::now()) => self.toggle_fullscreen(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.update_keep_awake();
        if let Some(deadline) = self.quit_deadline {
            if Instant::now() >= deadline {
                // `finish` reports it and sets the exit code.
                event_loop.exit();
            }
        }
    }
}

/// Whether a window placed at `pos` (physical) would have its top-left
/// corner area on a connected monitor.
fn on_a_monitor(event_loop: &ActiveEventLoop, [x, y]: [i32; 2]) -> bool {
    // Some of the title bar must be reachable.
    let (x, y) = (i64::from(x) + 40, i64::from(y) + 10);
    event_loop.available_monitors().any(|m| {
        let p = m.position();
        let s = m.size();
        let (left, top) = (i64::from(p.x), i64::from(p.y));
        x >= left && y >= top && x < left + i64::from(s.width) && y < top + i64::from(s.height)
    })
}
