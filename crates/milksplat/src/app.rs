//! The main thread (spec §3.2, §9.1): command line, logging, the window and
//! the winit event loop. It never renders; it forwards input and window
//! events to the render thread and handles fullscreen, monitors and quit.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
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

use crate::config::Config;
use crate::hotkeys::{map_key, Action};
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
  -h, --help             show this help";

/// Command-line options.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Args {
    pub presets: Option<PathBuf>,
    pub exit_after: Option<f64>,
    pub log_fps: bool,
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

/// Sends logs to stderr: info and up, and only warnings from wgpu, naga
/// and winit. Task 24 adds the log files.
pub fn init_logging() {
    use std::io::IsTerminal;
    use tracing::Level;
    use tracing_subscriber::filter::Targets;
    use tracing_subscriber::prelude::*;

    let filter = Targets::new()
        .with_default(Level::INFO)
        .with_target("wgpu", Level::WARN)
        .with_target("wgpu_core", Level::WARN)
        .with_target("wgpu_hal", Level::WARN)
        .with_target("naga", Level::WARN)
        .with_target("winit", Level::WARN);
    let layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal());
    let _ = tracing_subscriber::registry()
        .with(layer)
        .with(filter)
        .try_init();
}

/// The app entry point (see `main.rs`).
pub fn main() -> ExitCode {
    init_logging();
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
    let cfg = Config::default();
    let paths = Paths::discover(args.presets.clone());
    tracing::info!("milksplat {} starting", env!("CARGO_PKG_VERSION"));

    let event_loop = match EventLoop::<UserEvent>::with_user_event().build() {
        Ok(event_loop) => event_loop,
        Err(e) => {
            tracing::error!("cannot create the event loop: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut app = App::new(cfg, paths, args, event_loop.create_proxy());
    if let Err(e) = event_loop.run_app(&mut app) {
        tracing::error!("the event loop failed: {e}");
        app.failed = true;
    }
    app.finish()
}

/// Events the render thread sends the main thread.
#[derive(Debug)]
pub enum UserEvent {
    /// The render thread has ended (`clean` is false after a fatal error or
    /// a panic).
    RenderExited { clean: bool },
    /// After a device loss the render thread needs a new window surface;
    /// winit only gives out the window handle on the main thread.
    NeedSurface,
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
}

impl App {
    fn new(cfg: Config, paths: Paths, args: Args, proxy: EventLoopProxy<UserEvent>) -> App {
        let (tx, rx) = crossbeam_channel::unbounded();
        App {
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
        }
    }

    /// Creates the window and its surface and starts the render thread.
    fn start(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let ws = &self.cfg.window;
        let mut attrs = Window::default_attributes()
            .with_title("Milksplat")
            .with_inner_size(LogicalSize::new(ws.size[0].max(1), ws.size[1].max(1)))
            .with_min_inner_size(LogicalSize::new(160, 120));
        if let Some([x, y]) = ws.position {
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
        let opts = RenderOptions {
            paths: self.paths.clone(),
            initial_size: [size.width, size.height],
            initial_scale: window.scale_factor(),
            exit_after: self.args.exit_after,
            log_fps: self.args.log_fps,
            lose_device_after: self.args.debug_lose_device,
            on_exit: Box::new(move |exit| {
                let clean = exit == RenderExit::Clean;
                // The loop may already be gone if the main thread quit first.
                let _ = proxy.send_event(UserEvent::RenderExited { clean });
            }),
            request_surface: Box::new(move || {
                let _ = surface_proxy.send_event(UserEvent::NeedSurface);
            }),
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
    /// remembers it as the fullscreen monitor (Task 24 saves it).
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

    /// After the event loop: waits (bounded) for the render thread and
    /// gives the exit code.
    fn finish(&mut self) -> ExitCode {
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
                tracing::warn!(
                    "the render thread did not stop within {JOIN_TIMEOUT:?}; exiting anyway"
                );
            }
        }
        self.instance = None;
        self.window = None;
        if self.failed {
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
            UserEvent::RenderExited { clean } => {
                if !clean {
                    self.failed = true;
                }
                event_loop.exit();
            }
            UserEvent::NeedSurface => self.send_new_surface(),
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
            }
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
        if let Some(deadline) = self.quit_deadline {
            if Instant::now() >= deadline {
                tracing::warn!("the render thread did not stop within {JOIN_TIMEOUT:?}");
                event_loop.exit();
            }
        }
    }
}
