//! The loader thread (spec §3.2): prepares presets off the render thread so
//! that switching never hitches.
//!
//! Per request, on the `ms-loader` thread (16 MB stack, for deeply nested
//! preset code):
//! 1. read the file (at most [`MAX_PRESET_BYTES`]; a larger file fails);
//! 2. `parse` (warnings are collected) and `compile_preset`;
//! 3. `RendererShared::compile_shaders` for the display format, with user
//!    textures from the [`TextureResolver`] and the preset's random picks
//!    from `PresetPicks::new(seed ^ PICKS_SALT)`;
//! 4. the warm-up: `RendererShared::warm_up` draws one warp pass and one
//!    composite with the new shaders into private 64x64 targets, inside
//!    error scopes, then waits for the GPU.
//!
//! The result is sent back as [`LoadResult::Ready`]; any error (including a
//! panic in a dependency, caught per request) becomes
//! [`LoadResult::Failed`] (with `device_lost` set if the GPU device was lost
//! at the time). Requests are answered in order. Every request gets
//! a ticket (`req`) that is unique across all loaders of the process, so a
//! caller can tell a stale answer from the one it is waiting for.

use std::cell::RefCell;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use ms_preset::{compile_preset, parse, CompiledPreset};
use ms_render::{PresetPicks, RendererShared, ShaderSet, TextureResolver};

/// Stack size of the loader thread (parsing and compiling deeply nested
/// expressions recurses).
pub const LOADER_STACK_BYTES: usize = 16 * 1024 * 1024;

/// Largest preset file the loader reads.
pub const MAX_PRESET_BYTES: u64 = 1024 * 1024;

/// Mixed into a preset's seed for its random texture picks, so they are a
/// different stream from the expression VM's (which uses the seed as is).
pub const PICKS_SALT: u64 = 0x5EED_7E47;

/// How long dropping a [`Loader`] waits for its thread before detaching it.
const JOIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Request tickets, unique across every loader in the process.
static NEXT_REQ: AtomicU64 = AtomicU64::new(1);
/// Loader ids, unique across every loader in the process.
static NEXT_LOADER: AtomicU64 = AtomicU64::new(1);

/// The answer to one [`Loader::request`].
pub enum LoadResult {
    /// Parsed, compiled, shaders compiled and warmed up.
    Ready {
        /// The ticket [`Loader::request`] returned.
        req: u64,
        id: String,
        preset: Arc<CompiledPreset>,
        /// Compiled with the loader's `RendererShared` for its display
        /// format; a failed stage is the built-in one, listed in `errors`.
        shaders: ShaderSet,
        /// Parse warnings and code-block compile errors (a failed block is
        /// skipped when the preset runs).
        warnings: Vec<String>,
    },
    /// The preset could not be prepared.
    Failed {
        req: u64,
        id: String,
        error: String,
        /// The GPU device was lost when the load failed: the preset itself
        /// is probably fine, so it should be tried again on a new device
        /// rather than skipped.
        device_lost: bool,
    },
}

impl LoadResult {
    /// The ticket of the request this answers.
    pub fn req(&self) -> u64 {
        match self {
            LoadResult::Ready { req, .. } | LoadResult::Failed { req, .. } => *req,
        }
    }

    /// The preset id of the request this answers.
    pub fn id(&self) -> &str {
        match self {
            LoadResult::Ready { id, .. } | LoadResult::Failed { id, .. } => id,
        }
    }
}

impl std::fmt::Debug for LoadResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadResult::Ready {
                req, id, warnings, ..
            } => f
                .debug_struct("Ready")
                .field("req", req)
                .field("id", id)
                .field("warnings", warnings)
                .finish_non_exhaustive(),
            LoadResult::Failed {
                req,
                id,
                error,
                device_lost,
            } => f
                .debug_struct("Failed")
                .field("req", req)
                .field("id", id)
                .field("error", error)
                .field("device_lost", device_lost)
                .finish(),
        }
    }
}

struct Job {
    req: u64,
    id: String,
    path: PathBuf,
    seed: u64,
}

/// State shared between the [`Loader`] handle and its thread.
#[derive(Default)]
struct Control {
    paused: Mutex<bool>,
    resume: Condvar,
    stop: AtomicBool,
    /// Name of the thread that ran the latest warm-up draw.
    warm_up_thread: Mutex<Option<String>>,
    /// Requests made and not yet answered (queued or in progress).
    outstanding: AtomicU64,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Everything the loader thread needs to prepare a preset.
struct Context {
    shared: RendererShared,
    resolver: TextureResolver,
    comp_format: wgpu::TextureFormat,
    /// Compile the preset's own warp and composite shaders (false in safe
    /// mode: every preset gets the built-in ones).
    user_shaders: bool,
    control: Arc<Control>,
}

/// Handle to the loader thread. Dropping it stops the thread: the request
/// in progress finishes, queued ones are dropped, and the thread is joined
/// (waiting at most 2 s, then detached with a warning).
pub struct Loader {
    jobs: Option<Sender<Job>>,
    results: Receiver<LoadResult>,
    /// Used to answer requests directly if the thread could not start.
    results_tx: Sender<LoadResult>,
    thread: Option<JoinHandle<()>>,
    control: Arc<Control>,
    requests: AtomicU64,
    id: u64,
}

impl Loader {
    /// Starts the `ms-loader` thread ([`LOADER_STACK_BYTES`] of stack).
    /// Shaders are compiled with `shared` (fetch it again, and spawn a new
    /// loader, after `Renderer::recreate`), for the display target format
    /// `comp_format`; user textures come from `resolver`.
    ///
    /// If the thread cannot be started, the error is logged and every
    /// request is answered with [`LoadResult::Failed`].
    pub fn spawn(
        shared: RendererShared,
        resolver: TextureResolver,
        comp_format: wgpu::TextureFormat,
    ) -> Self {
        Self::start(shared, resolver, comp_format, true)
    }

    /// Like [`spawn`](Self::spawn), for safe mode: the presets' own warp
    /// and composite shaders are ignored (compiled as `None`, so every
    /// preset gets the built-in ones); their code and parameters still run.
    pub fn spawn_safe(
        shared: RendererShared,
        resolver: TextureResolver,
        comp_format: wgpu::TextureFormat,
    ) -> Self {
        Self::start(shared, resolver, comp_format, false)
    }

    fn start(
        shared: RendererShared,
        resolver: TextureResolver,
        comp_format: wgpu::TextureFormat,
        user_shaders: bool,
    ) -> Self {
        let (jobs_tx, jobs_rx) = crossbeam_channel::unbounded::<Job>();
        let (results_tx, results_rx) = crossbeam_channel::unbounded();
        let control = Arc::new(Control::default());
        let ctx = Context {
            shared,
            resolver,
            comp_format,
            user_shaders,
            control: Arc::clone(&control),
        };
        let tx = results_tx.clone();
        let spawned = std::thread::Builder::new()
            .name("ms-loader".into())
            .stack_size(LOADER_STACK_BYTES)
            .spawn(move || run(&ctx, &jobs_rx, &tx));
        let (jobs, thread) = match spawned {
            Ok(handle) => (Some(jobs_tx), Some(handle)),
            Err(e) => {
                tracing::error!("cannot start the preset loader thread: {e}");
                (None, None)
            }
        };
        Loader {
            jobs,
            results: results_rx,
            results_tx,
            thread,
            control,
            requests: AtomicU64::new(0),
            id: NEXT_LOADER.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// This loader's id, unique in the process. A request made to one
    /// loader is never answered by another, so a caller that is handed a
    /// different loader knows to send its pending request again.
    pub fn loader_id(&self) -> u64 {
        self.id
    }

    /// Queues preset `id` (file `path`) for loading; `seed` seeds its random
    /// texture picks (see [`PICKS_SALT`]). Returns the request's ticket,
    /// which its [`LoadResult`] carries. Never blocks.
    pub fn request(&self, id: String, path: PathBuf, seed: u64) -> u64 {
        let req = NEXT_REQ.fetch_add(1, Ordering::Relaxed);
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.control.outstanding.fetch_add(1, Ordering::SeqCst);
        tracing::debug!("loader: request {req}: {id}");
        let job = Job {
            req,
            id,
            path,
            seed,
        };
        let unsent = match &self.jobs {
            Some(tx) => tx.send(job).err().map(|e| e.0),
            None => Some(job),
        };
        if let Some(job) = unsent {
            self.control.outstanding.fetch_sub(1, Ordering::SeqCst);
            let _ = self.results_tx.send(LoadResult::Failed {
                req,
                id: job.id,
                error: "the preset loader thread is not running".into(),
                device_lost: false,
            });
        }
        req
    }

    /// True when every request made has been answered (none is queued or
    /// in progress). Never blocks.
    pub fn idle(&self) -> bool {
        self.control.outstanding.load(Ordering::SeqCst) == 0
    }

    /// The next finished request, if any. Never blocks.
    pub fn poll(&self) -> Option<LoadResult> {
        self.results.try_recv().ok()
    }

    /// Test hook: while paused, the thread takes no new request.
    #[doc(hidden)]
    pub fn set_paused(&self, paused: bool) {
        *lock(&self.control.paused) = paused;
        self.control.resume.notify_all();
    }

    /// Test hook: how many requests were made.
    #[doc(hidden)]
    pub fn request_count(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    /// Test hook: the name of the thread that ran the latest warm-up draw.
    #[doc(hidden)]
    pub fn warm_up_thread(&self) -> Option<String> {
        lock(&self.control.warm_up_thread).clone()
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.control.stop.store(true, Ordering::SeqCst);
        self.set_paused(false);
        // Disconnects the job channel, which ends the thread's loop.
        self.jobs = None;
        let Some(thread) = self.thread.take() else {
            return;
        };
        let start = Instant::now();
        while !thread.is_finished() && start.elapsed() < JOIN_TIMEOUT {
            std::thread::sleep(Duration::from_millis(2));
        }
        if thread.is_finished() {
            if thread.join().is_err() {
                tracing::error!("the preset loader thread panicked");
            }
        } else {
            tracing::warn!("the preset loader thread did not stop within 2 s; detaching it");
        }
    }
}

/// The loader thread's loop.
fn run(ctx: &Context, jobs: &Receiver<Job>, results: &Sender<LoadResult>) {
    while let Ok(job) = jobs.recv() {
        {
            let mut paused = lock(&ctx.control.paused);
            while *paused && !ctx.control.stop.load(Ordering::SeqCst) {
                paused = ctx
                    .control
                    .resume
                    .wait(paused)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        }
        if ctx.control.stop.load(Ordering::SeqCst) {
            break;
        }
        let result =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load(ctx, &job))) {
                Ok(Ok((preset, shaders, warnings))) => LoadResult::Ready {
                    req: job.req,
                    id: job.id,
                    preset,
                    shaders,
                    warnings,
                },
                Ok(Err(error)) => LoadResult::Failed {
                    req: job.req,
                    id: job.id,
                    error,
                    device_lost: ctx.shared.is_lost(),
                },
                Err(payload) => {
                    let msg = payload
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| payload.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic");
                    LoadResult::Failed {
                        req: job.req,
                        id: job.id,
                        error: format!("the loader panicked: {msg}"),
                        device_lost: ctx.shared.is_lost(),
                    }
                }
            };
        let sent = results.send(result).is_ok();
        ctx.control.outstanding.fetch_sub(1, Ordering::SeqCst);
        if !sent {
            break;
        }
    }
}

type Loaded = (Arc<CompiledPreset>, ShaderSet, Vec<String>);

/// Prepares one preset (see the module docs).
fn load(ctx: &Context, job: &Job) -> Result<Loaded, String> {
    let bytes = read_capped(&job.path)?;
    let (preset, parse_warnings) =
        parse(&bytes).map_err(|e| format!("{}: {e}", job.path.display()))?;
    let preset = compile_preset(preset);
    let mut warnings: Vec<String> = parse_warnings.into_iter().map(|w| w.message).collect();
    warnings.extend(
        preset
            .errors
            .iter()
            .map(|e| format!("{}: {}", e.block, e.error)),
    );

    let picks = RefCell::new(PresetPicks::new(job.seed ^ PICKS_SALT));
    let tex = |name: &str| ctx.resolver.resolve_with(name, &mut picks.borrow_mut());
    let shader = &preset.source.shader;
    let (warp, comp) = if ctx.user_shaders {
        (shader.warp.as_deref(), shader.comp.as_deref())
    } else {
        (None, None)
    };
    let shaders = ctx
        .shared
        .compile_shaders(warp, comp, ctx.comp_format, &tex);
    let feedback = feedback_format(preset.source.params.feedback_format);
    ctx.shared
        .warm_up(&shaders, feedback, ctx.comp_format)
        .map_err(|e| format!("{}: {e}", job.id))?;
    *lock(&ctx.control.warm_up_thread) = std::thread::current().name().map(str::to_string);
    Ok((Arc::new(preset), shaders, warnings))
}

/// Reads `path`, failing if it is larger than [`MAX_PRESET_BYTES`].
fn read_capped(path: &Path) -> Result<Vec<u8>, String> {
    let fail = |e: std::io::Error| format!("{}: {e}", path.display());
    let file = std::fs::File::open(path).map_err(fail)?;
    let too_large = || {
        format!(
            "{}: file too large (over {MAX_PRESET_BYTES} bytes)",
            path.display()
        )
    };
    if file.metadata().map_err(fail)?.len() > MAX_PRESET_BYTES {
        return Err(too_large());
    }
    let mut bytes = Vec::new();
    file.take(MAX_PRESET_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(fail)?;
    if bytes.len() as u64 > MAX_PRESET_BYTES {
        return Err(too_large());
    }
    Ok(bytes)
}

/// The renderer's feedback format for a preset's `feedback_format`.
pub fn feedback_format(f: ms_preset::FeedbackFormat) -> ms_render::FeedbackFormat {
    match f {
        ms_preset::FeedbackFormat::Rgba8 => ms_render::FeedbackFormat::Rgba8,
        ms_preset::FeedbackFormat::Fp16 => ms_render::FeedbackFormat::Fp16,
    }
}
