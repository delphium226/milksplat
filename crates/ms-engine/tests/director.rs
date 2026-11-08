//! The loader thread and the director (spec §3.2, §6.7, §7): preparing
//! presets off the render thread, auto-advance, blends, hard cuts, locking,
//! history, failures and the device-loss blocklist.
//!
//! Time is simulated: every `update` gets an explicit `now`. Waiting for the
//! loader thread is a short polling loop with a 10 s real-time deadline.
//! The GPU parts run on the machine's real adapter, or on WARP with
//! `MS_FORCE_WARP=1`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use ms_analysis::AudioFrame;
use ms_engine::{
    Command, Director, DirectorConfig, Globals, LoadResult, Loader, State, TickInputBase,
    DEFAULT_PRESET_ID, LOADER_STACK_BYTES, MAX_PRESET_BYTES,
};
use ms_preset::{compile_preset, parse, Library, Prefs};
use ms_render::{read_texture, FeedbackFormat, Gpu, RenderConfig, Renderer, TextureResolver};

const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEADLINE: Duration = Duration::from_secs(10);
const DT: f64 = 1.0 / 60.0;

// ---------------------------------------------------------------- fixtures

/// A temp directory of preset files, removed on drop.
struct Fixture {
    dir: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn fixture(files: &[(&str, &str)]) -> (Fixture, Library) {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "ms-director-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (name, text) in files {
        std::fs::write(dir.join(name), text).unwrap();
    }
    let lib = Library::scan(std::slice::from_ref(&dir));
    (Fixture { dir }, lib)
}

fn good(name: &str) -> String {
    format!("version = 1\n[meta]\nname = \"{name}\"\n[params]\nzoom = 1.02\nwave_a = 0.9\n")
}

const BAD_TOML: &str = "version = = 1\n[meta\n";
const BAD_SHADER: &str = "version = 1\n[shader]\nwarp = \"this is not wgsl at all\"\n";
const FP16: &str = "version = 1\n[params]\nfeedback_format = \"fp16\"\n";

fn gpu() -> Gpu {
    match Gpu::headless() {
        Ok(gpu) => gpu,
        Err(e) => panic!("Gpu::headless() failed (no hardware adapter and no WARP): {e}"),
    }
}

fn renderer(gpu: &Gpu) -> Renderer {
    Renderer::new(
        gpu,
        RenderConfig {
            internal_size: [64, 64],
            format: FeedbackFormat::Rgba8,
        },
    )
}

fn cfg() -> DirectorConfig {
    DirectorConfig::default()
}

/// A long auto-advance and no hard cuts: only commands change the preset.
fn manual_cfg() -> DirectorConfig {
    DirectorConfig {
        auto_advance_s: 1.0e6,
        hard_cuts: false,
        ..DirectorConfig::default()
    }
}

struct Rig {
    director: Director,
    loader: Loader,
    renderer: Renderer,
    gpu: Gpu,
}

fn make_rig(cfg: DirectorConfig, lib: Library, prefs: Prefs) -> Rig {
    make_rig_seeded(cfg, lib, prefs, 7)
}

fn make_rig_seeded(cfg: DirectorConfig, lib: Library, prefs: Prefs, seed: u64) -> Rig {
    let gpu = gpu();
    let renderer = renderer(&gpu);
    let loader = Loader::spawn(renderer.shared(), TextureResolver::new(Vec::new()), FMT);
    let director = Director::new(cfg, lib, prefs, seed, &renderer.shared(), FMT);
    Rig {
        director,
        loader,
        renderer,
        gpu,
    }
}

impl Rig {
    fn update(&mut self, now: f64) {
        self.director
            .update(now, &AudioFrame::default(), &self.loader);
    }

    fn update_beat(&mut self, now: f64, beat_strength: f32) {
        let audio = AudioFrame {
            beat_strength,
            ..AudioFrame::default()
        };
        self.director.update(now, &audio, &self.loader);
    }

    /// Updates at a fixed `now` until `cond` holds (10 s real-time deadline).
    fn wait(&mut self, now: f64, what: &str, cond: impl Fn(&Director) -> bool) {
        let start = Instant::now();
        loop {
            self.update(now);
            if cond(&self.director) {
                return;
            }
            assert!(start.elapsed() < DEADLINE, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn wait_ready(&mut self, now: f64) {
        self.wait(now, "the next preset to be ready", |d| {
            d.next_ready().is_some()
        });
    }

    fn wait_showing(&mut self, now: f64, id: &str) {
        self.wait(now, &format!("{id} to be shown"), |d| shown(d) == id);
    }
}

/// The preset on screen: the current one, or the incoming one of a blend.
fn shown(d: &Director) -> String {
    match d.current() {
        State::Showing { cur } => cur,
        State::Blending { to, .. } => to,
    }
}

fn showing(id: &str) -> State {
    State::Showing {
        cur: id.to_string(),
    }
}

fn wait_result(loader: &Loader) -> LoadResult {
    let start = Instant::now();
    loop {
        if let Some(r) = loader.poll() {
            return r;
        }
        assert!(
            start.elapsed() < DEADLINE,
            "timed out waiting for the loader"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn path_of(lib: &Library, id: &str) -> PathBuf {
    lib.entries
        .iter()
        .find(|e| e.id == id)
        .unwrap()
        .path
        .clone()
}

// ---------------------------------------------------------------- loader

#[test]
fn warmup_runs_off_thread() {
    let (_fx, lib) = fixture(&[("a.splat", &good("a"))]);
    let gpu = gpu();
    let r = renderer(&gpu);
    let loader = Loader::spawn(r.shared(), TextureResolver::new(Vec::new()), FMT);
    assert_eq!(loader.warm_up_thread(), None);
    let req = loader.request("a.splat".into(), path_of(&lib, "a.splat"), 1);
    // The render thread does nothing but poll.
    match wait_result(&loader) {
        LoadResult::Ready {
            req: got,
            id,
            preset,
            shaders,
            warnings,
        } => {
            assert_eq!((got, id.as_str()), (req, "a.splat"));
            assert_eq!(preset.source.meta.name, "a");
            assert!(shaders.errors.is_empty(), "{:?}", shaders.errors);
            assert!(warnings.is_empty(), "{warnings:?}");
        }
        LoadResult::Failed { error, .. } => panic!("load failed: {error}"),
    }
    let name = loader.warm_up_thread().expect("the warm-up draw ran");
    assert_eq!(name, "ms-loader");
    assert_ne!(Some(name.as_str()), std::thread::current().name());
    assert_eq!(loader.request_count(), 1);
}

#[test]
fn oversized_preset_fails() {
    let big = format!("version = 1\n# {}\n", "x".repeat(MAX_PRESET_BYTES as usize));
    let (_fx, lib) = fixture(&[("big.splat", &big)]);
    let gpu = gpu();
    let r = renderer(&gpu);
    let loader = Loader::spawn(r.shared(), TextureResolver::new(Vec::new()), FMT);
    loader.request("big.splat".into(), path_of(&lib, "big.splat"), 1);
    match wait_result(&loader) {
        LoadResult::Failed { id, error, .. } => {
            assert_eq!(id, "big.splat");
            assert!(error.contains("too large"), "{error}");
        }
        LoadResult::Ready { .. } => panic!("a preset over the size cap loaded"),
    }
    // A missing file fails too, and the loader keeps working.
    loader.request(
        "gone.splat".into(),
        Path::new("no/such/file.splat").into(),
        1,
    );
    assert!(matches!(wait_result(&loader), LoadResult::Failed { .. }));
}

#[test]
fn loader_stack_runs_deep_presets() {
    const { assert!(LOADER_STACK_BYTES >= 16 * 1024 * 1024) };
    // A 1000-term chain overflows a 1 MB stack in a debug build.
    let chain = vec!["a"; 1000].join("+");
    let text = format!("version = 1\n[code]\nframe = '''\na = 1;\nq1 = {chain};\n'''\n");
    let (_fx, lib) = fixture(&[("deep.splat", &text)]);
    let gpu = gpu();
    let r = renderer(&gpu);
    let loader = Loader::spawn(r.shared(), TextureResolver::new(Vec::new()), FMT);
    loader.request("deep.splat".into(), path_of(&lib, "deep.splat"), 1);
    match wait_result(&loader) {
        LoadResult::Ready { preset, .. } => {
            assert!(preset.errors.is_empty(), "{:?}", preset.errors.len());
            assert!(preset.frame.is_some());
        }
        LoadResult::Failed { error, .. } => panic!("load failed: {error}"),
    }
}

#[test]
fn default_preset_parses_clean() {
    let text = include_str!("../../../presets/default.splat");
    let (preset, warnings) = parse(text.as_bytes()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(preset.meta.author, "Milksplat");
    assert_eq!(preset.meta.license, "MIT OR Apache-2.0");
    assert!(preset.shader.warp.is_none() && preset.shader.comp.is_none());
    assert_eq!(preset.params.param("zoom"), 1.01);
    assert_eq!(preset.params.param("decay"), 0.97);
    assert_eq!(preset.params.param("wave_mode"), 0.0);
    let compiled = compile_preset(preset);
    assert!(compiled.errors.is_empty(), "{:?}", compiled.errors.len());
    assert!(compiled.frame.is_some());
}

// ---------------------------------------------------------------- director

#[test]
fn empty_library_stays_on_default() {
    let mut rig = make_rig(cfg(), Library::default(), Prefs::default());
    for i in 0..3600 {
        rig.update(i as f64 * DT);
        if i % 600 == 0 {
            rig.director.command(Command::Next);
        }
    }
    assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
    assert_eq!(rig.loader.request_count(), 0);

    // Everything banned or blocklisted is the same as an empty library.
    let (_fx, lib) = fixture(&[("a.splat", &good("a")), ("b.splat", &good("b"))]);
    let prefs = Prefs {
        bans: ["a.splat".to_string()].into(),
        blocklist: ["b.splat".to_string()].into(),
        ..Prefs::default()
    };
    let mut rig = make_rig(cfg(), lib, prefs);
    for i in 0..3600 {
        rig.update(i as f64 * DT);
    }
    assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
    assert_eq!(rig.loader.request_count(), 0);
}

#[test]
fn waits_for_ready() {
    let (_fx, lib) = fixture(&[("a.splat", &good("a")), ("b.splat", &good("b"))]);
    let mut rig = make_rig(manual_cfg(), lib, Prefs::default());
    rig.loader.set_paused(true);
    rig.director.command(Command::Next);
    for i in 0..300 {
        rig.update(i as f64 * DT);
        assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
    }
    assert!(rig.loader.request_count() >= 1);
    // Once the loader runs, the requested preset blends in.
    rig.loader.set_paused(false);
    rig.wait(5.0, "the switch", |d| {
        matches!(d.current(), State::Blending { .. })
    });
    match rig.director.current() {
        State::Blending {
            from, start, dur, ..
        } => {
            assert_eq!(from, DEFAULT_PRESET_ID);
            assert_eq!((start, dur), (5.0, 2.7));
        }
        s => panic!("{s:?}"),
    }
}

#[test]
fn auto_advance_starts_blend() {
    let (_fx, lib) = fixture(&[
        ("a.splat", &good("a")),
        ("b.splat", &good("b")),
        ("c.splat", &good("c")),
    ]);
    let mut rig = make_rig(cfg(), lib, Prefs::default());
    rig.wait_ready(0.0);
    // 20 s +/- 25% puts the switch in [15, 25); simulate 30 s so the blend
    // that starts last (just before 25 s) also ends.
    let mut blends = Vec::new();
    let mut was_blending = false;
    let mut blend_ticks = 0;
    for i in 1..=(30 * 60) {
        let now = i as f64 * DT;
        rig.update(now);
        let s = rig.director.current();
        let blending = matches!(s, State::Blending { .. });
        if blending {
            blend_ticks += 1;
            if !was_blending {
                blends.push((now, s));
            }
        }
        was_blending = blending;
    }
    assert_eq!(blends.len(), 1, "{blends:?}");
    let (at, s) = &blends[0];
    assert!((15.0..25.0 + DT).contains(at), "blend started at {at}");
    match s {
        State::Blending {
            from,
            to,
            start,
            dur,
            ..
        } => {
            assert_eq!(from, DEFAULT_PRESET_ID);
            assert_ne!(to, DEFAULT_PRESET_ID);
            assert_eq!((*start, *dur), (*at, 2.7));
            // Showing again 2.7 s later.
            assert!((161..=163).contains(&blend_ticks), "{blend_ticks} ticks");
            assert_eq!(rig.director.current(), showing(to));
        }
        s => panic!("{s:?}"),
    }
}

#[test]
fn lock_prevents_advance() {
    let (_fx, lib) = fixture(&[("a.splat", &good("a")), ("b.splat", &good("b"))]);
    let mut rig = make_rig(cfg(), lib, Prefs::default());
    rig.director.command(Command::ToggleLock);
    rig.wait_ready(0.0);
    assert!(rig.director.locked());
    for i in 1..=(60 * 60) {
        // Strong beats all along: no hard cut either.
        rig.update_beat(i as f64 * DT, 5.0);
        assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
    }
    // Manual commands still work while locked.
    rig.director.command(Command::Next);
    rig.update(61.0);
    assert!(matches!(rig.director.current(), State::Blending { .. }));
    // Unlocking brings auto-advance back.
    rig.director.command(Command::ToggleLock);
    rig.update(61.0);
    assert!(!rig.director.locked());
}

#[test]
fn hard_cut_on_strong_beat() {
    let (_fx, lib) = fixture(&[
        ("a.splat", &good("a")),
        ("b.splat", &good("b")),
        ("c.splat", &good("c")),
    ]);
    let config = DirectorConfig {
        auto_advance_s: 1.0e6,
        ..DirectorConfig::default()
    };
    let mut rig = make_rig(config, lib, Prefs::default());
    rig.wait_ready(0.0);
    // Within the 10 s gap: no cut, however strong the beat.
    rig.update_beat(5.0, 10.0);
    assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
    // Not above the threshold: no cut.
    rig.update_beat(10.5, 2.0);
    assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
    // Above it after the gap: an instant switch, no blend.
    rig.update_beat(10.5, 2.5);
    let first = shown(&rig.director);
    assert_ne!(first, DEFAULT_PRESET_ID);
    assert_eq!(rig.director.current(), showing(&first));

    rig.wait_ready(10.5);
    rig.update_beat(15.0, 9.0);
    assert_eq!(rig.director.current(), showing(&first), "gap not respected");
    rig.update_beat(20.4, 9.0);
    assert_eq!(rig.director.current(), showing(&first), "gap not respected");
    rig.update_beat(20.6, 9.0);
    let second = shown(&rig.director);
    assert_ne!(second, first);
    assert_eq!(rig.director.current(), showing(&second));

    // Hard cuts can be turned off.
    let (_fx2, lib) = fixture(&[("a.splat", &good("a")), ("b.splat", &good("b"))]);
    let config = DirectorConfig {
        hard_cuts: false,
        auto_advance_s: 1.0e6,
        ..DirectorConfig::default()
    };
    let mut rig = make_rig(config, lib, Prefs::default());
    rig.wait_ready(0.0);
    rig.update_beat(50.0, 9.0);
    assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
}

#[test]
fn prev_returns_previous() {
    let (_fx, lib) = fixture(&[
        ("a.splat", &good("a")),
        ("b.splat", &good("b")),
        ("c.splat", &good("c")),
    ]);
    let mut rig = make_rig(manual_cfg(), lib, Prefs::default());
    rig.director.command(Command::Load("a.splat".into()));
    rig.wait_showing(1.0, "a.splat");
    rig.director.command(Command::Load("b.splat".into()));
    // The blend to `a` completes first; then `b` comes in.
    rig.wait_showing(5.0, "b.splat");
    rig.director.command(Command::Load("c.splat".into()));
    rig.wait_showing(10.0, "c.splat");
    rig.director.command(Command::Prev);
    rig.wait_showing(15.0, "b.splat");
    rig.director.command(Command::Prev);
    rig.wait_showing(20.0, "a.splat");
    // Nothing before the first: stays.
    rig.director.command(Command::Prev);
    for _ in 0..50 {
        rig.update(25.0);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(rig.director.current(), showing("a.splat"));
}

#[test]
fn switch_during_blend_waits_for_it() {
    let (_fx, lib) = fixture(&[
        ("a.splat", &good("a")),
        ("b.splat", &good("b")),
        ("c.splat", &good("c")),
    ]);
    let mut rig = make_rig(manual_cfg(), lib, Prefs::default());
    rig.director.command(Command::Load("a.splat".into()));
    rig.wait_showing(1.0, "a.splat");
    let blending_to_a = rig.director.current();
    assert!(matches!(blending_to_a, State::Blending { .. }));
    // `b` is ready during the blend: it waits; a newer request replaces it.
    rig.director.command(Command::Load("b.splat".into()));
    rig.wait(1.5, "b to be queued", |d| d.queued() == Some("b.splat"));
    rig.director.command(Command::Load("c.splat".into()));
    rig.wait(2.0, "c to be queued", |d| d.queued() == Some("c.splat"));
    assert_eq!(rig.director.current(), blending_to_a);
    // The blend to `a` completes (1.0 + 2.7), then `c` blends in at once.
    rig.update(3.6);
    assert_eq!(rig.director.current(), blending_to_a);
    rig.update(3.8);
    match rig.director.current() {
        State::Blending {
            from, to, start, ..
        } => assert_eq!(
            (from.as_str(), to.as_str(), start),
            ("a.splat", "c.splat", 3.8)
        ),
        s => panic!("{s:?}"),
    }
    assert_eq!(rig.director.queued(), None);
}

#[test]
fn zero_blend_switches_instantly() {
    for blend_s in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let (_fx, lib) = fixture(&[("a.splat", &good("a"))]);
        let config = DirectorConfig {
            blend_s,
            ..manual_cfg()
        };
        let mut rig = make_rig(config, lib, Prefs::default());
        rig.director.command(Command::Load("a.splat".into()));
        rig.wait(1.0, "the switch", |d| shown(d) == "a.splat");
        assert_eq!(
            rig.director.current(),
            showing("a.splat"),
            "blend_s {blend_s}"
        );
    }
}

#[test]
fn stale_ready_ignored() {
    let (_fx, lib) = fixture(&[
        ("a.splat", &good("a")),
        ("b.splat", &good("b")),
        ("c.splat", &good("c")),
    ]);
    let mut rig = make_rig(manual_cfg(), lib, Prefs::default());
    rig.loader.set_paused(true);
    rig.update(0.0);
    rig.director.command(Command::Load("a.splat".into()));
    rig.update(0.0);
    rig.director.command(Command::Load("b.splat".into()));
    rig.update(0.0);
    let requests = rig.loader.request_count();
    rig.loader.set_paused(false);
    rig.wait(0.0, "b", |d| shown(d) == "b.splat");
    // Keep polling until every queued request has been answered: `a` never
    // shows, even after its Ready arrived.
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(300) {
        rig.update(0.0);
        assert_ne!(shown(&rig.director), "a.splat");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(requests >= 2, "{requests}");
    match rig.director.current() {
        State::Blending { from, to, .. } => {
            assert_eq!((from.as_str(), to.as_str()), (DEFAULT_PRESET_ID, "b.splat"))
        }
        s => panic!("{s:?}"),
    }
}

#[test]
fn failed_preset_skipped() {
    let (_fx, lib) = fixture(&[("bad.splat", BAD_TOML), ("good.splat", &good("good"))]);
    let config = DirectorConfig {
        auto_advance_s: 4.0,
        blend_s: 0.5,
        ..manual_cfg()
    };
    // Several seeds, so both pick orders (bad first, good first) occur.
    for seed in 1..=4 {
        let mut rig = make_rig_seeded(config, lib.clone(), Prefs::default(), seed);
        // Whichever is picked first, only the good preset ever becomes ready.
        rig.wait_ready(0.0);
        assert_eq!(rig.director.next_ready(), Some("good.splat"));
        let mut seen_good = false;
        for i in 1..=(20 * 60) {
            rig.update(i as f64 * DT);
            let s = rig.director.current();
            let ids = match &s {
                State::Showing { cur } => vec![cur.clone()],
                State::Blending { from, to, .. } => vec![from.clone(), to.clone()],
            };
            assert!(!ids.iter().any(|id| id == "bad.splat"), "{s:?}");
            seen_good |= ids.iter().any(|id| id == "good.splat");
        }
        assert!(seen_good, "the director never moved on to the good preset");
        // The bad preset was requested (first, or as the pick after `good`),
        // failed, and was skipped: the director stays on `good`.
        let start = Instant::now();
        while rig.loader.request_count() < 2 {
            assert!(start.elapsed() < DEADLINE, "the bad preset was never tried");
            rig.update(20.0);
            std::thread::sleep(Duration::from_millis(1));
        }
        // Let its answer arrive, then a minute passes: nothing else to show.
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(300) {
            rig.update(20.0);
            std::thread::sleep(Duration::from_millis(1));
        }
        for i in 0..(60 * 60) {
            rig.update(20.0 + i as f64 * DT);
        }
        assert_eq!(rig.director.current(), showing("good.splat"));
        assert_eq!(rig.loader.request_count(), 2);
        // Asking for it by name fails too, and the current preset stays.
        rig.director.command(Command::Load("bad.splat".into()));
        for _ in 0..100 {
            rig.update(100.0);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(rig.director.current(), showing("good.splat"));
        assert_eq!(rig.loader.request_count(), 3);
    }
}

#[test]
fn consecutive_failures_capped() {
    let names: Vec<String> = (0..15).map(|i| format!("bad{i:02}.splat")).collect();
    let files: Vec<(&str, &str)> = names.iter().map(|n| (n.as_str(), BAD_TOML)).collect();
    let (_fx, lib) = fixture(&files);
    let mut rig = make_rig(cfg(), lib, Prefs::default());
    rig.wait(0.0, "10 failures", |d| d.consecutive_failures() == 10);
    assert_eq!(rig.loader.request_count(), 10);
    // No more requests for 10 s.
    for i in 0..(9 * 60) {
        rig.update(i as f64 * DT);
    }
    std::thread::sleep(Duration::from_millis(50));
    rig.update(9.9);
    assert_eq!(rig.loader.request_count(), 10);
    assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
    // Then it tries again.
    rig.update(10.1);
    assert_eq!(rig.loader.request_count(), 11);
    rig.wait(10.1, "the other 5 failures", |d| {
        d.consecutive_failures() == 5
    });
    assert_eq!(rig.loader.request_count(), 15);
    assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
    // Every file is now skipped: no more requests, ever.
    for i in 0..600 {
        rig.update(30.0 + i as f64 * 0.1);
    }
    assert_eq!(rig.loader.request_count(), 15);
}

#[test]
fn bad_shader_preset_still_shows() {
    let (_fx, lib) = fixture(&[("shader.splat", BAD_SHADER)]);
    let mut rig = make_rig(manual_cfg(), lib.clone(), Prefs::default());
    // The loader reports the shader error and falls back to the built-in.
    rig.loader
        .request("shader.splat".into(), path_of(&lib, "shader.splat"), 1);
    match wait_result(&rig.loader) {
        LoadResult::Ready { shaders, .. } => {
            assert!(!shaders.errors.is_empty(), "no shader error reported");
            assert!(shaders.warp_source.is_some());
        }
        LoadResult::Failed { error, .. } => panic!("load failed: {error}"),
    }
    rig.director.command(Command::Load("shader.splat".into()));
    rig.wait_showing(1.0, "shader.splat");
    // It renders (with the built-in warp in place of the broken one).
    let mut g = Globals::default();
    let audio = AudioFrame::default();
    let base = TickInputBase {
        fps: 60.0,
        audio: &audio,
        aspect: [1.0, 1.0],
        pixels: [64, 64],
    };
    for i in 0..10 {
        let now = 1.0 + i as f64 * DT;
        rig.update(now);
        rig.director
            .tick_and_render(now, &base, &mut g, &mut rig.renderer, &rig.gpu);
    }
    assert!(rig.gpu.poll_wait().is_ok());
}

#[test]
fn device_lost_blocklists_recent() {
    let (_fx, lib) = fixture(&[
        ("a.splat", &good("a")),
        ("b.splat", &good("b")),
        ("c.splat", &good("c")),
    ]);
    let config = DirectorConfig {
        blend_s: 0.0,
        ..manual_cfg()
    };
    let mut rig = make_rig(config, lib, Prefs::default());
    // The default preset is never blocklisted.
    rig.update(0.0);
    assert_eq!(rig.director.on_device_lost(0.5), None);

    // 5 s after `a` started: not blamed.
    rig.director.command(Command::Load("a.splat".into()));
    rig.wait_showing(100.0, "a.splat");
    assert_eq!(rig.director.on_device_lost(105.0), None);
    assert_eq!(rig.director.current(), showing("a.splat"));

    // 1 s after `b` started: blamed, and taken off screen.
    rig.director.command(Command::Load("b.splat".into()));
    rig.wait_showing(200.0, "b.splat");
    assert_eq!(
        rig.director.on_device_lost(201.0),
        Some("b.splat".to_string())
    );
    assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));

    // History is [a, b, c]: `Prev` from `c` walks past the blocklisted `b`.
    rig.director.command(Command::Load("c.splat".into()));
    rig.wait_showing(300.0, "c.splat");
    rig.director.command(Command::Prev);
    rig.wait(310.0, "Prev to land", |d| shown(d) != "c.splat");
    assert_eq!(rig.director.current(), showing("a.splat"));

    // `b` is skipped from now on, even when asked for.
    rig.director.command(Command::Load("b.splat".into()));
    for _ in 0..100 {
        rig.update(320.0);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(rig.director.current(), showing("a.splat"));
    // Nothing allowed before `a`: `Prev` does nothing.
    rig.director.command(Command::Prev);
    for _ in 0..50 {
        rig.update(330.0);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(rig.director.current(), showing("a.splat"));
}

#[test]
fn device_lost_mid_blend_handles_queued() {
    let (_fx, lib) = fixture(&[("a.splat", &good("a")), ("b.splat", &good("b"))]);
    let mut rig = make_rig(manual_cfg(), lib, Prefs::default());
    rig.director.command(Command::Load("a.splat".into()));
    rig.wait_showing(10.0, "a.splat");
    rig.director.command(Command::Load("b.splat".into()));
    rig.wait(10.5, "b to be queued", |d| d.queued() == Some("b.splat"));
    // The incoming `a` is blamed: the blend is abandoned, and the queued `b`
    // (allowed, and loaded) blends in from the outgoing preset at once.
    assert_eq!(
        rig.director.on_device_lost(11.0),
        Some("a.splat".to_string())
    );
    assert_eq!(rig.director.queued(), None);
    match rig.director.current() {
        State::Blending {
            from, to, start, ..
        } => assert_eq!(
            (from.as_str(), to.as_str(), start),
            (DEFAULT_PRESET_ID, "b.splat", 11.0)
        ),
        s => panic!("{s:?}"),
    }
    // `b` again, queued behind its own blend; then `b` is blamed, so the
    // queued switch (now blocklisted) is dropped, not applied later.
    rig.director.command(Command::Load("b.splat".into()));
    rig.wait(11.2, "b to be queued again", |d| {
        d.queued() == Some("b.splat")
    });
    assert_eq!(
        rig.director.on_device_lost(11.5),
        Some("b.splat".to_string())
    );
    assert_eq!(rig.director.queued(), None);
    assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
    for i in 0..600 {
        rig.update(12.0 + i as f64 * DT);
    }
    assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
}

#[test]
fn loader_swapped_mid_request() {
    let (_fx, lib) = fixture(&[("a.splat", &good("a")), ("b.splat", &good("b"))]);
    let mut rig = make_rig(manual_cfg(), lib, Prefs::default());
    rig.loader.set_paused(true);
    rig.director.command(Command::Load("a.splat".into()));
    rig.update(0.0);
    assert_eq!(rig.loader.request_count(), 1);
    // The app replaces the loader (e.g. a new surface format). The old
    // one's queued job is dropped with it; the request goes to the new one.
    let old_id = rig.loader.loader_id();
    rig.loader = Loader::spawn(rig.renderer.shared(), TextureResolver::new(Vec::new()), FMT);
    assert_ne!(rig.loader.loader_id(), old_id);
    rig.wait_showing(1.0, "a.splat");
    // Prefetching carries on with the new loader.
    rig.wait_ready(2.0);
}

#[test]
fn unanswered_request_times_out() {
    let (_fx, lib) = fixture(&[("a.splat", &good("a")), ("b.splat", &good("b"))]);
    let mut rig = make_rig(manual_cfg(), lib, Prefs::default());
    rig.loader.set_paused(true);
    rig.director.command(Command::Load("a.splat".into()));
    rig.update(0.0);
    assert_eq!(rig.loader.request_count(), 1);
    for i in 1..(30 * 60) {
        rig.update(i as f64 * DT);
    }
    assert_eq!(rig.loader.request_count(), 1, "re-requested before 30 s");
    rig.update(30.0);
    assert_eq!(rig.loader.request_count(), 2, "not re-requested after 30 s");
    assert_eq!(rig.director.consecutive_failures(), 0);
    // Once the loader answers, the first (stale) answer is ignored and
    // the second one shows.
    rig.loader.set_paused(false);
    rig.wait_showing(31.0, "a.splat");
}

#[test]
fn device_lost_failure_is_retried_on_new_loader() {
    let (_fx, lib) = fixture(&[("a.splat", &good("a"))]);
    let mut rig = make_rig(manual_cfg(), lib.clone(), Prefs::default());
    // The device is lost: the warm-up's poll fails.
    rig.gpu.lost.store(true, Ordering::SeqCst);
    rig.loader
        .request("a.splat".into(), path_of(&lib, "a.splat"), 1);
    match wait_result(&rig.loader) {
        LoadResult::Failed { device_lost, .. } => assert!(device_lost),
        LoadResult::Ready { .. } => panic!("loaded on a lost device"),
    }
    rig.director.command(Command::Load("a.splat".into()));
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(500) {
        rig.update(1.0);
        std::thread::sleep(Duration::from_millis(1));
    }
    // Not skipped, not counted, and not sent again to the lost loader.
    assert_eq!(rig.loader.request_count(), 2);
    assert_eq!(rig.director.consecutive_failures(), 0);
    assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
    // The app recreates the device and the loader: the request goes out
    // again and the preset shows.
    rig.gpu.lost.store(false, Ordering::SeqCst);
    rig.renderer.recreate(&rig.gpu);
    rig.loader = Loader::spawn(rig.renderer.shared(), TextureResolver::new(Vec::new()), FMT);
    rig.director.on_renderer_recreated(
        &rig.renderer.shared(),
        &TextureResolver::new(Vec::new()),
        FMT,
    );
    rig.wait_showing(2.0, "a.splat");
}

#[test]
fn manual_request_suppresses_auto_advance_and_cuts() {
    let (_fx, lib) = fixture(&[
        ("a.splat", &good("a")),
        ("b.splat", &good("b")),
        ("c.splat", &good("c")),
    ]);
    let mut rig = make_rig(cfg(), lib, Prefs::default());
    rig.wait_ready(0.0);
    let next = rig.director.next_ready().unwrap().to_string();
    let wanted = ["a.splat", "b.splat", "c.splat"]
        .into_iter()
        .find(|id| *id != next)
        .unwrap();
    rig.loader.set_paused(true);
    rig.director.command(Command::Load(wanted.into()));
    rig.update(0.5);
    // Past the auto-advance deadline (at most 25 s), with strong beats after
    // the hard-cut gap: the ready next preset is not used while the manual
    // request waits.
    for i in 0..(28 * 60) {
        rig.update_beat(1.0 + i as f64 * DT, 9.0);
        assert_eq!(rig.director.current(), showing(DEFAULT_PRESET_ID));
    }
    rig.loader.set_paused(false);
    rig.wait_showing(29.0, wanted);
}

#[test]
fn non_finite_now_is_ignored() {
    let (_fx, lib) = fixture(&[("a.splat", &good("a")), ("b.splat", &good("b"))]);
    let config = DirectorConfig {
        auto_advance_s: 1.0e6,
        ..DirectorConfig::default()
    };
    let mut rig = make_rig(config, lib, Prefs::default());
    // A NaN first update must not become the default preset's start time.
    rig.update(f64::NAN);
    rig.update(f64::INFINITY);
    rig.wait_ready(0.0);
    rig.update_beat(10.5, 9.0);
    let first = shown(&rig.director);
    assert_ne!(first, DEFAULT_PRESET_ID, "the hard cut never came");
    // NaN neither blames the preset nor moves its start time.
    assert_eq!(rig.director.on_device_lost(f64::NAN), None);
    rig.update(f64::NAN);
    assert_eq!(rig.director.on_device_lost(11.0), Some(first));
}

#[test]
fn renders_solo_blend_and_recreated_renderer() {
    let (_fx, lib) = fixture(&[("a.splat", &good("a")), ("f.splat", FP16)]);
    let mut rig = make_rig(manual_cfg(), lib, Prefs::default());
    let target = rig.gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("display"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let mut g = Globals::default();
    let audio = AudioFrame::default();
    let base = TickInputBase {
        fps: 60.0,
        audio: &audio,
        aspect: [1.0, 1.0],
        pixels: [64, 64],
    };
    let mut now = 0.0;
    let mut step = |rig: &mut Rig, now: &mut f64, n: usize| {
        for _ in 0..n {
            *now += DT;
            rig.update(*now);
            rig.director
                .tick_and_render(*now, &base, &mut g, &mut rig.renderer, &rig.gpu);
            rig.director
                .composite(*now, &mut rig.renderer, &rig.gpu, &view, FMT);
        }
    };
    // The default preset: its main wave lights pixels up.
    step(&mut rig, &mut now, 30);
    let lit = |img: &image::RgbaImage| img.pixels().filter(|p| p.0[..3] != [0, 0, 0]).count();
    assert!(lit(&read_texture(&rig.gpu, &target)) > 0);
    assert_eq!(rig.renderer.format(), FeedbackFormat::Rgba8);

    // A blend into an fp16 preset: the feedback format follows it.
    rig.director.command(Command::Load("f.splat".into()));
    rig.wait(now, "the blend", |d| {
        matches!(d.current(), State::Blending { .. })
    });
    let plan = rig.director.frame_plan(now);
    assert_eq!(plan.feedback_format, FeedbackFormat::Fp16);
    assert!(plan.blend_progress.is_some());
    step(&mut rig, &mut now, 60);
    assert_eq!(rig.renderer.format(), FeedbackFormat::Fp16);
    assert!(lit(&read_texture(&rig.gpu, &target)) > 0);
    step(&mut rig, &mut now, 200);
    assert_eq!(rig.director.current(), showing("f.splat"));
    assert_eq!(rig.director.frame_plan(now).blend_progress, None);

    // Device loss: the app recreates the renderer and the loader, and the
    // director recompiles its shaders; the show goes on.
    assert_eq!(rig.director.on_device_lost(now), None);
    rig.renderer.recreate(&rig.gpu);
    rig.loader = Loader::spawn(rig.renderer.shared(), TextureResolver::new(Vec::new()), FMT);
    rig.director.on_renderer_recreated(
        &rig.renderer.shared(),
        &TextureResolver::new(Vec::new()),
        FMT,
    );
    step(&mut rig, &mut now, 30);
    assert_eq!(rig.director.current(), showing("f.splat"));
    assert!(lit(&read_texture(&rig.gpu, &target)) > 0);
    // A request made after the recreate goes to the new loader.
    rig.director.command(Command::Load("a.splat".into()));
    rig.wait_showing(now, "a.splat");
    assert!(rig.gpu.poll_wait().is_ok());
}
