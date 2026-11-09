//! End-to-end tests of the bundled presets (`presets/*.splat`): each one is
//! parsed, compiled (code and shaders) and run through the whole pipeline
//! (tick, warp, overlays, blur, composite) for 120 ticks of deterministic
//! synthetic music, and the composite is compared with its golden image
//! (`tests/golden/preset_<name>.png`). A reaction check makes sure every
//! preset visibly responds to the music.
//!
//! The GPU tests run on the machine's real adapter, or on WARP with
//! `MS_FORCE_WARP=1`. `MS_BLESS=1` rewrites the goldens.
//!
//! `cargo test -p ms-engine --test presets_e2e -- --ignored --nocapture`
//! runs the tuning harness: 600 ticks of every preset (or only those named
//! in `MS_TUNE_ONLY`, comma-separated), printing the mean luminance over
//! time and, if `MS_TUNE_OUT` names a folder, writing the composites of
//! ticks 60, 300 and 600 there as PNGs (see `tune_presets` for the tick
//! count, size, shot and timing options). With `MS_TUNE_AUDIO` naming a raw
//! recording (interleaved little-endian f32 stereo at 48 kHz, at least 10 s,
//! looped if shorter than the run; never committed), it uses that instead
//! of the synthetic music.

use ms_render::test_support::{test_gpu, TestGpu};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use image::RgbaImage;
use ms_analysis::{Analyzer, AudioFrame, AudioHistory, Frame};
use ms_engine::{feedback_format, Globals, PresetInstance, TickInput, PICKS_SALT};
use ms_preset::{compile_preset, parse, CompiledPreset};
use ms_render::{
    assert_golden, golden, read_texture, CompositeLayer, Gpu, PresetPicks, RenderConfig, Renderer,
    ShaderSet, TextureResolver, WarpLayer,
};

/// Internal (feedback) and composite size.
const W: u32 = 256;
const H: u32 = 144;
/// Ticks rendered before the golden is taken.
const TICKS: usize = 120;
/// Simulation rate (ticks per second).
const TICK_HZ: f64 = 60.0;
/// Seed of every preset instance (and, salted, of its random textures).
const SEED: u64 = 25;
/// Audio sample rate of the synthetic music.
const RATE: u32 = 48_000;
/// Seed of the hi-hat noise.
const HAT_SEED: u64 = 0x4A75_6E6B_4861_7473;
const RGBA8: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The bundled presets, by file stem. `presets/` must hold exactly these.
const BUNDLED: [&str; 19] = [
    "default",
    "ink-bloom",
    "bass-tunnel",
    "prism-drift",
    "pulse-grid",
    "aurora-sweep",
    "kaleido-spin",
    "ember-rain",
    "liquid-chrome",
    "starfield-warp",
    "neon-orbit",
    "infinite-coil",
    "turing-reef",
    "lumen-cathedral",
    "ebru-tide",
    "tesseract-drift",
    "interstice",
    "event-horizon",
    "singing-sand",
];

// ------------------------------------------------------------ synthetic audio

/// splitmix64: a well-mixed hash of `x`.
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// White noise in [-1, 1) for sample `n` of stream `seed`.
fn noise(seed: u64, n: u64) -> f64 {
    (mix64(seed ^ mix64(n)) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// The synthetic music at time `t` (seconds), as one stereo frame:
/// - a 55 Hz kick every 0.5 s (a decaying sine burst);
/// - a 440 Hz pad that swells over 4 s (slightly out of phase on the right,
///   so X-Y scopes draw a figure, not a line);
/// - noise hi-hats on the off-beats (seeded, so every run is identical).
pub fn synth_audio(t: f64) -> Frame {
    synth_audio_hats(t, 0.5)
}

/// [`synth_audio`] with a hi-hat every `hat_period` seconds, half a period
/// after the kick grid (0.5 s: off-beats; 0.25 s: double-time, 4 Hz).
fn synth_audio_hats(t: f64, hat_period: f64) -> Frame {
    use std::f64::consts::TAU;
    let tk = t.rem_euclid(0.5);
    let kick = 0.7 * (TAU * 55.0 * tk).sin() * (-tk * 7.0).exp();
    let swell = 0.65 + 0.35 * (TAU * t / 4.0).sin();
    let pad_l = 0.12 * swell * (TAU * 440.0 * t).sin();
    let pad_r = 0.12 * swell * (TAU * 440.0 * t + 0.9).sin();
    let th = (t - hat_period / 2.0).rem_euclid(hat_period);
    let n = (t * f64::from(RATE)).round() as u64;
    let env = 0.22 * (-th * 40.0).exp();
    let hat_l = env * noise(HAT_SEED, n);
    let hat_r = env * noise(HAT_SEED ^ 1, n);
    [(kick + pad_l + hat_l) as f32, (kick + pad_r + hat_r) as f32]
}

/// The analysed audio of `ticks` ticks of `sig` (frame n of the signal is
/// `sig(n)`, at 48 kHz), fed in 10 ms packets through `AudioHistory` and
/// `Analyzer` as the app does.
fn analyse(ticks: usize, mut sig: impl FnMut(u64) -> Frame) -> Vec<AudioFrame> {
    let mut hist = AudioHistory::new(RATE, 4.0);
    let mut an = Analyzer::new(RATE);
    let per_packet = (RATE / 100) as usize;
    let mut out = Vec::with_capacity(ticks);
    let mut buf: Vec<Frame> = Vec::with_capacity(per_packet);
    let mut n: u64 = 0;
    let mut next_tick = 1.0 / TICK_HZ;
    while out.len() < ticks {
        buf.clear();
        for _ in 0..per_packet {
            buf.push(sig(n));
            n += 1;
        }
        let t = (n - 1) as f64 / f64::from(RATE);
        hist.push(&buf, t);
        while next_tick <= t && out.len() < ticks {
            out.push(*an.tick(&hist, t, 0.0, 1.0 / TICK_HZ));
            next_tick += 1.0 / TICK_HZ;
        }
    }
    out
}

/// 600 ticks (10 s) of analysed music, computed once.
fn music() -> &'static [AudioFrame] {
    static MUSIC: OnceLock<Vec<AudioFrame>> = OnceLock::new();
    MUSIC.get_or_init(|| analyse(600, |n| synth_audio(n as f64 / f64::from(RATE))))
}

/// 600 ticks of analysed silence, computed once.
fn quiet() -> &'static [AudioFrame] {
    static QUIET: OnceLock<Vec<AudioFrame>> = OnceLock::new();
    QUIET.get_or_init(|| analyse(600, |_| [0.0, 0.0]))
}

// ------------------------------------------------------------ presets

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn preset_path(stem: &str) -> PathBuf {
    repo().join("presets").join(format!("{stem}.splat"))
}

/// Parses and compiles a bundled preset; any parse warning or code error
/// fails the test.
fn load(stem: &str) -> Arc<CompiledPreset> {
    let path = preset_path(stem);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (p, warnings) = match parse(&bytes) {
        Ok(v) => v,
        Err(e) => panic!("{stem}: parse error: {e:?}"),
    };
    assert!(warnings.is_empty(), "{stem}: parse warnings: {warnings:?}");
    let p = compile_preset(p);
    assert!(p.errors.is_empty(), "{stem}: code errors: {:?}", p.errors);
    Arc::new(p)
}

/// A headless GPU holding the GPU test lock (see `ms_render::test_support`).
fn gpu() -> TestGpu {
    test_gpu()
}

/// A renderer in the preset's feedback format, `size` pixels.
fn renderer(gpu: &Gpu, p: &CompiledPreset, size: [u32; 2]) -> Renderer {
    let mut r = Renderer::new(
        gpu,
        RenderConfig {
            internal_size: size,
            format: ms_render::FeedbackFormat::Rgba8,
        },
    );
    let format = feedback_format(p.source.params.feedback_format);
    if r.format() != format {
        r.set_format(gpu, format);
    }
    r
}

/// Compiles the preset's shaders, with user textures from the bundled
/// `textures/` folder (random picks seeded as the loader does). Any error,
/// including a fallback to a built-in shader, fails the test.
fn shaders(stem: &str, r: &Renderer, gpu: &Gpu, p: &CompiledPreset) -> ShaderSet {
    let resolver = TextureResolver::new(texture_roots());
    let picks = RefCell::new(PresetPicks::new(SEED ^ PICKS_SALT));
    let tex = |name: &str| resolver.resolve_with(name, &mut picks.borrow_mut());
    let s = &p.source.shader;
    let set = r.compile_shaders(gpu, s.warp.as_deref(), s.comp.as_deref(), RGBA8, &tex);
    assert!(
        set.errors.is_empty(),
        "{stem}: shader errors: {:?}",
        set.errors
    );
    set
}

/// The random textures the goldens were made with. `randNN` picks among
/// exactly these (copied into a test-owned folder), so adding a file to
/// `textures/random/` cannot change an unrelated golden.
const RANDOM_TEXTURES: [&str; 3] = ["cells.png", "marble.png", "weave.png"];

/// Texture roots for the tests: first a folder whose `random/` holds only
/// [`RANDOM_TEXTURES`] (so random picks come from there), then the bundled
/// `textures/` (for named textures such as `clouds`).
fn texture_roots() -> Vec<PathBuf> {
    static PINNED: OnceLock<PathBuf> = OnceLock::new();
    let pinned = PINNED.get_or_init(|| {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("presets_e2e_textures");
        let random = root.join("random");
        let _ = std::fs::remove_dir_all(&random);
        std::fs::create_dir_all(&random).expect("create pinned random-texture folder");
        for name in RANDOM_TEXTURES {
            let from = repo().join("textures").join("random").join(name);
            std::fs::copy(&from, random.join(name)).unwrap_or_else(|e| {
                panic!("pinned random texture {} is missing: {e}", from.display())
            });
        }
        root
    });
    vec![pinned.clone(), repo().join("textures")]
}

/// The aspect multipliers the app uses for a `w x h` image.
fn aspect([w, h]: [u32; 2]) -> [f64; 2] {
    let (w, h) = (f64::from(w), f64::from(h));
    if w >= h {
        [1.0, h / w]
    } else {
        [w / h, 1.0]
    }
}

fn composite_target(gpu: &Gpu, [width, height]: [u32; 2]) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("preset e2e target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RGBA8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Runs `stem` from scratch for `audio.len()` ticks (tick k at time
/// k / 60 s, with `audio[k - 1]`) through the full pipeline, calling
/// `capture(k, image)` with the composite after every tick k for which
/// `want(k)` is true.
fn render(
    gpu: &Gpu,
    stem: &str,
    audio: &[AudioFrame],
    want: impl Fn(usize) -> bool,
    capture: impl FnMut(usize, RgbaImage),
) {
    render_at(gpu, stem, audio, At::size([W, H]), want, capture, None);
}

/// Where and when [`render_at`] renders.
#[derive(Clone, Copy)]
struct At {
    /// Internal and composite size in pixels.
    size: [u32; 2],
    /// Preset time (seconds) of tick 0; tick k is at `start + k / 60`.
    start: f64,
}

impl At {
    fn size(size: [u32; 2]) -> Self {
        At { size, start: 0.0 }
    }
}

/// [`render`] at `at`. With `times`, also composites every tick, waits for
/// the GPU and pushes each tick's wall-clock cost (engine tick, warp,
/// overlays, blur and composite) in milliseconds.
fn render_at(
    gpu: &Gpu,
    stem: &str,
    audio: &[AudioFrame],
    at: At,
    want: impl Fn(usize) -> bool,
    mut capture: impl FnMut(usize, RgbaImage),
    mut times: Option<&mut Vec<f64>>,
) {
    let size = at.size;
    let p = load(stem);
    let mut r = renderer(gpu, &p, size);
    let set = shaders(stem, &r, gpu, &p);
    let mut inst = PresetInstance::new(p, SEED, [192, 144]);
    let mut globals = Globals::default();
    let target = composite_target(gpu, size);
    let view = target.create_view(&Default::default());
    let mut ones = Vec::new();
    for (i, a) in audio.iter().enumerate() {
        let k = i + 1;
        let start = std::time::Instant::now();
        let input = TickInput {
            time: at.start + k as f64 / TICK_HZ,
            fps: TICK_HZ,
            frame: k as u64,
            progress: 0.0,
            audio: a,
            aspect: aspect(size),
            pixels: size,
        };
        let out = inst.tick(&input, &mut globals);
        r.set_spectrum(gpu, &a.spectrum);
        ones.resize(out.mesh.uv.len(), 1.0);
        r.warp(
            gpu,
            &[WarpLayer {
                mesh: &out.mesh,
                weight: &ones,
                shader: &set.warp,
                uniforms: &out.uniforms,
            }],
        );
        r.draw(gpu, &out.draw, 1.0);
        r.blur(gpu, set.blur_levels);
        if let Some(times) = times.as_deref_mut() {
            r.composite(
                gpu,
                &view,
                RGBA8,
                &[CompositeLayer {
                    shader: &set.comp,
                    uniforms: &out.uniforms,
                    alpha: 1.0,
                }],
            );
            gpu.poll_wait().expect("GPU lost while timing");
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        if want(k) {
            r.composite(
                gpu,
                &view,
                RGBA8,
                &[CompositeLayer {
                    shader: &set.comp,
                    uniforms: &out.uniforms,
                    alpha: 1.0,
                }],
            );
            capture(k, read_texture(gpu, &target));
        }
    }
}

/// The composite after `TICKS` ticks of `audio`.
fn image_after(gpu: &Gpu, stem: &str, audio: &[AudioFrame]) -> RgbaImage {
    let mut img = None;
    render(
        gpu,
        stem,
        &audio[..TICKS],
        |k| k == TICKS,
        |_, i| img = Some(i),
    );
    img.expect("tick 120 captured")
}

/// Fraction of pixels differing by more than the golden channel tolerance.
fn diff_fraction(a: &RgbaImage, b: &RgbaImage) -> f64 {
    assert_eq!(a.dimensions(), b.dimensions());
    let differing = a
        .pixels()
        .zip(b.pixels())
        .filter(|(p, q)| {
            p.0.iter()
                .zip(q.0.iter())
                .any(|(x, y)| x.abs_diff(*y) > golden::MAX_CHANNEL_DELTA)
        })
        .count();
    differing as f64 / f64::from(a.width() * a.height())
}

/// Mean Rec. 709 luminance in [0, 1].
fn luminance(img: &RgbaImage) -> f64 {
    let sum: f64 = img
        .pixels()
        .map(|p| {
            0.2126 * f64::from(p.0[0]) + 0.7152 * f64::from(p.0[1]) + 0.0722 * f64::from(p.0[2])
        })
        .sum();
    sum / 255.0 / f64::from(img.width() * img.height())
}

/// The golden check and the reaction check of one bundled preset.
fn check_preset(stem: &str) {
    let gpu = gpu();
    let with_music = image_after(&gpu, stem, music());
    let lum = luminance(&with_music);
    assert!(
        (0.04..=0.9).contains(&lum),
        "{stem}: tick-{TICKS} mean luminance {lum:.3} is outside [0.04, 0.9]"
    );
    assert_golden!(with_music, &format!("preset_{}", stem.replace('-', "_")));
    let with_silence = image_after(&gpu, stem, quiet());
    let d = diff_fraction(&with_music, &with_silence);
    println!(
        "{stem}: tick-{TICKS} luminance {lum:.3}; {:.1}% of pixels differ from silence",
        d * 100.0
    );
    assert!(
        d > 0.05,
        "{stem}: tick-{TICKS} image with music differs from silence on only {:.2}% of pixels",
        d * 100.0
    );
}

macro_rules! preset_tests {
    ($($test:ident => $stem:literal,)*) => {
        $(
            #[test]
            fn $test() {
                check_preset($stem);
            }
        )*
    };
}

preset_tests! {
    preset_default => "default",
    preset_ink_bloom => "ink-bloom",
    preset_bass_tunnel => "bass-tunnel",
    preset_prism_drift => "prism-drift",
    preset_pulse_grid => "pulse-grid",
    preset_aurora_sweep => "aurora-sweep",
    preset_kaleido_spin => "kaleido-spin",
    preset_ember_rain => "ember-rain",
    preset_liquid_chrome => "liquid-chrome",
    preset_starfield_warp => "starfield-warp",
    preset_neon_orbit => "neon-orbit",
    preset_infinite_coil => "infinite-coil",
    preset_turing_reef => "turing-reef",
    preset_lumen_cathedral => "lumen-cathedral",
    preset_ebru_tide => "ebru-tide",
    preset_tesseract_drift => "tesseract-drift",
    preset_interstice => "interstice",
    preset_event_horizon => "event-horizon",
    preset_singing_sand => "singing-sand",
}

/// `presets/` holds exactly the bundled set (so every file has a test
/// above), and every bundled preset has its metadata.
#[test]
fn bundled_set_is_complete() {
    let mut found: Vec<String> = std::fs::read_dir(repo().join("presets"))
        .expect("presets/ exists")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "splat"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    found.sort();
    let mut want: Vec<String> = BUNDLED.iter().map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(
        found, want,
        "presets/ must hold exactly the bundled presets"
    );
    for stem in BUNDLED {
        let p = load(stem);
        let m = &p.source.meta;
        assert!(!m.name.trim().is_empty(), "{stem}: meta.name is empty");
        assert_eq!(m.author, "Milksplat", "{stem}: meta.author");
        assert_eq!(m.license, "MIT OR Apache-2.0", "{stem}: meta.license");
    }
}

/// The presets checked by [`no_late_flashing`], with how many ticks each
/// runs: a minute for those with long-running state (accumulated phases,
/// state machines), twenty seconds for the rest (enough for their feedback
/// to settle; every bundled preset but the gentle default is covered).
const LONG_RUN: [(&str, usize); 18] = [
    ("ink-bloom", 1200),
    ("bass-tunnel", 1200),
    ("prism-drift", 1200),
    ("pulse-grid", 1200),
    ("aurora-sweep", 1200),
    ("kaleido-spin", 1200),
    ("ember-rain", 1200),
    ("liquid-chrome", 1200),
    ("starfield-warp", 1200),
    ("neon-orbit", 1200),
    ("infinite-coil", 3600),
    ("turing-reef", 3600),
    ("lumen-cathedral", 3600),
    ("ebru-tide", 3600),
    ("tesseract-drift", 3600),
    ("interstice", 3600),
    ("event-horizon", 3600),
    ("singing-sand", 3600),
];

/// A cheap guard against flashing that only develops over time (for
/// example a phase computed as time times an audio-driven speed, which
/// jumps further every tick as time grows): each preset in [`LONG_RUN`] is
/// ticked for its tick count of the synthetic music at 64x36, starting with
/// the preset clock at 15 minutes. Over the last two seconds no tick may
/// change the mean luminance by more than 0.06, and it may not flicker
/// (step one way and straight back by more than 0.005).
#[test]
fn no_late_flashing() {
    const LAST: usize = 120;
    let gpu = gpu();
    let longest = LONG_RUN.iter().map(|(_, t)| *t).max().unwrap_or(0);
    let all = analyse(longest, |n| synth_audio(n as f64 / f64::from(RATE)));
    for (stem, ticks) in LONG_RUN {
        let mut lum = Vec::with_capacity(LAST + 1);
        render_at(
            &gpu,
            stem,
            &all[..ticks],
            At {
                size: [64, 36],
                start: 900.0,
            },
            |k| k >= ticks - LAST,
            |_, img| lum.push(luminance(&img)),
            None,
        );
        let jump = lum
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f64::max);
        // A flicker: the luminance steps one way and straight back, both
        // steps large. Smooth motion only reverses where its steps are small.
        let flicker = lum
            .windows(3)
            .map(|w| (w[1] - w[0], w[2] - w[1]))
            .filter(|(a, b)| a * b < 0.0)
            .map(|(a, b)| a.abs().min(b.abs()))
            .fold(0.0, f64::max);
        println!(
            "{stem}: near tick {ticks}, largest step {jump:.4}, largest reversal {flicker:.4}"
        );
        assert!(
            jump <= 0.06,
            "{stem}: mean luminance jumps by {jump:.4} between ticks near tick {ticks}"
        );
        // Healthy presets stay under about 0.004 here; the old Turing Reef
        // pulse bug (phase = time x bass-driven speed) flickered at 0.0074.
        assert!(
            flicker <= 0.005,
            "{stem}: mean luminance flickers (steps {flicker:.4} and back) near tick {ticks}"
        );
    }
}

/// The bundled set covers the engine's features (task 25's coverage list).
#[test]
fn bundled_set_coverage() {
    let gpu = gpu();
    let mut counts = [0usize; 8];
    let names = [
        "custom warp shader",
        "comp shader",
        "blur",
        "shape with instances",
        "custom wave",
        "fp16 feedback",
        "user texture",
        "megabuf",
    ];
    for stem in BUNDLED {
        let p = load(stem);
        let r = renderer(&gpu, &p, [W, H]);
        let set = shaders(stem, &r, &gpu, &p);
        let src = &p.source;
        let code_uses_megabuf = [&src.code.init, &src.code.frame, &src.code.vertex]
            .into_iter()
            .chain(src.waves.iter().flat_map(|w| [&w.init, &w.frame, &w.point]))
            .chain(src.shapes.iter().flat_map(|s| [&s.init, &s.frame]))
            .any(|c| c.to_lowercase().contains("megabuf("));
        let has = [
            src.shader.warp.is_some(),
            src.shader.comp.is_some(),
            set.blur_levels > 0,
            src.shapes.iter().any(|s| s.enabled && s.instances > 1),
            src.waves.iter().any(|w| w.enabled),
            p.source.params.feedback_format == ms_preset::FeedbackFormat::Fp16,
            !set.user_textures.is_empty(),
            code_uses_megabuf,
        ];
        for (c, h) in counts.iter_mut().zip(has) {
            *c += usize::from(h);
        }
    }
    let need = [3, 3, 2, 2, 2, 1, 1, 1];
    for ((name, have), need) in names.iter().zip(counts).zip(need) {
        assert!(have >= need, "{name}: {have} presets, need at least {need}");
    }
}

/// Tuning harness (see the module docs): `MS_TUNE_TICKS` (default 600)
/// ticks per preset at `MS_TUNE_SIZE` (`WxH`, default the golden size),
/// mean luminance every tick, PNGs of the ticks in `MS_TUNE_SHOTS`
/// (comma-separated, default 60,300,600) to `MS_TUNE_OUT`. Prints the
/// range, the mean, the range and mean after the first five seconds (the
/// settled look) and the largest tick-to-tick luminance change (a
/// full-frame flash detector), for the music and, unless `MS_TUNE_AUDIO` is
/// set, again for the music with double-time (4 Hz) hats. With
/// `MS_TUNE_BENCH=1` it also times 600 ticks of each preset at 1920x1080
/// (engine tick plus every GPU pass, waiting for the GPU each tick). Fails
/// if a preset's mean luminance leaves [0.05, 0.85] after the first second.
#[test]
#[ignore = "tuning harness: slow, prints numbers and writes PNGs"]
fn tune_presets() {
    let gpu = gpu();
    let env = |k: &str| std::env::var(k).ok();
    let only = env("MS_TUNE_ONLY");
    let out = env("MS_TUNE_OUT").map(PathBuf::from);
    let ticks: usize = env("MS_TUNE_TICKS").map_or(600, |t| t.parse().expect("MS_TUNE_TICKS"));
    let shots: Vec<usize> = env("MS_TUNE_SHOTS").map_or(vec![60, 300, 600], |s| {
        s.split(',')
            .map(|k| k.trim().parse().expect("MS_TUNE_SHOTS"))
            .collect()
    });
    let size = env("MS_TUNE_SIZE").map_or([W, H], |s| {
        let (w, h) = s.split_once('x').expect("MS_TUNE_SIZE is WxH");
        [w.parse().expect("width"), h.parse().expect("height")]
    });
    let bench = env("MS_TUNE_BENCH").is_some_and(|b| b == "1");
    let recorded = env("MS_TUNE_AUDIO").map(|path| {
        let bytes = std::fs::read(&path).expect("read MS_TUNE_AUDIO");
        let samples: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        // Loop the recording if the run is longer than it.
        let frames = (samples.len() / 2).max(1);
        analyse(ticks, |n| {
            let i = n as usize % frames * 2;
            [
                samples.get(i).copied().unwrap_or(0.0),
                samples.get(i + 1).copied().unwrap_or(0.0),
            ]
        })
    });
    let runs: Vec<(&str, Vec<AudioFrame>)> = match recorded {
        Some(r) => vec![("recorded", r)],
        None => vec![
            (
                "synthetic",
                analyse(ticks, |n| synth_audio(n as f64 / f64::from(RATE))),
            ),
            (
                "4 Hz hats",
                analyse(ticks, |n| {
                    synth_audio_hats(n as f64 / f64::from(RATE), 0.25)
                }),
            ),
        ],
    };
    let mut bad = Vec::new();
    for (label, audio) in &runs {
        println!("--- {label}");
        for stem in BUNDLED {
            if let Some(only) = &only {
                if !only.split(',').any(|s| s.trim() == stem) {
                    continue;
                }
            }
            let mut lum = Vec::new();
            render_at(
                &gpu,
                stem,
                audio,
                At::size(size),
                |_| true,
                |k, img| {
                    lum.push((k, luminance(&img)));
                    if let (Some(dir), "synthetic" | "recorded") = (&out, *label) {
                        if shots.contains(&k) {
                            let path = dir.join(format!("{stem}-{k}.png"));
                            img.save(&path).expect("save tuning PNG");
                        }
                    }
                },
                None,
            );
            let settled = lum.iter().filter(|(k, _)| *k >= 60).map(|(_, l)| *l);
            let (lo, hi) = settled.fold((f64::MAX, f64::MIN), |(lo, hi), l| (lo.min(l), hi.max(l)));
            let mean = lum.iter().map(|(_, l)| l).sum::<f64>() / lum.len() as f64;
            // After the first five seconds (the settled look).
            let late: Vec<f64> = lum
                .iter()
                .filter(|(k, _)| *k > 300)
                .map(|(_, l)| *l)
                .collect();
            let (late_lo, late_hi, late_mean) = if late.is_empty() {
                (mean, mean, mean)
            } else {
                (
                    late.iter().copied().fold(f64::MAX, f64::min),
                    late.iter().copied().fold(f64::MIN, f64::max),
                    late.iter().sum::<f64>() / late.len() as f64,
                )
            };
            // Largest change from one tick to the next (a flash detector).
            let jump = lum
                .windows(2)
                .map(|w| (w[1].1 - w[0].1).abs())
                .fold(0.0, f64::max);
            let trace: Vec<String> = lum
                .iter()
                .filter(|(k, _)| k % 60 == 0)
                .map(|(_, l)| format!("{l:.3}"))
                .collect();
            println!(
                "{stem:15} luminance min {lo:.3} max {hi:.3} mean {mean:.3} settled {late_lo:.3}..{late_hi:.3} mean {late_mean:.3} max-tick-delta {jump:.4}  per second: {}",
                trace.join(" ")
            );
            if !(0.05..=0.85).contains(&lo) || !(0.05..=0.85).contains(&hi) {
                bad.push(format!("{label} {stem}: [{lo:.3}, {hi:.3}]"));
            }
            if bench && *label != "4 Hz hats" {
                let mut times = Vec::new();
                let n = audio.len().min(600);
                render_at(
                    &gpu,
                    stem,
                    &audio[..n],
                    At::size([1920, 1080]),
                    |_| false,
                    |_, _| {},
                    Some(&mut times),
                );
                let mut t: Vec<f64> = times.into_iter().skip(30).collect();
                t.sort_by(f64::total_cmp);
                let avg = t.iter().sum::<f64>() / t.len() as f64;
                println!(
                    "{stem:15} 1920x1080 ms/tick: mean {avg:.2} p95 {:.2} max {:.2}",
                    t[t.len() * 95 / 100],
                    t[t.len() - 1]
                );
            }
        }
    }
    assert!(bad.is_empty(), "luminance out of [0.05, 0.85]: {bad:?}");
}
