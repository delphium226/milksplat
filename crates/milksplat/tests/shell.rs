//! The app shell's pure parts: command line, double-click, surface format,
//! internal size, aspect, config defaults, quality cycle and paths.

use std::ffi::OsString;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use milksplat::app::{Args, DoubleClick};
use milksplat::config::{Config, Quality, WindowState};
use milksplat::paths::{user_presets_in, Paths};
use milksplat::render_thread::{aspect, display_name, internal_size, pick_format};
use wgpu::TextureFormat as F;

fn parse(args: &[&str]) -> Result<Option<Args>, String> {
    Args::parse(args.iter().map(OsString::from))
}

#[test]
fn args_parse() {
    assert_eq!(parse(&[]), Ok(Some(Args::default())));
    let a = parse(&["--presets", "p", "--exit-after", "8", "--log-fps"])
        .unwrap()
        .unwrap();
    assert_eq!(a.presets, Some(PathBuf::from("p")));
    assert_eq!(a.exit_after, Some(8.0));
    assert!(a.log_fps);
    assert_eq!(a.debug_lose_device, None);
    let a = parse(&["--debug-lose-device", "2.5"]).unwrap().unwrap();
    assert_eq!(a.debug_lose_device, Some(2.5));
    assert_eq!(parse(&["--help"]), Ok(None));
    assert_eq!(parse(&["-h"]), Ok(None));
    for bad in [
        &["--presets"][..],
        &["--exit-after"],
        &["--exit-after", "x"],
        &["--exit-after", "0"],
        &["--exit-after", "-1"],
        &["--exit-after", "inf"],
        &["--exit-after", "NaN"],
        &["--frobnicate"],
    ] {
        assert!(parse(bad).is_err(), "{bad:?} should be refused");
    }
}

#[test]
fn double_click_within_400_ms() {
    let t0 = Instant::now();
    let ms = |n| t0 + Duration::from_millis(n);
    let mut d = DoubleClick::default();
    assert!(!d.press(ms(0)));
    assert!(d.press(ms(400)), "two clicks 400 ms apart");
    // The pair is used up: the next click starts a new one.
    assert!(!d.press(ms(500)));
    assert!(!d.press(ms(901)), "401 ms is too slow");
    assert!(d.press(ms(1000)));
}

#[test]
fn surface_format_prefers_non_srgb_8_bit() {
    assert_eq!(
        pick_format(&[F::Bgra8UnormSrgb, F::Rgba8Unorm, F::Bgra8Unorm]),
        Some(F::Bgra8Unorm)
    );
    assert_eq!(
        pick_format(&[F::Bgra8UnormSrgb, F::Rgba8Unorm]),
        Some(F::Rgba8Unorm)
    );
    assert_eq!(
        pick_format(&[F::Bgra8UnormSrgb, F::Rgb10a2Unorm]),
        Some(F::Rgb10a2Unorm)
    );
    assert_eq!(pick_format(&[F::Bgra8UnormSrgb]), Some(F::Bgra8UnormSrgb));
    assert_eq!(pick_format(&[]), None);
}

#[test]
fn internal_size_scales_and_rounds() {
    assert_eq!(internal_size([1920, 1080], 1.0), [1920, 1080]);
    assert_eq!(internal_size([1920, 1080], 0.5), [960, 540]);
    assert_eq!(internal_size([1001, 3], 0.5), [501, 2]);
    assert_eq!(internal_size([1, 1], 0.5), [1, 1]);
    assert_eq!(internal_size([800, 600], f32::NAN), [800, 600]);
    assert_eq!(internal_size([800, 600], 0.0), [800, 600]);
}

#[test]
fn aspect_scales_the_shorter_side() {
    assert_eq!(aspect([1920, 1080]), [1.0, 0.5625]);
    assert_eq!(aspect([1080, 1920]), [0.5625, 1.0]);
    assert_eq!(aspect([500, 500]), [1.0, 1.0]);
    assert_eq!(aspect([0, 0]), [1.0, 1.0]);
}

#[test]
fn config_defaults() {
    let c = Config::default();
    assert_eq!(c.auto_advance_s, 20.0);
    assert_eq!(c.blend_s, 2.7);
    assert!(c.hard_cuts);
    assert_eq!(c.hard_cut_strength, 2.0);
    assert_eq!(c.quality, Quality::Auto);
    assert!(c.favourites.is_empty() && c.bans.is_empty() && c.blocklist.is_empty());
    assert!(c.av_offsets_ms.is_empty());
    assert_eq!(c.audio_device, None);
    assert_eq!(c.fullscreen_monitor, None);
    assert_eq!(
        c.window,
        WindowState {
            size: [1280, 720],
            position: None,
            monitor: None
        }
    );
}

#[test]
fn quality_cycles_and_caps_the_mesh() {
    let mut q = Quality::Auto;
    let mut seen = Vec::new();
    for _ in 0..4 {
        seen.push(q);
        q = q.next();
    }
    assert_eq!(
        seen,
        [Quality::Auto, Quality::High, Quality::Medium, Quality::Low]
    );
    assert_eq!(q, Quality::Auto);
    assert_eq!(Quality::Auto.fixed_scale(), None);
    assert_eq!(Quality::Low.fixed_scale(), Some(0.5));
    for q in seen {
        let [c, r] = q.mesh_cap();
        assert!(c <= 192 && r <= 144, "{q:?}");
    }
}

#[test]
fn paths_layout() {
    let p = Paths::new(PathBuf::from("dist").join("presets"), None);
    assert_eq!(p.bundled_textures, PathBuf::from("dist").join("textures"));
    assert_eq!(
        p.preset_roots(),
        vec![PathBuf::from("dist").join("presets")]
    );
    assert_eq!(
        p.texture_roots(),
        vec![PathBuf::from("dist").join("textures")]
    );

    // `--presets presets`: textures beside it, in the current directory.
    let p = Paths::new(PathBuf::from("presets"), Some(PathBuf::from("user")));
    assert_eq!(p.bundled_textures, PathBuf::from("textures"));
    assert_eq!(
        p.preset_roots(),
        vec![PathBuf::from("user"), PathBuf::from("presets")],
        "user presets first"
    );

    assert_eq!(
        user_presets_in(&PathBuf::from("home")),
        PathBuf::from("home")
            .join("Documents")
            .join("Milksplat")
            .join("Presets")
    );
    assert_eq!(display_name("dir/ink-bloom.splat"), "dir/ink-bloom");
    assert_eq!(display_name("<default>"), "<default>");
}
