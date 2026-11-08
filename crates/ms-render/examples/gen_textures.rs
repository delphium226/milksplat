//! Generates the bundled textures (original, procedural, deterministic):
//!
//! - `textures/clouds.png`: 512x512 soft fractal clouds, near-grey with a
//!   faint cool tint, tileable;
//! - `textures/random/weave.png`, `cells.png`, `marble.png`: 256x256
//!   colour patterns for the `rand00`..`rand15` random-texture names, all
//!   tileable.
//!
//! Run from the repository root:
//! `cargo run -p ms-render --example gen_textures [-- <out dir>]`
//! (default out dir: `textures`).

use std::f64::consts::TAU;
use std::path::PathBuf;

use image::{Rgba, RgbaImage};

/// splitmix64 finaliser.
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Hash of a lattice point in [0, 1).
fn lattice(seed: u64, ix: i64, iy: i64) -> f64 {
    let h = mix64(seed ^ mix64(ix as u64 ^ mix64(iy as u64)));
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn smooth(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Value noise with `period` cells per unit, wrapping every `period` cells
/// (so it tiles over [0, 1)).
fn value_noise(seed: u64, x: f64, y: f64, period: i64) -> f64 {
    let (fx, fy) = (x * period as f64, y * period as f64);
    let (ix, iy) = (fx.floor() as i64, fy.floor() as i64);
    let (tx, ty) = (smooth(fx - ix as f64), smooth(fy - iy as f64));
    let at = |dx: i64, dy: i64| {
        lattice(
            seed,
            (ix + dx).rem_euclid(period),
            (iy + dy).rem_euclid(period),
        )
    };
    let top = at(0, 0) + (at(1, 0) - at(0, 0)) * tx;
    let bottom = at(0, 1) + (at(1, 1) - at(0, 1)) * tx;
    top + (bottom - top) * ty
}

/// Tileable fractal noise in about [0, 1].
fn fbm(seed: u64, x: f64, y: f64, base: i64, octaves: u32) -> f64 {
    let (mut sum, mut amp, mut norm, mut period) = (0.0, 1.0, 0.0, base);
    for o in 0..octaves {
        sum += amp * value_noise(seed.wrapping_add(u64::from(o) * 7919), x, y, period);
        norm += amp;
        amp *= 0.55;
        period *= 2;
    }
    sum / norm
}

fn to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn rgb(r: f64, g: f64, b: f64) -> Rgba<u8> {
    Rgba([to_u8(r), to_u8(g), to_u8(b), 255])
}

/// Soft clouds: fbm with a gentle contrast curve.
fn clouds() -> RgbaImage {
    let n = 512;
    RgbaImage::from_fn(n, n, |px, py| {
        let (x, y) = (f64::from(px) / f64::from(n), f64::from(py) / f64::from(n));
        // Domain-warp the clouds a little with a second noise field.
        let wx = fbm(11, x, y, 4, 3) - 0.5;
        let wy = fbm(12, x, y, 4, 3) - 0.5;
        let v = fbm(
            10,
            (x + 0.08 * wx).rem_euclid(1.0),
            (y + 0.08 * wy).rem_euclid(1.0),
            4,
            6,
        );
        let v = smooth(((v - 0.25) / 0.5).clamp(0.0, 1.0));
        let v = 0.08 + 0.86 * v;
        rgb(v * 0.97, v * 0.99, v)
    })
}

/// Woven interference of sine bands in teal, gold and rose.
fn weave() -> RgbaImage {
    let n = 256;
    RgbaImage::from_fn(n, n, |px, py| {
        let (x, y) = (f64::from(px) / f64::from(n), f64::from(py) / f64::from(n));
        let a = (TAU * (3.0 * x + 2.0 * y)).sin();
        let b = (TAU * (-2.0 * x + 5.0 * y)).sin();
        let c = (TAU * (4.0 * x - 1.0 * y) + 2.0 * a).sin();
        let r = 0.5 + 0.35 * b + 0.15 * c;
        let g = 0.45 + 0.3 * a * c;
        let bl = 0.5 + 0.4 * a - 0.1 * b;
        rgb(0.2 + 0.7 * r, 0.15 + 0.7 * g, 0.25 + 0.6 * bl)
    })
}

/// Soft Voronoi cells (wrapping), each with its own hue and a dark seam.
fn cells() -> RgbaImage {
    let n = 256;
    let k = 7i64;
    let point = |cx: i64, cy: i64| {
        let (wx, wy) = (cx.rem_euclid(k), cy.rem_euclid(k));
        (
            (cx as f64 + 0.15 + 0.7 * lattice(31, wx, wy)) / k as f64,
            (cy as f64 + 0.15 + 0.7 * lattice(32, wx, wy)) / k as f64,
            lattice(33, wx, wy),
        )
    };
    RgbaImage::from_fn(n, n, |px, py| {
        let (x, y) = (f64::from(px) / f64::from(n), f64::from(py) / f64::from(n));
        let (cx, cy) = ((x * k as f64).floor() as i64, (y * k as f64).floor() as i64);
        let (mut d1, mut d2, mut hue) = (f64::MAX, f64::MAX, 0.0);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (sx, sy, h) = point(cx + dx, cy + dy);
                let d = ((x - sx).powi(2) + (y - sy).powi(2)).sqrt();
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                    hue = h;
                } else if d < d2 {
                    d2 = d;
                }
            }
        }
        let seam = smooth(((d2 - d1) * k as f64 / 0.25).clamp(0.0, 1.0));
        let shade = (0.55 + 0.45 * (1.0 - d1 * k as f64)) * seam;
        let r = 0.5 + 0.5 * (TAU * hue).cos();
        let g = 0.5 + 0.5 * (TAU * (hue - 0.33)).cos();
        let b = 0.5 + 0.5 * (TAU * (hue - 0.67)).cos();
        rgb(shade * r, shade * g, shade * b)
    })
}

/// Marble: sine veins displaced by fractal noise, cream and slate blue.
fn marble() -> RgbaImage {
    let n = 256;
    RgbaImage::from_fn(n, n, |px, py| {
        let (x, y) = (f64::from(px) / f64::from(n), f64::from(py) / f64::from(n));
        let t = fbm(41, x, y, 3, 5);
        let v = 0.5 + 0.5 * (TAU * (2.0 * x + y) + 9.0 * t).sin();
        let v = v.powf(0.6);
        let (a, b) = ([0.93, 0.88, 0.78], [0.18, 0.24, 0.38]);
        rgb(
            b[0] + (a[0] - b[0]) * v,
            b[1] + (a[1] - b[1]) * v,
            b[2] + (a[2] - b[2]) * v,
        )
    })
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("textures"));
    let random = out.join("random");
    std::fs::create_dir_all(&random).expect("create output folders");
    let images = [
        (out.join("clouds.png"), clouds()),
        (random.join("weave.png"), weave()),
        (random.join("cells.png"), cells()),
        (random.join("marble.png"), marble()),
    ];
    for (path, img) in images {
        img.save(&path).expect("save PNG");
        println!("wrote {}", path.display());
    }
}
