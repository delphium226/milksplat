//! Custom waves and the main waveform (spec §6.4).
//!
//! The main waveform has eight modes. Each is a function below whose doc
//! comment says what it looks like and how samples become points. Common to
//! all of them:
//!
//! - samples are sanitised (NaN/Inf become 0) and clamped to [-1, 1], so for
//!   any centre in [0, 1] every point stays within [-0.5, 1.5];
//! - `x`, `y` (`wave_x`, `wave_y`) place the wave's centre, in MilkDrop
//!   coordinates ([0, 1], origin at the bottom left);
//! - `mystery` (`wave_mystery`, clamped to [-1, 1]) changes one aspect of
//!   the shape, documented per mode;
//! - the radial modes (0, 1 and 5) divide their x offsets by `aspect[0]` and
//!   their y offsets by `aspect[1]` (see `draw::aspect_of`), so a circle
//!   stays round on a non-square target. The other modes span the screen in
//!   [0, 1] units;
//! - only the waveform is used (the spectrum field is unused in v1).

use std::f32::consts::{PI, TAU};

use crate::draw::{finite, point, unit, Blend, Geometry, MAX_WAVE_POINTS};

/// Samples per channel in [`MainWave::wave`].
pub(crate) const WAVE_SAMPLES: usize = 576;
/// Points along the main waveform (per line).
const MAIN_POINTS: usize = 512;

/// One custom wave, already evaluated by the preset's per-point code.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WaveDraw {
    /// MilkDrop coordinates; only the first 512 are used.
    pub points: Vec<[f32; 2]>,
    /// One straight-alpha colour per point. A shorter list repeats its last
    /// colour; an empty list draws nothing.
    pub colors: Vec<[f32; 4]>,
    /// Draw a 2x2 px dot per point instead of lines.
    pub dots: bool,
    pub thick: bool,
    pub additive: bool,
}

/// The main waveform.
#[derive(Clone, Debug, PartialEq)]
pub struct MainWave {
    /// Shape, 0..=7 (other values wrap modulo 8).
    pub mode: u8,
    /// Left and right channel samples, nominally in [-1, 1].
    pub wave: [[f32; WAVE_SAMPLES]; 2],
    /// Unused in v1.
    pub spectrum: [f32; 512],
    /// Straight-alpha colour.
    pub color: [f32; 4],
    /// Centre of the wave.
    pub x: f32,
    pub y: f32,
    /// Mode-specific modifier in [-1, 1].
    pub mystery: f32,
    pub dots: bool,
    pub thick: bool,
    pub additive: bool,
    /// Scale the colour so its largest channel is 1 (before alpha).
    pub brighten: bool,
}

impl Default for MainWave {
    /// Mode 0, silent, white, centred.
    fn default() -> Self {
        MainWave {
            mode: 0,
            wave: [[0.0; WAVE_SAMPLES]; 2],
            spectrum: [0.0; 512],
            color: [1.0; 4],
            x: 0.5,
            y: 0.5,
            mystery: 0.0,
            dots: false,
            thick: false,
            additive: false,
            brighten: false,
        }
    }
}

/// Sanitised inputs shared by the mode functions.
struct Input {
    left: [f32; WAVE_SAMPLES],
    right: [f32; WAVE_SAMPLES],
    centre: [f32; 2],
    mystery: f32,
    aspect: [f32; 2],
}

impl Input {
    fn new(w: &MainWave, aspect: [f32; 2]) -> Input {
        let sample = |s: f32| finite(s).clamp(-1.0, 1.0);
        Input {
            left: w.wave[0].map(sample),
            right: w.wave[1].map(sample),
            centre: point([w.x, w.y]),
            mystery: finite(w.mystery).clamp(-1.0, 1.0),
            // Values below 1 (or not finite) count as 1.
            aspect: aspect.map(|a| finite(a).clamp(1.0, 1000.0)),
        }
    }

    /// The centre plus a radial offset, divided by the aspect.
    fn radial(&self, d: [f32; 2]) -> [f32; 2] {
        [
            self.centre[0] + d[0] / self.aspect[0],
            self.centre[1] + d[1] / self.aspect[1],
        ]
    }

    /// x of point `i` of a horizontal line: the line is 1 wide, centred on
    /// the wave's x.
    fn line_x(&self, i: usize) -> f32 {
        self.centre[0] - 0.5 + i as f32 / (MAIN_POINTS - 1) as f32
    }
}

/// `v` rotated counter-clockwise by `a` radians.
fn rotate([x, y]: [f32; 2], a: f32) -> [f32; 2] {
    let (s, c) = a.sin_cos();
    [x * c - y * s, x * s + y * c]
}

/// The main waveform's points for its mode (see the mode functions). For
/// mode 7 the result is the left line's points followed by the right line's
/// (two separate lines of equal length). `aspect` is `[aspect_x, aspect_y]`
/// as described in the module docs.
pub fn main_wave_points(w: &MainWave, aspect: [f32; 2]) -> Vec<[f32; 2]> {
    let input = Input::new(w, aspect);
    let points = match w.mode % 8 {
        0 => mode0_ring(&input),
        1 => mode1_scope(&input),
        2 | 3 => mode2_line(&input),
        4 => mode4_derivative(&input),
        5 => mode5_explosive(&input),
        6 => mode6_angled(&input),
        _ => mode7_dual(&input),
    };
    // The modes stay well inside this; it only guards the arithmetic.
    points.into_iter().map(point).collect()
}

/// Mode 0, ring: a closed circle around the centre whose radius breathes
/// with the left channel. Point `i` of 512 sits at angle `2 pi i / 512`
/// (counter-clockwise from +x) and radius `0.25 + 0.1 * mystery + 0.1 *
/// s(i)`; a 513th point repeats the first to close the loop. So that the
/// closing segment has no step, `s` cross-fades the first 64 samples with
/// the 64 that follow sample 511: `s(i) = mix(left[512 + i], left[i], i /
/// 64)` for `i < 64`, else `left[i]`. `mystery` grows or shrinks the ring.
fn mode0_ring(input: &Input) -> Vec<[f32; 2]> {
    const FADE: usize = WAVE_SAMPLES - MAIN_POINTS;
    let base = 0.25 + 0.1 * input.mystery;
    let mut points: Vec<[f32; 2]> = (0..MAIN_POINTS)
        .map(|i| {
            let s = if i < FADE {
                let t = i as f32 / FADE as f32;
                input.left[MAIN_POINTS + i] * (1.0 - t) + input.left[i] * t
            } else {
                input.left[i]
            };
            let r = base + 0.1 * s;
            let a = TAU * i as f32 / MAIN_POINTS as f32;
            input.radial([r * a.cos(), r * a.sin()])
        })
        .collect();
    if let Some(&first) = points.first() {
        points.push(first);
    }
    points
}

/// Mode 1, X-Y oscilloscope: a Lissajous figure around the centre, the left
/// channel driving x and the right channel y. Point `i` is `0.3 * (left[i],
/// right[i])`, rotated counter-clockwise by `mystery * 45` degrees, for the
/// first 512 samples, joined in order.
fn mode1_scope(input: &Input) -> Vec<[f32; 2]> {
    let angle = input.mystery * PI / 4.0;
    (0..MAIN_POINTS)
        .map(|i| {
            let d = rotate([input.left[i], input.right[i]], angle);
            input.radial([0.3 * d[0], 0.3 * d[1]])
        })
        .collect()
}

/// Mode 2, horizontal line (and mode 3, which uses the same points): a line
/// one screen wide centred on the wave's position, displaced vertically by
/// the left channel. Point `i` of 512 is at `x - 0.5 + i / 511` and `y + A *
/// left[i]`, with amplitude `A = 0.2 * (1 + 0.5 * mystery)`, so `mystery`
/// sets the height (0.1 to 0.3). Mode 3 also scales the alpha by the volume
/// (see `mode3_alpha`).
fn mode2_line(input: &Input) -> Vec<[f32; 2]> {
    let amp = 0.2 * (1.0 + 0.5 * input.mystery);
    (0..MAIN_POINTS)
        .map(|i| [input.line_x(i), input.centre[1] + amp * input.left[i]])
        .collect()
}

/// Mode 3's alpha factor: the overall volume, the mean of `|sample|` over
/// both channels' 576 samples, clamped to [0, 1]. Silence is invisible.
fn mode3_alpha(w: &MainWave) -> f32 {
    let sum: f32 = w
        .wave
        .iter()
        .flatten()
        .map(|&s| finite(s).clamp(-1.0, 1.0).abs())
        .sum();
    (sum / (2 * WAVE_SAMPLES) as f32).clamp(0.0, 1.0)
}

/// Mode 4, derivative line: the same horizontal line as mode 2, but
/// displaced by the change from one left sample to the next, which
/// emphasises the high frequencies. Point `i` of 512 is at `x - 0.5 + i /
/// 511` and `y + A * clamp(4 * (left[i + 1] - left[i]), -1, 1)`, with `A =
/// 0.2 * (1 + 0.5 * mystery)` as in mode 2 (the factor 4 makes the small
/// sample-to-sample differences visible).
fn mode4_derivative(input: &Input) -> Vec<[f32; 2]> {
    let amp = 0.2 * (1.0 + 0.5 * input.mystery);
    (0..MAIN_POINTS)
        .map(|i| {
            let d = (4.0 * (input.left[i + 1] - input.left[i])).clamp(-1.0, 1.0);
            [input.line_x(i), input.centre[1] + amp * d]
        })
        .collect()
}

/// Mode 5, explosive X-Y: like mode 1 (left drives x, right drives y), but
/// every point is pushed away from the centre in proportion to its own
/// magnitude, so loud moments burst outwards. With `p = (left[i],
/// right[i])` and push `k = 2 + mystery` (1 to 3), point `i` is `0.06 * p *
/// (1 + k * |p|)` from the centre, for the first 512 samples, joined in
/// order.
fn mode5_explosive(input: &Input) -> Vec<[f32; 2]> {
    let k = 2.0 + input.mystery;
    (0..MAIN_POINTS)
        .map(|i| {
            let p = [input.left[i], input.right[i]];
            let m = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let s = 0.06 * (1.0 + k * m);
            input.radial([s * p[0], s * p[1]])
        })
        .collect()
}

/// Mode 6, angled line: a line 0.8 long through the centre, at
/// `mystery * 180` degrees counter-clockwise from horizontal, displaced
/// perpendicular to itself by the left channel. Point `i` of 512 is the
/// centre plus `t * dir + 0.2 * left[i] * normal`, with `t` running from
/// -0.4 to 0.4, `dir = (cos, sin)` of the angle and `normal` the direction
/// turned 90 degrees counter-clockwise. The angle is in [0, 1] units (not
/// aspect corrected).
fn mode6_angled(input: &Input) -> Vec<[f32; 2]> {
    let (s, c) = (input.mystery * PI).sin_cos();
    (0..MAIN_POINTS)
        .map(|i| {
            let t = -0.4 + 0.8 * i as f32 / (MAIN_POINTS - 1) as f32;
            let o = 0.2 * input.left[i];
            [
                input.centre[0] + t * c - o * s,
                input.centre[1] + t * s + o * c,
            ]
        })
        .collect()
}

/// Mode 7, dual lines: two horizontal lines like mode 2, the left channel
/// 0.1 above the centre and the right channel 0.1 below it. Point `i` of
/// each is at `x - 0.5 + i / 511`, and `y + 0.1 + A * left[i]` or `y - 0.1 +
/// A * right[i]`, with `A = 0.15 * (1 + 0.5 * mystery)` (0.075 to 0.225).
/// Returns the 512 left points, then the 512 right points.
fn mode7_dual(input: &Input) -> Vec<[f32; 2]> {
    let amp = 0.15 * (1.0 + 0.5 * input.mystery);
    let line = |offset: f32, samples: &[f32; WAVE_SAMPLES]| -> Vec<[f32; 2]> {
        (0..MAIN_POINTS)
            .map(|i| [input.line_x(i), input.centre[1] + offset + amp * samples[i]])
            .collect()
    };
    let mut points = line(0.1, &input.left);
    points.extend(line(-0.1, &input.right));
    points
}

/// Tessellates a custom wave.
pub(crate) fn push_wave(g: &mut Geometry, wave: &WaveDraw) {
    let Some(&last_color) = wave.colors.last() else {
        return;
    };
    let blend = Blend::from_additive(wave.additive);
    let points: Vec<([f32; 2], [f32; 4])> = wave
        .points
        .iter()
        .take(MAX_WAVE_POINTS)
        .enumerate()
        .map(|(i, &p)| {
            let c = wave.colors.get(i).copied().unwrap_or(last_color);
            (point(p), g.color(c))
        })
        .collect();
    push_points(g, blend, &points, wave.dots, wave.thick);
}

fn push_points(
    g: &mut Geometry,
    blend: Blend,
    points: &[([f32; 2], [f32; 4])],
    dots: bool,
    thick: bool,
) {
    if dots {
        for &(p, c) in points {
            g.dot(blend, p, c);
        }
    } else {
        g.polyline(blend, points, thick);
    }
}

/// Tessellates the main waveform.
pub(crate) fn push_main_wave(g: &mut Geometry, w: &MainWave) {
    let mut color = w.color.map(unit);
    if w.brighten {
        let max = color[0].max(color[1]).max(color[2]);
        if max > 0.0 {
            for c in &mut color[..3] {
                *c /= max;
            }
        }
    }
    if w.mode % 8 == 3 {
        color[3] *= mode3_alpha(w);
    }
    let color = g.color(color);
    if color[3] <= 0.0 {
        return;
    }
    let blend = Blend::from_additive(w.additive);
    let points: Vec<([f32; 2], [f32; 4])> = main_wave_points(w, g.aspect)
        .into_iter()
        .map(|p| (p, color))
        .collect();
    if w.mode % 8 == 7 {
        let (left, right) = points.split_at(points.len() / 2);
        push_points(g, blend, left, w.dots, w.thick);
        push_points(g, blend, right, w.dots, w.thick);
    } else {
        push_points(g, blend, &points, w.dots, w.thick);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_seam_is_continuous() {
        // A ramp: without the cross-fade, the closing segment would jump.
        let mut w = MainWave::default();
        for (i, s) in w.wave[0].iter_mut().enumerate() {
            *s = i as f32 / WAVE_SAMPLES as f32;
        }
        let p = main_wave_points(&w, [1.0, 1.0]);
        let radius = |q: [f32; 2]| ((q[0] - 0.5).powi(2) + (q[1] - 0.5).powi(2)).sqrt();
        let n = p.len();
        let step = (radius(p[n - 2]) - radius(p[n - 1])).abs();
        assert!(step < 0.001, "seam step {step}");
    }

    #[test]
    fn mode3_alpha_is_mean_volume() {
        let mut w = MainWave::default();
        assert_eq!(mode3_alpha(&w), 0.0);
        w.wave[0] = [0.5; WAVE_SAMPLES];
        w.wave[1] = [-0.5; WAVE_SAMPLES];
        assert!((mode3_alpha(&w) - 0.5).abs() < 1e-6);
        w.wave[0] = [f32::NAN; WAVE_SAMPLES];
        w.wave[1] = [9.0; WAVE_SAMPLES];
        assert!((mode3_alpha(&w) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mode7_is_two_lines() {
        let w = MainWave {
            mode: 7,
            ..MainWave::default()
        };
        let p = main_wave_points(&w, [1.0, 1.0]);
        assert_eq!(p.len(), 2 * MAIN_POINTS);
        assert!(p[..MAIN_POINTS].iter().all(|q| (q[1] - 0.6).abs() < 1e-6));
        assert!(p[MAIN_POINTS..].iter().all(|q| (q[1] - 0.4).abs() < 1e-6));
    }

    #[test]
    fn mode6_follows_mystery_angle() {
        let w = MainWave {
            mode: 6,
            mystery: 0.5,
            ..MainWave::default()
        };
        let p = main_wave_points(&w, [1.0, 1.0]);
        // Vertical line through the centre.
        assert!(p.iter().all(|q| (q[0] - 0.5).abs() < 1e-5));
        assert!((p[0][1] - 0.1).abs() < 1e-5 && (p[MAIN_POINTS - 1][1] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn modes_wrap() {
        let a = MainWave {
            mode: 9,
            ..MainWave::default()
        };
        let b = MainWave {
            mode: 1,
            ..MainWave::default()
        };
        assert_eq!(
            main_wave_points(&a, [1.0, 1.0]),
            main_wave_points(&b, [1.0, 1.0])
        );
    }
}
