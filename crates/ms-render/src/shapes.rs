//! Custom shapes (spec §6.4): regular polygons with a centre-to-rim colour
//! gradient, an optional border, and optional texturing from the previous
//! feedback image.

use std::f32::consts::{FRAC_PI_4, TAU};

use crate::draw::{coord, finite, point, Blend, Geometry, Vertex};

/// Fewest and most polygon sides.
const MIN_SIDES: u32 = 3;
const MAX_SIDES: u32 = 100;
/// Smallest `|tex_zoom|` (smaller values, and 0, are raised to it).
const MIN_TEX_ZOOM: f32 = 1e-3;

/// One shape instance, already evaluated by the shape's per-instance code.
///
/// The shape is a regular polygon of `sides` (clamped to 3..=100) centred
/// on `(x, y)` (MilkDrop coordinates) with circumradius `rad`, measured in
/// [0, 1] units of the target's shorter side (so the polygon is regular on
/// screen). Its first vertex is at `ang + 45` degrees counter-clockwise from
/// +x, so a 4-sided shape at `ang = 0` is an axis-aligned square.
///
/// The fill is a triangle fan from the centre (`inner` colour) to the rim
/// (`outer` colour). The border, when `border` has alpha, is a line loop
/// around the rim (thick when `thick_outline`). `additive` applies to both.
///
/// When `textured`, the fill's colour multiplies the previous feedback image
/// (the frame the warp pass read from): the centre samples the image at the
/// shape's centre, and a rim vertex at offset `d` from the centre samples
/// at offset `rotate(d, tex_ang) / tex_zoom` (counter-clockwise, in the same
/// aspect-corrected units). With `tex_zoom = 1` and `tex_ang = 0` the shape
/// shows exactly what was under it; `tex_zoom > 1` magnifies.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeDraw {
    pub x: f32,
    pub y: f32,
    pub rad: f32,
    pub ang: f32,
    pub sides: u32,
    /// Straight-alpha colours.
    pub inner: [f32; 4],
    pub outer: [f32; 4],
    pub border: [f32; 4],
    pub additive: bool,
    pub thick_outline: bool,
    pub textured: bool,
    pub tex_zoom: f32,
    pub tex_ang: f32,
}

impl Default for ShapeDraw {
    /// An invisible (all colours transparent) square of radius 0.1 at the
    /// centre, with an identity texture mapping.
    fn default() -> Self {
        ShapeDraw {
            x: 0.5,
            y: 0.5,
            rad: 0.1,
            ang: 0.0,
            sides: 4,
            inner: [0.0; 4],
            outer: [0.0; 4],
            border: [0.0; 4],
            additive: false,
            thick_outline: false,
            textured: false,
            tex_zoom: 1.0,
            tex_ang: 0.0,
        }
    }
}

/// Tessellates one shape instance: the fill, then the border.
pub(crate) fn push_shape(g: &mut Geometry, s: &ShapeDraw) {
    let inner = g.color(s.inner);
    let outer = g.color(s.outer);
    let border = g.color(s.border);
    let fill = inner[3] > 0.0 || outer[3] > 0.0;
    if !fill && border[3] <= 0.0 {
        return;
    }
    let sides = s.sides.clamp(MIN_SIDES, MAX_SIDES);
    let centre = point([s.x, s.y]);
    let rad = finite(s.rad).clamp(0.0, 10.0);
    let start = finite(s.ang).rem_euclid(TAU) + FRAC_PI_4;
    let [ax, ay] = g.aspect;
    let blend = Blend::from_additive(s.additive);

    // Rim vertex `i`'s offset in square units (before the aspect
    // division); `i == sides` wraps to the first vertex.
    let offset = |i: u32| {
        let a = start + TAU * (i % sides) as f32 / sides as f32;
        [rad * a.cos(), rad * a.sin()]
    };
    let rim = |d: [f32; 2]| point([centre[0] + d[0] / ax, centre[1] + d[1] / ay]);

    if fill {
        // (1 / zoom, sin, cos) of the texture mapping.
        let tex = s.textured.then(|| {
            let zoom = finite(s.tex_zoom);
            let zoom = zoom
                .abs()
                .clamp(MIN_TEX_ZOOM, 1.0 / MIN_TEX_ZOOM)
                .copysign(zoom);
            let (sin, cos) = finite(s.tex_ang).rem_euclid(TAU).sin_cos();
            (1.0 / zoom, sin, cos)
        });
        let uv = |d: [f32; 2]| match tex {
            Some((k, sin, cos)) => {
                let r = [(d[0] * cos - d[1] * sin) * k, (d[0] * sin + d[1] * cos) * k];
                // Texture space: v = 0 at the top.
                [
                    coord(centre[0] + r[0] / ax),
                    coord(1.0 - (centre[1] + r[1] / ay)),
                ]
            }
            None => [0.0; 2],
        };
        let textured = if tex.is_some() { 1.0 } else { 0.0 };
        let vertex = |d: [f32; 2], color| Vertex {
            pos: rim(d),
            uv: uv(d),
            color,
            textured,
        };
        let c = vertex([0.0; 2], inner);
        for i in 0..sides {
            g.tri(
                blend,
                [c, vertex(offset(i), outer), vertex(offset(i + 1), outer)],
            );
        }
    }

    if border[3] > 0.0 {
        for i in 0..sides {
            g.segment(
                blend,
                (rim(offset(i)), border),
                (rim(offset(i + 1)), border),
                s.thick_outline,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> Geometry {
        Geometry::for_tests([200, 100])
    }

    #[test]
    fn fan_has_one_triangle_per_side_and_is_regular_on_screen() {
        let mut g = geometry();
        push_shape(
            &mut g,
            &ShapeDraw {
                sides: 6,
                rad: 0.4,
                inner: [1.0; 4],
                outer: [1.0; 4],
                ..ShapeDraw::default()
            },
        );
        let v = g.vertices();
        assert_eq!(v.len(), 6 * 3);
        // On the 200x100 target every rim vertex is 0.4 * 100 = 40 px from
        // the centre.
        for t in v.chunks(3) {
            for r in &t[1..] {
                let px = [(r.pos[0] - 0.5) * 200.0, (r.pos[1] - 0.5) * 100.0];
                let d = (px[0] * px[0] + px[1] * px[1]).sqrt();
                assert!((d - 40.0).abs() < 1e-3, "{d}");
            }
        }
    }

    #[test]
    fn sides_are_clamped_and_invisible_shapes_skipped() {
        let mut g = geometry();
        let visible = ShapeDraw {
            inner: [1.0; 4],
            ..ShapeDraw::default()
        };
        push_shape(
            &mut g,
            &ShapeDraw {
                sides: 1,
                ..visible.clone()
            },
        );
        assert_eq!(g.vertices().len(), 3 * 3);
        push_shape(
            &mut g,
            &ShapeDraw {
                sides: 1000,
                ..visible
            },
        );
        assert_eq!(g.vertices().len(), 3 * 3 + 100 * 3);
        push_shape(&mut g, &ShapeDraw::default());
        assert_eq!(g.vertices().len(), 3 * 3 + 100 * 3);
    }
}
