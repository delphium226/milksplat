//! Motion vectors (spec §6.2, drawn first): a grid of short line segments.
//!
//! v1 does not measure actual motion. Each segment starts at its grid point
//! and points toward the screen centre, which is simple, deterministic and
//! visible.

use crate::draw::{finite, point, Blend, Geometry};

/// Most grid columns and rows.
const MAX_NX: f32 = 64.0;
const MAX_NY: f32 = 48.0;
/// Length of one `len` unit, in [0, 1] screen units (1% of the screen).
const MV_STEP: f32 = 0.01;
/// Largest `len` (a full screen).
const MAX_LEN: f32 = 100.0;

/// The motion-vector grid (`mv_*` preset variables).
///
/// `nx` x `ny` segments (each rounded down, capped at 64 x 48; fewer than
/// one column or row draws nothing). Grid point `(i, j)` is at `((i + 0.5)
/// / nx + dx, (j + 0.5) / ny + dy)` in MilkDrop coordinates; its segment
/// runs from there toward the screen centre, `len * 0.01` long (`len`
/// clamped to [0, 100]). Segments are thin lines in `color` (straight
/// alpha; alpha 0 draws nothing), blended normally.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotionVectors {
    pub nx: f32,
    pub ny: f32,
    pub dx: f32,
    pub dy: f32,
    pub len: f32,
    pub color: [f32; 4],
}

/// Tessellates the motion-vector grid.
pub(crate) fn push_motion(g: &mut Geometry, mv: &MotionVectors) {
    let color = g.color(mv.color);
    // NaN becomes 0 columns; infinity counts as 0 too (`finite`).
    let nx = finite(mv.nx).clamp(0.0, MAX_NX).floor();
    let ny = finite(mv.ny).clamp(0.0, MAX_NY).floor();
    let length = finite(mv.len).clamp(0.0, MAX_LEN) * MV_STEP;
    if color[3] <= 0.0 || nx < 1.0 || ny < 1.0 || length <= 0.0 {
        return;
    }
    let offset = point([mv.dx, mv.dy]);
    for j in 0..ny as u32 {
        for i in 0..nx as u32 {
            let p = point([
                (i as f32 + 0.5) / nx + offset[0],
                (j as f32 + 0.5) / ny + offset[1],
            ]);
            let to_centre = [0.5 - p[0], 0.5 - p[1]];
            let d = (to_centre[0] * to_centre[0] + to_centre[1] * to_centre[1]).sqrt();
            if d < 1e-6 {
                continue;
            }
            let q = [
                p[0] + to_centre[0] / d * length,
                p[1] + to_centre[1] / d * length,
            ];
            g.segment(Blend::Normal, (p, color), (point(q), color), false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_rounded_down_and_capped() {
        let mv = MotionVectors {
            nx: 3.9,
            ny: 2.0,
            len: 1.0,
            color: [1.0; 4],
            ..MotionVectors::default()
        };
        let mut g = Geometry::for_tests([64, 64]);
        push_motion(&mut g, &mv);
        assert_eq!(g.vertices().len(), 3 * 2 * 6);
        let mut g = Geometry::for_tests([64, 64]);
        push_motion(
            &mut g,
            &MotionVectors {
                nx: 1000.0,
                ny: 1000.0,
                ..mv
            },
        );
        assert_eq!(g.vertices().len(), 64 * 48 * 6);
    }
}
