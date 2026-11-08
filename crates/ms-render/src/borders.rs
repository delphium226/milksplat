//! The outer and inner borders: frames around the edge of the screen,
//! drawn last (spec §6.2).

use crate::draw::{finite, Blend, Geometry};

/// The two borders, each `(size, straight-alpha colour)`.
///
/// `size` is the frame's width as a fraction of each axis (as in MilkDrop):
/// on a 160x120 target a size of 0.05 is 8 px at the left and right and 6 px
/// at the top and bottom. The outer border runs along the screen edge; the
/// inner border has its own size and sits immediately inside the outer one
/// (inside its size, even when the outer border is transparent). Sizes are
/// clamped so the frames never cross the centre; a size of 0 or an alpha of
/// 0 draws nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Borders {
    pub outer: (f32, [f32; 4]),
    pub inner: (f32, [f32; 4]),
}

/// Tessellates the outer border, then the inner one.
pub(crate) fn push_borders(g: &mut Geometry, b: &Borders) {
    let outer = finite(b.outer.0).clamp(0.0, 0.5);
    let inner = finite(b.inner.0).clamp(0.0, 0.5 - outer);
    frame(g, 0.0, outer, b.outer.1);
    frame(g, outer, inner, b.inner.1);
}

/// A frame `width` wide whose outside edge is `inset` from the screen edge,
/// as four non-overlapping bars (full-width bars at the top and bottom, the
/// sides between them).
fn frame(g: &mut Geometry, inset: f32, width: f32, color: [f32; 4]) {
    let color = g.color(color);
    if width <= 0.0 || color[3] <= 0.0 {
        return;
    }
    let (lo, hi) = (inset, 1.0 - inset);
    let blend = Blend::Normal;
    g.rect(blend, [lo, lo], [hi, lo + width], color);
    g.rect(blend, [lo, hi - width], [hi, hi], color);
    g.rect(blend, [lo, lo + width], [lo + width, hi - width], color);
    g.rect(blend, [hi - width, lo + width], [hi, hi - width], color);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_clamped() {
        let mut g = Geometry::for_tests([64, 64]);
        let white = [1.0; 4];
        push_borders(
            &mut g,
            &Borders {
                outer: (0.4, white),
                inner: (0.4, white),
            },
        );
        // The inner border is clamped to 0.1, so its bottom bar (the six
        // vertices after the outer frame's 24) ends at the centre.
        let bottom: Vec<f32> = g.vertices()[24..30].iter().map(|v| v.pos[1]).collect();
        assert!(bottom.iter().all(|&y| y == 0.4 || y == 0.5), "{bottom:?}");
        assert!(bottom.contains(&0.5));
    }
}
