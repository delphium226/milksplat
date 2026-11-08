//! Lazily paged big memory buffer backing `megabuf`/`gmegabuf`.

const PAGE: usize = 65_536;
const PAGES: usize = 16;
/// Total entries (1,048,576).
pub const MEGABUF_LEN: usize = PAGE * PAGES;

#[derive(Debug, Clone)]
pub struct Megabuf {
    pages: Vec<Option<Box<[f64]>>>,
}

impl Default for Megabuf {
    fn default() -> Self {
        Self::new()
    }
}

impl Megabuf {
    pub fn new() -> Self {
        Megabuf {
            pages: vec![None; PAGES],
        }
    }

    /// Floors `i`; returns `None` for NaN or out-of-range.
    fn slot(i: f64) -> Option<usize> {
        let f = i.floor();
        if f >= 0.0 && f < MEGABUF_LEN as f64 {
            Some(f as usize)
        } else {
            None
        }
    }

    pub fn get(&self, i: f64) -> f64 {
        match Self::slot(i) {
            Some(n) => match self.pages.get(n / PAGE) {
                Some(Some(p)) => p.get(n % PAGE).copied().unwrap_or(0.0),
                _ => 0.0,
            },
            None => 0.0,
        }
    }

    pub fn set(&mut self, i: f64, v: f64) {
        if let Some(n) = Self::slot(i) {
            if let Some(page) = self.pages.get_mut(n / PAGE) {
                let p = page.get_or_insert_with(|| vec![0.0; PAGE].into_boxed_slice());
                if let Some(cell) = p.get_mut(n % PAGE) {
                    *cell = v;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_floor() {
        let mut m = Megabuf::new();
        assert_eq!(m.get(3.0), 0.0);
        m.set(3.7, 5.0);
        assert_eq!(m.get(3.0), 5.0);
        assert_eq!(m.get(3.99), 5.0);
        m.set(70_000.0, 9.0);
        assert_eq!(m.get(70_000.0), 9.0);
        assert_eq!(m.get(2.0), 0.0);
    }

    #[test]
    fn bounds() {
        let mut m = Megabuf::new();
        m.set(0.0, 1.0);
        m.set(1_048_575.0, 2.0);
        assert_eq!(m.get(0.0), 1.0);
        assert_eq!(m.get(1_048_575.0), 2.0);
        assert_eq!(m.get(1_048_575.9), 2.0);
        for bad in [
            1_048_576.0,
            -1.0,
            -0.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e300,
        ] {
            m.set(bad, 7.0);
            assert_eq!(m.get(bad), 0.0, "{bad}");
        }
        assert_eq!(m.get(0.0), 1.0);
        assert_eq!(m.get(1_048_575.0), 2.0);
    }
}
