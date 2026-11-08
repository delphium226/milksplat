//! Golden-image comparison for rendering tests.
//!
//! References live under the *calling* crate's `tests/golden/<name>.png`;
//! use [`assert_golden!`](crate::assert_golden) so the caller's
//! `CARGO_MANIFEST_DIR` is captured. Run with `MS_BLESS=1` to (re)write
//! references. This is a test helper: it panics on mismatch.

use std::path::{Path, PathBuf};

use image::RgbaImage;

/// A channel difference larger than this marks a pixel as different.
pub const MAX_CHANNEL_DELTA: u8 = 8;
/// The image passes if at most this fraction of pixels differ.
pub const MAX_DIFF_FRACTION: f64 = 0.005;

/// Compares `actual` against `<manifest_dir>/tests/golden/<name>.png`
/// (or writes it when `MS_BLESS=1`). Panics with a descriptive message on
/// mismatch, size mismatch or missing reference, writing
/// `<name>.actual.png` next to the reference.
pub fn check(actual: &RgbaImage, manifest_dir: &str, name: &str) {
    let bless = std::env::var("MS_BLESS").is_ok_and(|v| v == "1");
    check_with(actual, manifest_dir, name, bless);
}

/// [`check`] with the bless decision passed explicitly (so tests need not
/// mutate the process environment).
#[doc(hidden)]
pub fn check_with(actual: &RgbaImage, manifest_dir: &str, name: &str, bless: bool) {
    let dir = Path::new(manifest_dir).join("tests").join("golden");
    let reference_path = dir.join(format!("{name}.png"));
    let actual_path = dir.join(format!("{name}.actual.png"));

    if bless {
        if let Err(e) = std::fs::create_dir_all(&dir) {
            panic!("golden '{name}': cannot create {}: {e}", dir.display());
        }
        save(actual, &reference_path, name);
        let _ = std::fs::remove_file(&actual_path);
        return;
    }

    if !reference_path.is_file() {
        write_actual(actual, &dir, &actual_path, name);
        panic!(
            "golden '{name}': reference {} is missing; run the test with MS_BLESS=1 to create it \
             (actual image written to {})",
            reference_path.display(),
            actual_path.display()
        );
    }

    let reference = match image::open(&reference_path) {
        Ok(img) => img.to_rgba8(),
        Err(e) => panic!(
            "golden '{name}': cannot read reference {}: {e}",
            reference_path.display()
        ),
    };

    if reference.dimensions() != actual.dimensions() {
        write_actual(actual, &dir, &actual_path, name);
        panic!(
            "golden '{name}': size mismatch: reference is {:?}, actual is {:?} \
             (reference {}, actual written to {})",
            reference.dimensions(),
            actual.dimensions(),
            reference_path.display(),
            actual_path.display()
        );
    }

    let differing = reference
        .pixels()
        .zip(actual.pixels())
        .filter(|(r, a)| {
            r.0.iter()
                .zip(a.0.iter())
                .any(|(r, a)| r.abs_diff(*a) > MAX_CHANNEL_DELTA)
        })
        .count();
    let total = (actual.width() as usize) * (actual.height() as usize);
    let fraction = if total == 0 {
        0.0
    } else {
        differing as f64 / total as f64
    };

    if fraction > MAX_DIFF_FRACTION {
        write_actual(actual, &dir, &actual_path, name);
        panic!(
            "golden '{name}': {differing} of {total} pixels ({:.3}%) differ by more than \
             {MAX_CHANNEL_DELTA} in some channel (limit {:.1}%); reference {}, actual written \
             to {}; if the change is intended, rerun with MS_BLESS=1",
            fraction * 100.0,
            MAX_DIFF_FRACTION * 100.0,
            reference_path.display(),
            actual_path.display()
        );
    }
    let _ = std::fs::remove_file(&actual_path);
}

fn save(img: &RgbaImage, path: &PathBuf, name: &str) {
    if let Err(e) = img.save(path) {
        panic!("golden '{name}': cannot write {}: {e}", path.display());
    }
}

fn write_actual(actual: &RgbaImage, dir: &Path, actual_path: &PathBuf, name: &str) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        panic!("golden '{name}': cannot create {}: {e}", dir.display());
    }
    save(actual, actual_path, name);
}

/// Asserts that an [`image::RgbaImage`] matches the calling crate's
/// `tests/golden/<name>.png`. See [`golden::check`](crate::golden::check).
#[macro_export]
macro_rules! assert_golden {
    ($img:expr, $name:expr) => {
        $crate::golden::check(&$img, env!("CARGO_MANIFEST_DIR"), $name)
    };
}
