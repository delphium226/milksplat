//! Finding user texture files (spec §4, §6.1), and the spectrum texture.
//!
//! A preset names a texture by its file name without extension (the shader
//! prefix already stripped). A [`TextureResolver`] holds the roots in lookup
//! order: the preset's own folder, the user `Presets\textures\`, then the
//! bundled `textures\`. It is stateless, so one resolver can serve many
//! preset loads; the per-preset random choices live in a caller-owned
//! [`PresetPicks`].
//!
//! Untrusted input is handled without panicking or unbounded allocation:
//! - names that could leave a root (path separators, `..`, a drive colon)
//!   are rejected;
//! - files over [`MAX_FILE_BYTES`] are rejected unread;
//! - the decoder rejects images over [`MAX_DECODE_DIM`] per side (from the
//!   header, before decoding) and allocations over [`MAX_DECODE_BYTES`];
//! - images over [`MAX_TEXTURE_DIM`] per side are downscaled to fit, keeping
//!   their aspect ratio, with a box filter in the image's own pixel type
//!   (`DynamicImage::thumbnail_exact`), so peak memory stays under about
//!   twice the decoded size.
//!
//! A file that fails any of these is logged and the next candidate is tried.

use std::collections::HashMap;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use image::RgbaImage;
use ms_expr::Rng;

use crate::feedback::Renderer;
use crate::gpu::Gpu;

/// Largest file read, in bytes (64 MiB).
pub const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// Largest texture side after loading; bigger images are downscaled.
pub const MAX_TEXTURE_DIM: u32 = 4096;

/// Largest image side the decoder accepts (checked against the header, so
/// a small file that would decode to a huge image is never decoded).
pub const MAX_DECODE_DIM: u32 = 8192;

/// Largest allocation the decoder may make, in bytes (256 MiB: an
/// 8192x8192 RGBA8 image).
pub const MAX_DECODE_BYTES: u64 = 256 * 1024 * 1024;

/// Extensions tried, in order, in each root.
const EXTENSIONS: [&str; 2] = ["png", "jpg"];

/// Looks up user textures by name across a list of root folders.
///
/// Stateless (only the roots), cheap to clone and `Send + Sync`: the loader
/// can keep one for every preset. Random textures (`rand00`..`rand15`) need
/// a per-preset [`PresetPicks`]; see [`TextureResolver::resolve_with`].
///
/// Symbolic links (and junctions) under a root are followed like ordinary
/// files. Only a plain file name is ever joined onto a root, and only
/// decoded pixels reach the GPU, so a link can at most show another image
/// file as a texture.
#[derive(Clone, Debug, Default)]
pub struct TextureResolver {
    /// Folders searched in order.
    roots: Vec<PathBuf>,
}

/// The random-texture state of one preset load: the resolver's own random
/// stream (not the expression VM's `rand()` generator) and the file picked
/// for each `randNN`, so the same name resolves to the same file however
/// often the preset's shaders bind it (`fw_rand03` and `pc_rand03`).
#[derive(Clone, Debug)]
pub struct PresetPicks {
    rng: Rng,
    /// `NN` of `randNN` to the file picked for it.
    picks: HashMap<u8, PathBuf>,
}

impl PresetPicks {
    /// Fresh picks drawing from a stream seeded with `seed` (the engine
    /// derives it from the preset's seed).
    pub fn new(seed: u64) -> Self {
        PresetPicks {
            rng: Rng::seeded(seed),
            picks: HashMap::new(),
        }
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        ((self.rng.next_f64() * n as f64) as usize).min(n.saturating_sub(1))
    }
}

impl TextureResolver {
    pub fn new(roots: Vec<PathBuf>) -> Self {
        TextureResolver { roots }
    }

    /// Loads `<root>/<name>.png`, then `<root>/<name>.jpg`, for each root in
    /// order, returning the first that loads. Random names (`rand00`..
    /// `rand15`, any case) and unsafe names give `None`; use
    /// [`TextureResolver::resolve_with`] for random names.
    pub fn resolve(&self, name_without_prefix: &str) -> Option<RgbaImage> {
        let name = name_without_prefix;
        if !is_safe_name(name) {
            tracing::warn!("texture name {name:?} rejected: it is not a plain file name");
            return None;
        }
        if random_index(name).is_some() {
            tracing::debug!("texture {name}: random textures need resolve_with");
            return None;
        }
        self.roots
            .iter()
            .flat_map(|root| EXTENSIONS.map(|ext| root.join(format!("{name}.{ext}"))))
            .filter(|path| path.is_file())
            .find_map(|path| load(&path))
    }

    /// Like [`TextureResolver::resolve`], but `rand00`..`rand15` (any case)
    /// pick a file uniformly at random, with `picks`' stream, from
    /// `<root>/random/*.png|jpg` of the first root whose `random` folder has
    /// any.
    ///
    /// A file already picked for that `NN` in `picks` is used again. If a
    /// pick does not load, the remaining files are tried in a random order
    /// (a lazily drawn shuffle whose first element is the pick); only a file
    /// that loads is remembered. `None` if nothing loads.
    pub fn resolve_with(
        &self,
        name_without_prefix: &str,
        picks: &mut PresetPicks,
    ) -> Option<RgbaImage> {
        let name = name_without_prefix;
        let Some(index) = random_index(name).filter(|_| is_safe_name(name)) else {
            return self.resolve(name);
        };
        if let Some(path) = picks.picks.get(&index) {
            if let Some(img) = load(path) {
                return Some(img);
            }
            picks.picks.remove(&index);
        }
        let Some(mut candidates) = self.random_candidates() else {
            tracing::warn!("texture {name}: no images in any random folder");
            return None;
        };
        for k in 0..candidates.len() {
            let j = k + picks.below(candidates.len() - k);
            candidates.swap(k, j);
            let path = &candidates[k];
            if let Some(img) = load(path) {
                tracing::debug!("texture {name}: picked {}", path.display());
                picks.picks.insert(index, path.clone());
                return Some(img);
            }
        }
        tracing::warn!("texture {name}: no file in the random folder loads");
        None
    }

    /// The image files in the first root's `random` folder that has any,
    /// sorted by path (so a seeded pick is reproducible).
    fn random_candidates(&self) -> Option<Vec<PathBuf>> {
        self.roots.iter().find_map(|root| {
            let dir = root.join("random");
            let entries = std::fs::read_dir(&dir).ok()?;
            let mut files: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| has_image_extension(p) && p.is_file())
                .collect();
            files.sort();
            (!files.is_empty()).then_some(files)
        })
    }
}

/// `NN` for `rand00`..`rand15` (the `rand` prefix in any case).
fn random_index(name: &str) -> Option<u8> {
    let (prefix, digits) = (name.get(..4)?, name.get(4..)?);
    if !prefix.eq_ignore_ascii_case("rand")
        || digits.len() != 2
        || !digits.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    digits.parse::<u8>().ok().filter(|&n| n <= 15)
}

/// A plain file name: not empty, no path separators, no `..`, no drive
/// colon.
fn is_safe_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['/', '\\', ':', '\0']) && !name.contains("..")
}

fn has_image_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTENSIONS.iter().any(|x| e.eq_ignore_ascii_case(x)))
}

/// Reads and decodes one file, downscaling it to fit
/// [`MAX_TEXTURE_DIM`]. Logs and returns `None` on any failure.
fn load(path: &Path) -> Option<RgbaImage> {
    match try_load(path) {
        Ok(img) => Some(img),
        Err(e) => {
            tracing::warn!("texture {}: {e}", path.display());
            None
        }
    }
}

fn try_load(path: &Path) -> Result<RgbaImage, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("cannot open: {e}"))?;
    let len = file
        .metadata()
        .map_err(|e| format!("cannot stat: {e}"))?
        .len();
    if len > MAX_FILE_BYTES {
        return Err(format!(
            "{len} bytes is over the {MAX_FILE_BYTES}-byte limit"
        ));
    }
    // Bounded read, in case the file grew after the size check.
    let mut bytes = Vec::with_capacity(len as usize);
    file.take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read: {e}"))?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(format!("over the {MAX_FILE_BYTES}-byte limit"));
    }

    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("cannot read: {e}"))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DECODE_DIM);
    limits.max_image_height = Some(MAX_DECODE_DIM);
    limits.max_alloc = Some(MAX_DECODE_BYTES);
    reader.limits(limits);
    // The decoded buffer is at most MAX_DECODE_BYTES.
    let img = reader.decode().map_err(|e| format!("cannot decode: {e}"))?;

    let (w, h) = (img.width(), img.height());
    if w <= MAX_TEXTURE_DIM && h <= MAX_TEXTURE_DIM {
        // Free for RGBA8 sources.
        return Ok(img.into_rgba8());
    }
    let [nw, nh] = fit_within([w, h], MAX_TEXTURE_DIM);
    tracing::info!(
        "texture {}: {w}x{h} downscaled to {nw}x{nh}",
        path.display()
    );
    // A box filter in the image's own pixel type, straight into a buffer of
    // the output size (no floating-point intermediate image), then the
    // (smaller) result is converted to RGBA8: peak memory is the decoded
    // image plus at most twice the output.
    Ok(img.thumbnail_exact(nw, nh).into_rgba8())
}

/// The largest size with `size`'s aspect ratio (rounded) whose sides are at
/// most `max`, each side at least 1.
fn fit_within([w, h]: [u32; 2], max: u32) -> [u32; 2] {
    // a * max / b, rounded, in 64 bits.
    let scale = |a: u32, b: u32| {
        let b = u64::from(b.max(1));
        let v = (u64::from(a) * u64::from(max) + b / 2) / b;
        (v as u32).clamp(1, max)
    };
    if w >= h {
        [max, scale(h, w)]
    } else {
        [scale(w, h), max]
    }
}

/// Number of spectrum bins in `sampler_spectrum`.
pub const SPECTRUM_BINS: usize = 512;

/// Creates the spectrum texture bound at group 0 binding 10: 512x1
/// `R16Float`, zeroed.
///
/// `R16Float` rather than `R32Float`: `R32Float` is not filterable without
/// an optional device feature, and the prelude declares every group 0
/// texture filterable so that any sampler (`s_fw`, `s_fc`, ...) works with
/// it. Half precision (11 significant bits) is plenty for spectrum levels.
pub(crate) fn create_spectrum_texture(gpu: &Gpu) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("spectrum"),
        size: wgpu::Extent3d {
            width: SPECTRUM_BINS as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

impl Renderer {
    /// Uploads the spectrum into `sampler_spectrum` (bin i at
    /// `x = (i + 0.5) / 512`), in place. Values are stored as f16:
    /// non-finite values become 0 and the rest are clamped to +-65504.
    pub fn set_spectrum(&mut self, gpu: &Gpu, spectrum: &[f32; SPECTRUM_BINS]) {
        let mut bytes = [0u8; SPECTRUM_BINS * 2];
        for (out, &v) in bytes.as_chunks_mut::<2>().0.iter_mut().zip(spectrum) {
            let v = if v.is_finite() {
                v.clamp(-65504.0, 65504.0)
            } else {
                0.0
            };
            *out = half::f16::from_f32(v).to_le_bytes();
        }
        let tex = &self.spectrum;
        gpu.queue.write_texture(
            tex.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SPECTRUM_BINS as u32 * 2),
                rows_per_image: Some(1),
            },
            tex.size(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_names() {
        for (n, i) in [
            ("rand00", 0),
            ("rand09", 9),
            ("rand15", 15),
            ("RAND03", 3),
            ("Rand14", 14),
        ] {
            assert_eq!(random_index(n), Some(i), "{n}");
        }
        for n in [
            "rand16",
            "rand3",
            "rand003",
            "rand-1",
            "rand+1",
            "rand",
            "xrand01",
            "rand0a",
            "r\u{e4}nd01",
        ] {
            assert_eq!(random_index(n), None, "{n}");
        }
    }

    #[test]
    fn safe_names() {
        for n in ["rock", "fc_x", "a.b", "noise_hq"] {
            assert!(is_safe_name(n), "{n}");
        }
        for n in ["", "..", "../x", "a/b", "a\\b", "C:x", "x..y", "a\0b"] {
            assert!(!is_safe_name(n), "{n:?}");
        }
    }

    #[test]
    fn fit_keeps_aspect() {
        assert_eq!(fit_within([8192, 16], 4096), [4096, 8]);
        assert_eq!(fit_within([8192, 8192], 4096), [4096, 4096]);
        assert_eq!(fit_within([5000, 8000], 4096), [2560, 4096]);
        assert_eq!(fit_within([8000, 1], 4096), [4096, 1]);
    }

    #[test]
    fn resolver_and_picks_are_shareable() {
        fn check<T: Clone + Send + Sync>() {}
        check::<TextureResolver>();
        check::<PresetPicks>();
    }

    /// CRC-32 (IEEE), for hand-built PNG chunks.
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = !0u32;
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    #[test]
    fn crc32_reference() {
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
    }

    #[test]
    fn oversized_header_rejected_before_decoding() {
        // Only a 10000x10000 RGBA8 header and an end chunk, with no pixel
        // data at all: the header is the only thing that can reject it.
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut chunk = |kind: &[u8], data: &[u8]| {
            png.extend_from_slice(&(data.len() as u32).to_be_bytes());
            let mut body = kind.to_vec();
            body.extend_from_slice(data);
            png.extend_from_slice(&body);
            png.extend_from_slice(&crc32(&body).to_be_bytes());
        };
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&10_000u32.to_be_bytes());
        ihdr.extend_from_slice(&10_000u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        chunk(b"IHDR", &ihdr);
        chunk(b"IEND", &[]);

        let dir = std::env::temp_dir().join(format!(
            "ms-render-header-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("big.png");
        std::fs::write(&path, &png).unwrap();
        let err = try_load(&path).unwrap_err();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            err.starts_with("cannot decode") && err.to_lowercase().contains("limit"),
            "{err}"
        );
    }
}
