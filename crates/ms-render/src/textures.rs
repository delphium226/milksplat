//! Finding user texture files (spec §4, §6.1).
//!
//! A preset names a texture by its file name without extension (the shader
//! prefix already stripped). The engine builds one [`TextureResolver`] per
//! preset load, with the roots in lookup order: the preset's own folder,
//! the user `Presets\textures\`, then the bundled `textures\`.
//!
//! Untrusted input is handled without panicking:
//! - names that could leave a root (path separators, `..`, a drive colon)
//!   are rejected;
//! - files over [`MAX_FILE_BYTES`] are rejected unread;
//! - decoding is limited to [`MAX_DECODE_DIM`] per side;
//! - images over [`MAX_TEXTURE_DIM`] per side are downscaled to fit,
//!   keeping their aspect ratio.
//!
//! A file that fails any of these is logged and the next candidate is tried.

use std::collections::HashMap;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use image::RgbaImage;
use ms_expr::Rng;

use crate::feedback::Renderer;
use crate::gpu::Gpu;

/// Largest file read, in bytes (64 MiB).
pub const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// Largest texture side after loading; bigger images are downscaled.
pub const MAX_TEXTURE_DIM: u32 = 4096;

/// Largest image side the decoder accepts (protects against small files
/// that decode to huge images).
pub const MAX_DECODE_DIM: u32 = 16384;

/// Extensions tried, in order, in each root.
const EXTENSIONS: [&str; 2] = ["png", "jpg"];

/// Looks up user textures by name across a list of root folders.
///
/// `rand00`..`rand15` are random textures (see
/// [`TextureResolver::resolve_with`]). The choice for each random name is
/// remembered by the resolver, so the same name resolves to the same file
/// for as long as this resolver lives (one preset load), even when the
/// shaders bind it more than once (`fw_rand03` and `pc_rand03`).
#[derive(Debug, Default)]
pub struct TextureResolver {
    /// Folders searched in order.
    pub roots: Vec<PathBuf>,
    /// Random name (`rand03`) to the file picked for it.
    picks: Mutex<HashMap<String, PathBuf>>,
}

impl TextureResolver {
    pub fn new(roots: Vec<PathBuf>) -> Self {
        TextureResolver {
            roots,
            picks: Mutex::default(),
        }
    }

    /// Loads `<root>/<name>.png`, then `<root>/<name>.jpg`, for each root in
    /// order, returning the first that loads. Random names (`rand00`..
    /// `rand15`) and unsafe names give `None`; use
    /// [`TextureResolver::resolve_with`] for random names.
    pub fn resolve(&self, name_without_prefix: &str) -> Option<RgbaImage> {
        let name = name_without_prefix;
        if !is_safe_name(name) {
            tracing::warn!("texture name {name:?} rejected: it is not a plain file name");
            return None;
        }
        if is_random_name(name) {
            tracing::debug!("texture {name}: random textures need resolve_with");
            return None;
        }
        self.roots
            .iter()
            .flat_map(|root| EXTENSIONS.map(|ext| root.join(format!("{name}.{ext}"))))
            .filter(|path| path.is_file())
            .find_map(|path| load(&path))
    }

    /// Like [`TextureResolver::resolve`], but `rand00`..`rand15` pick a file
    /// uniformly at random (with `rng`, the preset's generator) from
    /// `<root>/random/*.png|jpg` of the first root whose `random` folder has
    /// any. The pick is remembered per name for this resolver. If the picked
    /// file does not load, the result is `None`.
    pub fn resolve_with(&self, name_without_prefix: &str, rng: &mut Rng) -> Option<RgbaImage> {
        let name = name_without_prefix;
        if !is_safe_name(name) || !is_random_name(name) {
            return self.resolve(name);
        }
        let mut picks = match self.picks.lock() {
            Ok(picks) => picks,
            Err(poisoned) => poisoned.into_inner(),
        };
        let path = match picks.get(name) {
            Some(path) => path.clone(),
            None => {
                let Some(candidates) = self.random_candidates() else {
                    tracing::warn!("texture {name}: no images in any random folder");
                    return None;
                };
                let i =
                    ((rng.next_f64() * candidates.len() as f64) as usize).min(candidates.len() - 1);
                let path = candidates[i].clone();
                tracing::debug!("texture {name}: picked {}", path.display());
                picks.insert(name.to_string(), path.clone());
                path
            }
        };
        drop(picks);
        load(&path)
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

/// `rand00`..`rand15`.
fn is_random_name(name: &str) -> bool {
    name.strip_prefix("rand")
        .filter(|n| n.len() == 2 && n.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|n| n.parse::<u8>().ok())
        .is_some_and(|n| n <= 15)
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
    reader.limits(limits);
    let img = reader.decode().map_err(|e| format!("cannot decode: {e}"))?;

    let (w, h) = (img.width(), img.height());
    let img = if w > MAX_TEXTURE_DIM || h > MAX_TEXTURE_DIM {
        tracing::info!(
            "texture {}: {w}x{h} downscaled to fit {MAX_TEXTURE_DIM}x{MAX_TEXTURE_DIM}",
            path.display()
        );
        img.resize(
            MAX_TEXTURE_DIM,
            MAX_TEXTURE_DIM,
            image::imageops::FilterType::Triangle,
        )
    } else {
        img
    };
    Ok(img.into_rgba8())
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
        for n in ["rand00", "rand09", "rand15"] {
            assert!(is_random_name(n), "{n}");
        }
        for n in [
            "rand16", "rand3", "rand003", "rand-1", "rand+1", "rand", "xrand01", "rand0a",
        ] {
            assert!(!is_random_name(n), "{n}");
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
}
