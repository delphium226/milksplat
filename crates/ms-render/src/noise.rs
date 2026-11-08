//! Noise textures (spec §6.1): generated from a fixed seed, so every run and
//! every renderer gets the same bytes.
//!
//! Each texture is `Rgba8Unorm` with every channel of every texel an
//! independent uniform byte. The bytes come from one splitmix64 stream
//! seeded with [`NOISE_SEED`], consumed in the order `lq`, `mq`, `hq`,
//! `vol_lq`, `vol_hq` (eight bytes per step, little-endian), texels
//! row-major and slice by slice. Shaders pick the addressing and filtering
//! with the prelude's samplers (`s_fw`, `s_pc`, ...).

use crate::gpu::Gpu;

/// Seed of the noise stream ("MSPL").
pub const NOISE_SEED: u64 = 0x4D53_504C;

/// The five noise textures, bound at group 0 bindings 5-9 as
/// `sampler_noise_lq`/`_mq`/`_hq` and `sampler_noisevol_lq`/`_hq`.
pub struct NoiseSet {
    /// 32x32.
    pub lq: wgpu::Texture,
    /// 64x64.
    pub mq: wgpu::Texture,
    /// 256x256.
    pub hq: wgpu::Texture,
    /// 32x32x32, 3D.
    pub vol_lq: wgpu::Texture,
    /// 64x64x64, 3D.
    pub vol_hq: wgpu::Texture,
}

/// The splitmix64 generator (Steele, Lea and Flood), used only for noise.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `len` bytes (`len` is a multiple of 4 here, texels being 4 bytes).
    fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len.next_multiple_of(8));
        while out.len() < len {
            out.extend_from_slice(&self.next().to_le_bytes());
        }
        out.truncate(len);
        out
    }
}

/// Generates and uploads the noise textures (deterministic: the same bytes
/// on every call).
pub fn noise_textures(gpu: &Gpu) -> NoiseSet {
    use wgpu::TextureDimension::{D2, D3};
    let mut rng = SplitMix64(NOISE_SEED);
    let mut make = |label, size: u32, dimension| {
        let depth = if dimension == D3 { size } else { 1 };
        let data = rng.bytes(size as usize * size as usize * depth as usize * 4);
        upload(gpu, label, [size, size, depth], dimension, &data)
    };
    NoiseSet {
        lq: make("noise_lq", 32, D2),
        mq: make("noise_mq", 64, D2),
        hq: make("noise_hq", 256, D2),
        vol_lq: make("noisevol_lq", 32, D3),
        vol_hq: make("noisevol_hq", 64, D3),
    }
}

impl NoiseSet {
    /// Views in binding order: `[lq, mq, hq]` and `[vol_lq, vol_hq]`.
    pub(crate) fn views(&self) -> ([wgpu::TextureView; 3], [wgpu::TextureView; 2]) {
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        (
            [view(&self.lq), view(&self.mq), view(&self.hq)],
            [view(&self.vol_lq), view(&self.vol_hq)],
        )
    }
}

fn upload(
    gpu: &Gpu,
    label: &str,
    [width, height, depth]: [u32; 3],
    dimension: wgpu::TextureDimension,
    data: &[u8],
) -> wgpu::Texture {
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        tex.as_image_copy(),
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        tex.size(),
    );
    tex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_reference_values() {
        // splitmix64 seeded with 0 (reference implementation output).
        let mut r = SplitMix64(0);
        assert_eq!(r.next(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn bytes_are_little_endian_steps() {
        let mut a = SplitMix64(NOISE_SEED);
        let first = a.next().to_le_bytes();
        let mut b = SplitMix64(NOISE_SEED);
        let bytes = b.bytes(12);
        assert_eq!(bytes[..8], first);
        assert_eq!(bytes.len(), 12);
    }
}
