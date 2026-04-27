pub mod blue_noise;
pub mod halftone_dots;

use crate::error::BdipError;
use crate::gpu::shaders::AuxTextureDimension;
use std::collections::HashMap;
use wgpu::TextureFormat;

pub enum AuxAssetFormat {
    Png,
    CubeRaw { size: u32 },
}

/// A bundled auxiliary texture asset. Collected by `inventory` at link time.
pub struct AuxAssetRegistration {
    pub name: &'static str,
    pub raw_bytes: &'static [u8],
    pub format: AuxAssetFormat,
    pub dimension: AuxTextureDimension,
}

inventory::collect!(AuxAssetRegistration);

pub fn find_asset_by_name(name: &str) -> Option<&'static AuxAssetRegistration> {
    inventory::iter::<AuxAssetRegistration>
        .into_iter()
        .find(|a| a.name == name)
}

pub struct AuxTextureCache {
    gpu_textures: HashMap<&'static str, wgpu::Texture>,
}

impl Default for AuxTextureCache {
    fn default() -> Self {
        Self::new()
    }
}

impl AuxTextureCache {
    pub fn new() -> Self {
        Self {
            gpu_textures: HashMap::new(),
        }
    }

    pub fn get_or_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
    ) -> Result<&wgpu::Texture, BdipError> {
        if !self.gpu_textures.contains_key(name) {
            let asset = find_asset_by_name(name)
                .ok_or_else(|| BdipError::MissingAuxTexture(name.to_string()))?;
            let texture = decode_and_upload(device, queue, asset);
            self.gpu_textures.insert(asset.name, texture);
        }
        Ok(self.gpu_textures.get(name).unwrap())
    }

    pub fn get(&self, name: &str) -> Option<&wgpu::Texture> {
        self.gpu_textures.get(name)
    }
}

fn decode_and_upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    asset: &AuxAssetRegistration,
) -> wgpu::Texture {
    match asset.format {
        AuxAssetFormat::Png => decode_and_upload_png(device, queue, asset),
        AuxAssetFormat::CubeRaw { size } => decode_and_upload_cube_raw(device, queue, asset, size),
    }
}

fn decode_and_upload_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    asset: &AuxAssetRegistration,
) -> wgpu::Texture {
    let img = image::load_from_memory(asset.raw_bytes)
        .expect("Failed to decode PNG aux texture")
        .to_rgba8();
    let (width, height) = img.dimensions();

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(asset.name),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        texture.as_image_copy(),
        &img,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    texture
}

fn decode_and_upload_cube_raw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    asset: &AuxAssetRegistration,
    size: u32,
) -> wgpu::Texture {
    let floats: &[f32] = bytemuck::cast_slice(asset.raw_bytes);
    let pixel_count = (size * size * size) as usize;
    assert_eq!(
        floats.len(),
        pixel_count * 3,
        "CubeRaw: expected {} f32 RGB triples, got {} floats",
        pixel_count,
        floats.len()
    );

    let mut rgba_f16 = Vec::with_capacity(pixel_count * 8);
    for chunk in floats.chunks_exact(3) {
        for &val in chunk {
            rgba_f16.extend_from_slice(&f32_to_f16(val).to_le_bytes());
        }
        rgba_f16.extend_from_slice(&f32_to_f16(1.0).to_le_bytes());
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(asset.name),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        texture.as_image_copy(),
        &rgba_f16,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 8),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
    );

    texture
}

fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = bits & 0x007F_FFFF;

    if exponent > 15 {
        return (sign | 0x7C00) as u16;
    }
    if exponent < -14 {
        return sign as u16;
    }

    let e16 = ((exponent + 15) as u32) << 10;
    let m16 = mantissa >> 13;
    (sign | e16 | m16) as u16
}

#[cfg(test)]
mod test_assets {
    use super::AuxAssetFormat;
    use super::*;

    inventory::submit!(AuxAssetRegistration {
        name: "__test_2x2_white",
        raw_bytes: include_bytes!("test_2x2_white.png"),
        format: AuxAssetFormat::Png,
        dimension: AuxTextureDimension::D2,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;

    #[test]
    fn test_aux_cache_get_or_upload_returns_texture() {
        let engine = GpuEngine::new().unwrap();
        let mut cache = AuxTextureCache::new();
        let result = cache.get_or_upload(&engine.device, &engine.queue, "__test_2x2_white");
        assert!(
            result.is_ok(),
            "get_or_upload should succeed for registered asset"
        );
        let tex = result.unwrap();
        assert_eq!(tex.width(), 2);
        assert_eq!(tex.height(), 2);
    }

    #[test]
    fn test_aux_cache_second_call_returns_same_texture() {
        let engine = GpuEngine::new().unwrap();
        let mut cache = AuxTextureCache::new();
        cache
            .get_or_upload(&engine.device, &engine.queue, "__test_2x2_white")
            .unwrap();
        let tex1 = cache.get("__test_2x2_white").unwrap() as *const wgpu::Texture;
        cache
            .get_or_upload(&engine.device, &engine.queue, "__test_2x2_white")
            .unwrap();
        let tex2 = cache.get("__test_2x2_white").unwrap() as *const wgpu::Texture;
        assert!(
            std::ptr::eq(tex1, tex2),
            "second get_or_upload must return the same cached texture"
        );
    }

    #[test]
    fn test_aux_cache_missing_name_returns_error() {
        let engine = GpuEngine::new().unwrap();
        let mut cache = AuxTextureCache::new();
        let result = cache.get_or_upload(&engine.device, &engine.queue, "nonexistent_asset");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, BdipError::MissingAuxTexture(_)),
            "expected MissingAuxTexture, got {err:?}"
        );
    }
}
