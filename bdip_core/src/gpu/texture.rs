use crate::Rgba16Image;
use bytemuck::{Pod, Zeroable};
use half::f16;
use wgpu::{
    Device, Extent3d, Origin3d, Queue, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PixelF16 {
    pub r: f16,
    pub g: f16,
    pub b: f16,
    pub a: f16,
}

pub fn upload_texture(device: &Device, queue: &Queue, img: &Rgba16Image) -> wgpu::Texture {
    let (width, height) = img.dimensions();

    // Convert Rgba16 to f16 arrays.
    let mut float_data: Vec<PixelF16> = Vec::with_capacity((width * height) as usize);
    for pixel in img.pixels() {
        float_data.push(PixelF16 {
            r: f16::from_f32(pixel[0] as f32 / 65535.0),
            g: f16::from_f32(pixel[1] as f32 / 65535.0),
            b: f16::from_f32(pixel[2] as f32 / 65535.0),
            a: f16::from_f32(pixel[3] as f32 / 65535.0),
        });
    }

    let texture_size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&TextureDescriptor {
        label: Some("upload_texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::COPY_SRC
            | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(&float_data),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8), // 4 channels * 2 bytes = 8 bytes per pixel
            rows_per_image: Some(height),
        },
        texture_size,
    );

    texture
}

/// Downloads the output of `Renderer::present` from a tightly packed storage
/// buffer into a CPU-side `Rgba16Image`. No per-pixel decoding is performed —
/// the buffer bytes are cast directly to `&[u16]` and handed to
/// `Rgba16Image::from_raw`. This eliminates the per-pixel CPU loop that was
/// present in the previous texture-based readback path.
pub fn download_presentation_buffer(
    device: &Device,
    queue: &Queue,
    src_buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
) -> Result<Rgba16Image, crate::error::BdipError> {
    // Tightly packed: 4 channels × 2 bytes = 8 bytes per pixel, no row padding.
    let buffer_size = (width * height * 8) as wgpu::BufferAddress;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("presentation_staging_buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(src_buffer, 0, &staging_buffer, 0, buffer_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    if rx.recv().unwrap().is_err() {
        return Err(crate::error::BdipError::Gpu(
            "Failed to map presentation buffer for reading".into(),
        ));
    }

    let data = buffer_slice.get_mapped_range();
    // Cast raw bytes to u16. The buffer layout (R|G packed into the first u32,
    // B|A into the second) produces interleaved [R, G, B, A, R, G, B, A, ...]
    // as u16 values on little-endian hardware — exactly what Rgba16Image expects.
    let pixel_vec: Vec<u16> = bytemuck::cast_slice::<u8, u16>(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    Rgba16Image::from_raw(width, height, pixel_vec)
        .ok_or_else(|| crate::error::BdipError::Gpu("Presentation buffer size mismatch".into()))
}
