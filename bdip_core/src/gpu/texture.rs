use bytemuck::{Pod, Zeroable};
use half::f16;
use image::RgbaImage;
use wgpu::{
    Device, Extent3d, Origin3d, Queue, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PixelF16 {
    pub r: f16,
    pub g: f16,
    pub b: f16,
    pub a: f16,
}

pub fn upload_texture(device: &Device, queue: &Queue, img: &RgbaImage) -> wgpu::Texture {
    let (width, height) = img.dimensions();

    // Convert Rgba8 to f16 arrays.
    let mut float_data: Vec<PixelF16> = Vec::with_capacity((width * height) as usize);
    for pixel in img.pixels() {
        float_data.push(PixelF16 {
            r: f16::from_f32(pixel[0] as f32 / 255.0),
            g: f16::from_f32(pixel[1] as f32 / 255.0),
            b: f16::from_f32(pixel[2] as f32 / 255.0),
            a: f16::from_f32(pixel[3] as f32 / 255.0),
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

pub(crate) fn calculate_padded_bytes_per_row(img_width: u32, bytes_per_pixel: u32) -> u32 {
    let unpadded_bytes_per_row = img_width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    (unpadded_bytes_per_row + align - 1) & !(align - 1)
}

pub(crate) fn clamp_f32_to_u8(val: f32) -> u8 {
    (val.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn download_texture(
    device: &Device,
    queue: &Queue,
    texture: &wgpu::Texture,
    img_width: u32,
    img_height: u32,
) -> Result<RgbaImage, crate::error::BdipError> {
    // Reading from WGPU requires creating a staging buffer.

    // WebGPU requires bytes_per_row to be a multiple of 256.
    let byte_size = 8; // Rgba16Float -> 8 bytes per pixel
    let padded_bytes_per_row = calculate_padded_bytes_per_row(img_width, byte_size);

    let buffer_size = (padded_bytes_per_row * img_height) as wgpu::BufferAddress;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("download_staging_buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Encode a command to copy texture to staging buffer
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &staging_buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(img_height),
            },
        },
        Extent3d {
            width: img_width,
            height: img_height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    // Map the buffer securely via `pollster` blocking
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    // Poll the device in a blocking way
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    if rx.recv().unwrap().is_err() {
        return Err(crate::error::BdipError::Gpu(
            "Failed to map buffer for reading".into(),
        ));
    }

    let data = buffer_slice.get_mapped_range();

    // Decode bytes back to floats, then clamp to 0-255 u8.
    let mut out_img_buf = image::ImageBuffer::new(img_width, img_height);

    for y in 0..img_height {
        for x in 0..img_width {
            let offset = (y * padded_bytes_per_row + x * byte_size) as usize;
            // Decode 4 f16s
            let r_f16 = f16::from_bits(u16::from_ne_bytes([data[offset], data[offset + 1]]));
            let g_f16 = f16::from_bits(u16::from_ne_bytes([data[offset + 2], data[offset + 3]]));
            let b_f16 = f16::from_bits(u16::from_ne_bytes([data[offset + 4], data[offset + 5]]));
            let a_f16 = f16::from_bits(u16::from_ne_bytes([data[offset + 6], data[offset + 7]]));

            out_img_buf.put_pixel(
                x,
                y,
                image::Rgba([
                    clamp_f32_to_u8(r_f16.to_f32()),
                    clamp_f32_to_u8(g_f16.to_f32()),
                    clamp_f32_to_u8(b_f16.to_f32()),
                    clamp_f32_to_u8(a_f16.to_f32()),
                ]),
            );
        }
    }

    drop(data);
    staging_buffer.unmap();

    Ok(out_img_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padding_rounds_up_to_256() {
        // 10 pixels * 8 bytes = 80 bytes. Next multiple of 256 is 256.
        assert_eq!(calculate_padded_bytes_per_row(10, 8), 256);
    }

    #[test]
    fn test_padding_rounds_up_to_512() {
        // 40 pixels * 8 bytes = 320 bytes. Next multiple of 256 is 512.
        assert_eq!(calculate_padded_bytes_per_row(40, 8), 512);
    }

    #[test]
    fn test_padding_remains_at_256_when_exactly_aligned() {
        // 32 pixels * 8 bytes = 256 bytes. Starts exactly aligned.
        assert_eq!(calculate_padded_bytes_per_row(32, 8), 256);
    }

    #[test]
    fn test_clamp_f32_to_u8_lower_bound() {
        assert_eq!(clamp_f32_to_u8(0.0), 0);
    }

    #[test]
    fn test_clamp_f32_to_u8_upper_bound() {
        assert_eq!(clamp_f32_to_u8(1.0), 255);
    }

    #[test]
    fn test_clamp_f32_to_u8_midpoint() {
        assert_eq!(clamp_f32_to_u8(0.5), 128); // 127.5 rounds up
    }

    #[test]
    fn test_clamp_f32_to_u8_underflow_clips_to_zero() {
        assert_eq!(clamp_f32_to_u8(-0.5), 0);
    }

    #[test]
    fn test_clamp_f32_to_u8_overflow_clips_to_max() {
        assert_eq!(clamp_f32_to_u8(1.5), 255);
    }
}
