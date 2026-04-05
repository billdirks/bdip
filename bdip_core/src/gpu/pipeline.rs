use crate::gpu::engine::GpuEngine;
use wgpu::{util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, ComputePipeline, ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderStages, StorageTextureAccess, TextureFormat, TextureViewDescriptor, TextureViewDimension};

pub struct Renderer {
    pipeline: ComputePipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsUniform {
    brightness_offset: f32,
    _padding: [f32; 3], // WebGPU uniforms require 16-byte alignment
}

impl Renderer {
    pub fn new(engine: &GpuEngine) -> Self {
        let shader = engine.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Brightness Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Group 0: Textures
        let texture_bind_group_layout = engine.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
            entries: &[
                // src_texture
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // dst_texture (storage)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Group 1: Parameters
        let params_bind_group_layout = engine.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Params Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
        });

        let pipeline_layout = engine.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&texture_bind_group_layout), Some(&params_bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = engine.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Brightness Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            texture_bind_group_layout,
            params_bind_group_layout,
        }
    }

    pub fn apply_brightness(
        &self,
        engine: &GpuEngine,
        src_texture: &wgpu::Texture,
        brightness_val: f32,
    ) -> wgpu::Texture {
        let (width, height, depth) = (src_texture.width(), src_texture.height(), src_texture.depth_or_array_layers());
        
        let dst_texture = engine.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dst_texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: depth },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let src_view = src_texture.create_view(&TextureViewDescriptor::default());
        let dst_view = dst_texture.create_view(&TextureViewDescriptor::default());

        let texture_bind_group = engine.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&src_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&dst_view) },
            ],
        });

        let params = ParamsUniform {
            brightness_offset: brightness_val,
            _padding: [0.0; 3],
        };

        let params_buffer = engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Uniform Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let params_bind_group = engine.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Params Bind Group"),
            layout: &self.params_bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &texture_bind_group, &[]);
            cpass.set_bind_group(1, &params_bind_group, &[]);
            cpass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }

        engine.queue.submit(Some(encoder.finish()));

        dst_texture
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::texture::{upload_texture, download_texture};
    
    #[test]
    fn test_brightness_shader_positive() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);
        
        // 50% gray image
        let mut img = image::RgbaImage::new(2, 2);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([127, 127, 127, 255]);
        }
        
        let initial_texture = upload_texture(&engine.device, &engine.queue, &img);
        
        // Boost brightness by +0.5 (should hit max 255 because 127 + 127 = 254/255)
        let out_texture = renderer.apply_brightness(&engine, &initial_texture, 0.5);
        let out_img = download_texture(&engine.device, &engine.queue, &out_texture, 2, 2).unwrap();
        
        for pixel in out_img.pixels() {
            assert!(pixel[0] == 255);
            assert!(pixel[1] == 255);
            assert!(pixel[2] == 255);
            assert!(pixel[3] == 255); // Alpha untouched
        }
    }

    #[test]
    fn test_brightness_shader_negative() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);
        
        let mut img = image::RgbaImage::new(2, 2);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([127, 127, 127, 255]);
        }
        
        let initial_texture = upload_texture(&engine.device, &engine.queue, &img);
        let out_texture = renderer.apply_brightness(&engine, &initial_texture, -0.6); // heavily negative
        let out_img = download_texture(&engine.device, &engine.queue, &out_texture, 2, 2).unwrap();
        
        for pixel in out_img.pixels() {
            assert!(pixel[0] == 0);
            assert!(pixel[1] == 0);
            assert!(pixel[2] == 0);
            assert!(pixel[3] == 255); // Alpha untouched
        }
    }

    #[test]
    fn test_brightness_shader_zero() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);
        
        let mut img = image::RgbaImage::new(2, 2);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([42, 100, 200, 255]);
        }
        
        let initial_texture = upload_texture(&engine.device, &engine.queue, &img);
        let out_texture = renderer.apply_brightness(&engine, &initial_texture, 0.0);
        let out_img = download_texture(&engine.device, &engine.queue, &out_texture, 2, 2).unwrap();
        
        for pixel in out_img.pixels() {
            assert!(pixel[0] == 42); // 42 or 43 due to f16 back and forth precision
            // Allow precision variance of 1
            assert!((pixel[1] as i32 - 100).abs() <= 1);
            assert!((pixel[2] as i32 - 200).abs() <= 1);
            assert!(pixel[3] == 255);
        }
    }

    #[test]
    fn test_shader_chaining() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);
        
        let mut img = image::RgbaImage::new(2, 2);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([200, 200, 200, 255]);
        }
        
        let initial_texture = upload_texture(&engine.device, &engine.queue, &img);
        
        // Pipelined transformations strictly in video memory.
        let intermediate_texture = renderer.apply_brightness(&engine, &initial_texture, -0.2); 
        let final_texture = renderer.apply_brightness(&engine, &intermediate_texture, 0.5); 
        
        let out_img = download_texture(&engine.device, &engine.queue, &final_texture, 2, 2).unwrap();
        
        for pixel in out_img.pixels() {
            assert!(pixel[0] == 255);
            assert!(pixel[1] == 255);
            assert!(pixel[2] == 255);
            assert!(pixel[3] == 255); // Alpha untouched
        }
    }
}
