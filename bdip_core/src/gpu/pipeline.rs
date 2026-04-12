use crate::gpu::engine::GpuEngine;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, ComputePipeline, ComputePipelineDescriptor,
    PipelineLayoutDescriptor, ShaderStages, StorageTextureAccess, TextureFormat,
    TextureViewDescriptor, TextureViewDimension, util::DeviceExt,
};

pub struct Renderer {
    pipeline: ComputePipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    ingest_pipeline: ComputePipeline,
    ingest_bind_group_layout: wgpu::BindGroupLayout,
    present_pipeline: ComputePipeline,
    present_texture_bind_group_layout: wgpu::BindGroupLayout,
    present_params_bind_group_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsUniform {
    brightness_offset: f32,
    _padding: [f32; 3], // WebGPU uniforms require 16-byte alignment
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PresentParams {
    width: u32,
    _padding: [u32; 3], // WebGPU uniforms require 16-byte alignment
}

/// Shared bind group layout for passes that bind one source texture and one
/// destination storage texture (no uniforms). Used by the Ingest pass.
fn make_texture_only_bind_group_layout(
    device: &wgpu::Device,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
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
    })
}

impl Renderer {
    pub fn new(engine: &GpuEngine) -> Self {
        // --- Brightness pipeline ---
        let shader = engine
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Brightness Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            });

        let texture_bind_group_layout =
            make_texture_only_bind_group_layout(&engine.device, "Brightness Texture BGL");

        let params_bind_group_layout =
            engine
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Params Bind Group Layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let pipeline_layout = engine
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&texture_bind_group_layout),
                    Some(&params_bind_group_layout),
                ],
                immediate_size: 0,
            });

        let pipeline = engine
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Brightness Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Ingest pipeline (sRGB → linear) ---
        let ingest_shader = engine
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Ingest Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("ingest.wgsl").into()),
            });

        let ingest_bind_group_layout =
            make_texture_only_bind_group_layout(&engine.device, "Ingest Texture BGL");

        let ingest_pipeline_layout =
            engine
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("Ingest Pipeline Layout"),
                    bind_group_layouts: &[Some(&ingest_bind_group_layout)],
                    immediate_size: 0,
                });

        let ingest_pipeline = engine
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Ingest Pipeline"),
                layout: Some(&ingest_pipeline_layout),
                module: &ingest_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Presentation pipeline (linear → sRGB, writes to storage buffer) ---
        let present_shader = engine
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Presentation Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("presentation.wgsl").into()),
            });

        // Group 0: source texture (binding 0) + destination storage buffer (binding 1)
        let present_texture_bind_group_layout =
            engine
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Presentation Texture BGL"),
                    entries: &[
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
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Group 1: width uniform
        let present_params_bind_group_layout =
            engine
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Presentation Params BGL"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let present_pipeline_layout =
            engine
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("Presentation Pipeline Layout"),
                    bind_group_layouts: &[
                        Some(&present_texture_bind_group_layout),
                        Some(&present_params_bind_group_layout),
                    ],
                    immediate_size: 0,
                });

        let present_pipeline = engine
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Presentation Pipeline"),
                layout: Some(&present_pipeline_layout),
                module: &present_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            pipeline,
            texture_bind_group_layout,
            params_bind_group_layout,
            ingest_pipeline,
            ingest_bind_group_layout,
            present_pipeline,
            present_texture_bind_group_layout,
            present_params_bind_group_layout,
        }
    }

    /// Converts sRGB-encoded values in `src_texture` to linear light, returning
    /// a new `Rgba16Float` texture. Must be the first pass after `upload_texture`.
    pub fn ingest(&self, engine: &GpuEngine, src_texture: &wgpu::Texture) -> wgpu::Texture {
        self.run_color_space_pass(
            engine,
            src_texture,
            &self.ingest_pipeline,
            &self.ingest_bind_group_layout,
            "ingest_dst_texture",
        )
    }

    /// Converts linear-light values in `src_texture` to sRGB-encoded u16 values
    /// packed into a tightly packed storage buffer (2 u32s per pixel, no row padding).
    /// Must be the last pass before `download_presentation_buffer`.
    pub fn present(&self, engine: &GpuEngine, src_texture: &wgpu::Texture) -> wgpu::Buffer {
        let (width, height) = (src_texture.width(), src_texture.height());

        // Tightly packed output: 4 channels × 2 bytes = 8 bytes per pixel, no row padding.
        let dst_buffer = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present_dst_buffer"),
            size: (width * height * 8) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = PresentParams {
            width,
            _padding: [0; 3],
        };
        let params_buffer = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("present_params_buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let src_view = src_texture.create_view(&TextureViewDescriptor::default());

        let texture_bind_group = engine.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Present Texture Bind Group"),
            layout: &self.present_texture_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&src_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: dst_buffer.as_entire_binding(),
                },
            ],
        });

        let params_bind_group = engine.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Present Params Bind Group"),
            layout: &self.present_params_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.present_pipeline);
            cpass.set_bind_group(0, &texture_bind_group, &[]);
            cpass.set_bind_group(1, &params_bind_group, &[]);
            cpass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }

        engine.queue.submit(Some(encoder.finish()));

        dst_buffer
    }

    /// Dispatches a single-pass color-space conversion shader that reads a source
    /// texture and writes to a destination storage texture (no uniforms). Used by
    /// `ingest`.
    fn run_color_space_pass(
        &self,
        engine: &GpuEngine,
        src_texture: &wgpu::Texture,
        pipeline: &ComputePipeline,
        bind_group_layout: &wgpu::BindGroupLayout,
        dst_label: &str,
    ) -> wgpu::Texture {
        let (width, height, depth) = (
            src_texture.width(),
            src_texture.height(),
            src_texture.depth_or_array_layers(),
        );

        let dst_texture = engine.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(dst_label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let src_view = src_texture.create_view(&TextureViewDescriptor::default());
        let dst_view = dst_texture.create_view(&TextureViewDescriptor::default());

        let bind_group = engine.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Color Space Pass Bind Group"),
            layout: bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&src_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&dst_view),
                },
            ],
        });

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }

        engine.queue.submit(Some(encoder.finish()));

        dst_texture
    }

    pub fn apply_brightness(
        &self,
        engine: &GpuEngine,
        src_texture: &wgpu::Texture,
        brightness_val: f32,
    ) -> wgpu::Texture {
        let (width, height, depth) = (
            src_texture.width(),
            src_texture.height(),
            src_texture.depth_or_array_layers(),
        );

        let dst_texture = engine.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dst_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let src_view = src_texture.create_view(&TextureViewDescriptor::default());
        let dst_view = dst_texture.create_view(&TextureViewDescriptor::default());

        let texture_bind_group = engine.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&src_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&dst_view),
                },
            ],
        });

        let params = ParamsUniform {
            brightness_offset: brightness_val,
            _padding: [0.0; 3],
        };

        let params_buffer = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Uniform Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let params_bind_group = engine.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Params Bind Group"),
            layout: &self.params_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
    use crate::gpu::texture::{download_presentation_buffer, upload_texture};

    #[test]
    fn test_brightness_shader_positive() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);

        // 50% gray in sRGB (32767/65535 ≈ 0.500 sRGB → ~0.214 linear)
        let mut img = crate::Rgba16Image::new(2, 2);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([32767, 32767, 32767, 65535]);
        }

        let upload = upload_texture(&engine.device, &engine.queue, &img);
        let linear = renderer.ingest(&engine, &upload);
        let brightened = renderer.apply_brightness(&engine, &linear, 0.5);
        let present_buf = renderer.present(&engine, &brightened);
        let out_img =
            download_presentation_buffer(&engine.device, &engine.queue, &present_buf, 2, 2)
                .unwrap();

        // 0.214 + 0.5 = 0.714 linear → sRGB ≈ 0.862 → u16 ≈ 56500
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 56500).abs() <= 64,
                "R: expected ~56500, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 56500).abs() <= 64,
                "G: expected ~56500, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 56500).abs() <= 64,
                "B: expected ~56500, got {}",
                pixel[2]
            );
            assert!(pixel[3] == 65535); // Alpha untouched
        }
    }

    #[test]
    fn test_brightness_shader_negative() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);

        // 50% gray in sRGB → ~0.214 linear; 0.214 - 0.6 = -0.386 → clamped to 0
        let mut img = crate::Rgba16Image::new(2, 2);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([32767, 32767, 32767, 65535]);
        }

        let upload = upload_texture(&engine.device, &engine.queue, &img);
        let linear = renderer.ingest(&engine, &upload);
        let darkened = renderer.apply_brightness(&engine, &linear, -0.6);
        let present_buf = renderer.present(&engine, &darkened);
        let out_img =
            download_presentation_buffer(&engine.device, &engine.queue, &present_buf, 2, 2)
                .unwrap();

        for pixel in out_img.pixels() {
            assert!(pixel[0] == 0);
            assert!(pixel[1] == 0);
            assert!(pixel[2] == 0);
            assert!(pixel[3] == 65535); // Alpha untouched
        }
    }

    #[test]
    fn test_brightness_shader_zero() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(2, 2);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([10794, 25700, 51400, 65535]);
        }

        let upload = upload_texture(&engine.device, &engine.queue, &img);
        let linear = renderer.ingest(&engine, &upload);
        let unchanged = renderer.apply_brightness(&engine, &linear, 0.0);
        let present_buf = renderer.present(&engine, &unchanged);
        let out_img =
            download_presentation_buffer(&engine.device, &engine.queue, &present_buf, 2, 2)
                .unwrap();

        // sRGB → linear → sRGB is a mathematical identity; differences are f16 rounding.
        for pixel in out_img.pixels() {
            assert!((pixel[0] as i32 - 10794).abs() <= 64);
            assert!((pixel[1] as i32 - 25700).abs() <= 64);
            assert!((pixel[2] as i32 - 51400).abs() <= 64);
            assert!(pixel[3] == 65535);
        }
    }

    #[test]
    fn test_shader_chaining() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);

        // 51400/65535 ≈ 0.784 sRGB → ~0.577 linear
        let mut img = crate::Rgba16Image::new(2, 2);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([51400, 51400, 51400, 65535]);
        }

        let upload = upload_texture(&engine.device, &engine.queue, &img);
        let linear = renderer.ingest(&engine, &upload);

        // 0.577 - 0.2 = 0.377; 0.377 + 0.5 = 0.877 linear → sRGB ≈ 0.944 → u16 ≈ 61880
        let intermediate = renderer.apply_brightness(&engine, &linear, -0.2);
        let final_linear = renderer.apply_brightness(&engine, &intermediate, 0.5);
        let present_buf = renderer.present(&engine, &final_linear);
        let out_img =
            download_presentation_buffer(&engine.device, &engine.queue, &present_buf, 2, 2)
                .unwrap();

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 61880).abs() <= 64,
                "R: expected ~61880, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 61880).abs() <= 64,
                "G: expected ~61880, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 61880).abs() <= 64,
                "B: expected ~61880, got {}",
                pixel[2]
            );
            assert!(pixel[3] == 65535); // Alpha untouched
        }
    }

    #[test]
    fn test_shader_headroom_preservation() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);

        // 32767/65535 ≈ 0.500 sRGB → ~0.214 linear
        let mut img = crate::Rgba16Image::new(2, 2);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([32767, 32767, 32767, 65535]);
        }

        let upload = upload_texture(&engine.device, &engine.queue, &img);
        let linear = renderer.ingest(&engine, &upload);

        // 0.214 + 0.8 = 1.014 (above 1.0, held in headroom); 1.014 - 0.8 = 0.214 linear
        let up = renderer.apply_brightness(&engine, &linear, 0.8);
        let down = renderer.apply_brightness(&engine, &up, -0.8);
        let present_buf = renderer.present(&engine, &down);
        let out_img =
            download_presentation_buffer(&engine.device, &engine.queue, &present_buf, 2, 2)
                .unwrap();

        // linear_to_srgb(0.214) ≈ 0.500 sRGB → u16 ≈ 32767; allow ±64 for f16 precision
        for pixel in out_img.pixels() {
            assert!((pixel[0] as i32 - 32767).abs() <= 64);
            assert!((pixel[1] as i32 - 32767).abs() <= 64);
            assert!((pixel[2] as i32 - 32767).abs() <= 64);
            assert!(pixel[3] == 65535);
        }
    }

    #[test]
    fn test_ingest_present_roundtrip() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);

        // Use a mix of mid-tone values across the sRGB range.
        let mut img = crate::Rgba16Image::new(4, 4);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([16384, 32768, 49152, 65535]);
        }

        let upload = upload_texture(&engine.device, &engine.queue, &img);
        let linear = renderer.ingest(&engine, &upload);
        let present_buf = renderer.present(&engine, &linear);
        let out_img =
            download_presentation_buffer(&engine.device, &engine.queue, &present_buf, 4, 4)
                .unwrap();

        // sRGB → linear → sRGB is mathematically exact; allow ±128 to absorb
        // both directions of f16 + pow rounding.
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 16384).abs() <= 128,
                "R: expected ~16384, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 32768).abs() <= 128,
                "G: expected ~32768, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 49152).abs() <= 128,
                "B: expected ~49152, got {}",
                pixel[2]
            );
            assert!(pixel[3] == 65535);
        }
    }

    #[test]
    fn test_ingest_pure_black_and_white() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);

        // The sRGB transfer function fixes both endpoints exactly: f(0) = 0, f(1) = 1.
        let mut img = crate::Rgba16Image::new(2, 1);
        img.put_pixel(0, 0, image::Rgba([0, 0, 0, 65535]));
        img.put_pixel(1, 0, image::Rgba([65535, 65535, 65535, 65535]));

        let upload = upload_texture(&engine.device, &engine.queue, &img);
        let linear = renderer.ingest(&engine, &upload);
        let present_buf = renderer.present(&engine, &linear);
        let out_img =
            download_presentation_buffer(&engine.device, &engine.queue, &present_buf, 2, 1)
                .unwrap();

        let black = out_img.get_pixel(0, 0);
        let white = out_img.get_pixel(1, 0);

        assert_eq!(black[0], 0, "Black R should be 0");
        assert_eq!(black[1], 0, "Black G should be 0");
        assert_eq!(black[2], 0, "Black B should be 0");
        assert_eq!(black[3], 65535, "Alpha should be untouched");

        // Pure white (1.0 linear) should pass through both conversions cleanly.
        // Allow ±64 for GPU pow() floating-point imprecision near the 1.0 boundary.
        assert!(
            (white[0] as i32 - 65535).abs() <= 64,
            "White R: expected ~65535, got {}",
            white[0]
        );
        assert!(
            (white[1] as i32 - 65535).abs() <= 64,
            "White G: expected ~65535, got {}",
            white[1]
        );
        assert!(
            (white[2] as i32 - 65535).abs() <= 64,
            "White B: expected ~65535, got {}",
            white[2]
        );
        assert_eq!(white[3], 65535, "Alpha should be untouched");
    }

    #[test]
    fn test_presentation_buffer_layout() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);

        // 3×2 image: row 0 = black, row 1 = white.
        // If row stride or pixel packing is wrong, rows or channels will be swapped.
        let mut img = crate::Rgba16Image::new(3, 2);
        for x in 0..3 {
            img.put_pixel(x, 0, image::Rgba([0, 0, 0, 65535]));
            img.put_pixel(x, 1, image::Rgba([65535, 65535, 65535, 65535]));
        }

        let upload = upload_texture(&engine.device, &engine.queue, &img);
        let linear = renderer.ingest(&engine, &upload);
        let present_buf = renderer.present(&engine, &linear);
        let out_img =
            download_presentation_buffer(&engine.device, &engine.queue, &present_buf, 3, 2)
                .unwrap();

        // Row 0: black → sRGB(0.0) = 0.0 → u16 = 0
        for x in 0..3 {
            let px = out_img.get_pixel(x, 0);
            assert_eq!(px[0], 0, "Row 0, col {x}: R should be 0 (black)");
            assert_eq!(px[1], 0, "Row 0, col {x}: G should be 0 (black)");
            assert_eq!(px[2], 0, "Row 0, col {x}: B should be 0 (black)");
            assert_eq!(px[3], 65535, "Row 0, col {x}: A should be 65535");
        }

        // Row 1: white → sRGB(1.0) = 1.0 → u16 ≈ 65535
        for x in 0..3 {
            let px = out_img.get_pixel(x, 1);
            assert!(
                (px[0] as i32 - 65535).abs() <= 64,
                "Row 1, col {x}: R expected ~65535, got {}",
                px[0]
            );
            assert!(
                (px[1] as i32 - 65535).abs() <= 64,
                "Row 1, col {x}: G expected ~65535, got {}",
                px[1]
            );
            assert!(
                (px[2] as i32 - 65535).abs() <= 64,
                "Row 1, col {x}: B expected ~65535, got {}",
                px[2]
            );
            assert_eq!(px[3], 65535, "Row 1, col {x}: A should be 65535");
        }
    }
}
