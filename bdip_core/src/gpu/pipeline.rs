use crate::gpu::engine::GpuEngine;
use crate::transformation::Transformation;
use std::collections::HashMap;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, ComputePipeline, ComputePipelineDescriptor,
    PipelineLayoutDescriptor, ShaderStages, StorageTextureAccess, TextureFormat,
    TextureViewDescriptor, TextureViewDimension, util::DeviceExt,
};

// ========== Uniform structs ==========

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BrightnessParams {
    brightness_offset: f32,
    _padding: [f32; 3], // WebGPU uniforms require 16-byte alignment
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SaturationParams {
    saturation_offset: f32,
    _padding: [f32; 3], // WebGPU uniforms require 16-byte alignment
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ContrastParams {
    contrast_offset: f32,
    _padding: [f32; 3], // WebGPU uniforms require 16-byte alignment
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GrayscaleParams {
    _unused: [f32; 4], // WebGPU uniforms require 16-byte alignment; no user params
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct InvertParams {
    _unused: [f32; 4], // WebGPU uniforms require 16-byte alignment; no user params
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PresentParams {
    width: u32,
    y_offset: u32,
    tile_height: u32,
    _padding: u32, // WebGPU uniforms require 16-byte alignment
}

// ========== TransformKind ==========

/// Lightweight discriminant identifying which compiled pipeline to use.
/// Derived from a `Transformation` variant — carries no parameter values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TransformKind {
    Brightness,
    Saturation,
    Contrast,
    Grayscale,
    Invert,
}

impl From<&Transformation> for TransformKind {
    fn from(t: &Transformation) -> Self {
        match t {
            Transformation::Brightness(_) => TransformKind::Brightness,
            Transformation::Saturation(_) => TransformKind::Saturation,
            Transformation::Contrast(_) => TransformKind::Contrast,
            Transformation::Grayscale => TransformKind::Grayscale,
            Transformation::Invert => TransformKind::Invert,
        }
    }
}

// ========== CachedPipeline ==========

struct CachedPipeline {
    pipeline: ComputePipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
}

// ========== PipelineCache ==========

/// Lazily compiles and caches transform pipelines on first use.
struct PipelineCache {
    cache: HashMap<TransformKind, CachedPipeline>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Returns a reference to the compiled pipeline for `kind`, compiling it on
    /// first access and caching the result for all subsequent calls.
    fn get_or_create(&mut self, device: &wgpu::Device, kind: TransformKind) -> &CachedPipeline {
        self.cache
            .entry(kind)
            .or_insert_with(|| Self::compile(device, kind))
    }

    fn compile(device: &wgpu::Device, kind: TransformKind) -> CachedPipeline {
        let (
            shader_src,
            shader_label,
            pipeline_label,
            texture_bgl_label,
            params_bgl_label,
            pl_label,
        ) = match kind {
            TransformKind::Brightness => (
                include_str!("brightness.wgsl"),
                "Brightness Shader",
                "Brightness Pipeline",
                "Brightness Texture BGL",
                "Brightness Params BGL",
                "Brightness Pipeline Layout",
            ),
            TransformKind::Saturation => (
                include_str!("saturation.wgsl"),
                "Saturation Shader",
                "Saturation Pipeline",
                "Saturation Texture BGL",
                "Saturation Params BGL",
                "Saturation Pipeline Layout",
            ),
            TransformKind::Contrast => (
                include_str!("contrast.wgsl"),
                "Contrast Shader",
                "Contrast Pipeline",
                "Contrast Texture BGL",
                "Contrast Params BGL",
                "Contrast Pipeline Layout",
            ),
            TransformKind::Grayscale => (
                include_str!("grayscale.wgsl"),
                "Grayscale Shader",
                "Grayscale Pipeline",
                "Grayscale Texture BGL",
                "Grayscale Params BGL",
                "Grayscale Pipeline Layout",
            ),
            TransformKind::Invert => (
                include_str!("invert.wgsl"),
                "Invert Shader",
                "Invert Pipeline",
                "Invert Texture BGL",
                "Invert Params BGL",
                "Invert Pipeline Layout",
            ),
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_label),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let texture_bind_group_layout =
            make_texture_only_bind_group_layout(device, texture_bgl_label);

        let params_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(params_bgl_label),
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

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(pl_label),
            bind_group_layouts: &[
                Some(&texture_bind_group_layout),
                Some(&params_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(pipeline_label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        CachedPipeline {
            pipeline,
            texture_bind_group_layout,
            params_bind_group_layout,
        }
    }
}

// ========== Helpers ==========

/// Shared bind group layout for the core image data (one source texture, one
/// destination storage texture). Used by the Ingest pass and as Bind Group 0
/// for all transform passes.
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

// ========== Renderer ==========

pub struct Renderer {
    // Ingest and present pipelines are always needed — eagerly initialized.
    ingest_pipeline: ComputePipeline,
    ingest_bind_group_layout: wgpu::BindGroupLayout,
    present_pipeline: ComputePipeline,
    present_texture_bind_group_layout: wgpu::BindGroupLayout,
    present_params_bind_group_layout: wgpu::BindGroupLayout,

    // Transform pipelines are compiled on first use.
    pipeline_cache: PipelineCache,

    // Cached staging buffer for readback. Reused across `download` calls to
    // avoid per-call OS allocation of a large MAP_READ buffer.
    staging_buffer: Option<(wgpu::Buffer, u64)>, // (buffer, byte_size)
}

impl Renderer {
    pub fn new(engine: &GpuEngine) -> Self {
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
            ingest_pipeline,
            ingest_bind_group_layout,
            present_pipeline,
            present_texture_bind_group_layout,
            present_params_bind_group_layout,
            pipeline_cache: PipelineCache::new(),
            staging_buffer: None,
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
    /// packed into a tightly packed storage buffer (2 u32s per pixel, no row
    /// padding). Must be the last pass before `download_presentation_buffer`.
    ///
    /// For images whose full buffer exceeds `max_storage_buffer_binding_size`,
    /// the dispatch is split into row-tiles. Each tile writes to a
    /// binding-sized buffer, which is then copied into the correct offset of
    /// the full output buffer. The shader's `y_offset` uniform shifts texture
    /// reads so each tile processes the correct rows.
    pub fn present(&self, engine: &GpuEngine, src_texture: &wgpu::Texture) -> wgpu::Buffer {
        let max_binding = engine.device.limits().max_storage_buffer_binding_size;
        self.present_with_max_binding(engine, src_texture, max_binding)
    }

    /// Inner implementation of `present` that accepts an explicit binding-size
    /// limit. Tests use this to force tiling on small images.
    pub(crate) fn present_with_max_binding(
        &self,
        engine: &GpuEngine,
        src_texture: &wgpu::Texture,
        max_binding_size: u64,
    ) -> wgpu::Buffer {
        let (width, height) = (src_texture.width(), src_texture.height());
        let pixel_size: u64 = 8; // 4 channels × 2 bytes
        let full_size = width as u64 * height as u64 * pixel_size;

        let max_rows = (max_binding_size / (width as u64 * pixel_size)).min(height as u64) as u32;

        let output_buffer = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present_output_buffer"),
            size: full_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let src_view = src_texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut y_offset: u32 = 0;
        while y_offset < height {
            let tile_height = max_rows.min(height - y_offset);
            let tile_size = width as u64 * tile_height as u64 * pixel_size;

            let tile_buffer = engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("present_tile_buffer"),
                size: tile_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let params = PresentParams {
                width,
                y_offset,
                tile_height,
                _padding: 0,
            };
            let params_buffer =
                engine
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("present_params_buffer"),
                        contents: bytemuck::cast_slice(&[params]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

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
                        resource: tile_buffer.as_entire_binding(),
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

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.present_pipeline);
                cpass.set_bind_group(0, &texture_bind_group, &[]);
                cpass.set_bind_group(1, &params_bind_group, &[]);
                cpass.dispatch_workgroups(width.div_ceil(16), tile_height.div_ceil(16), 1);
            }

            encoder.copy_buffer_to_buffer(
                &tile_buffer,
                0,
                &output_buffer,
                y_offset as u64 * width as u64 * pixel_size,
                tile_size,
            );

            y_offset += tile_height;
        }

        engine.queue.submit(Some(encoder.finish()));

        output_buffer
    }

    fn ensure_staging_buffer(&mut self, engine: &GpuEngine, buffer_size: u64) -> &wgpu::Buffer {
        let needs_alloc = match &self.staging_buffer {
            Some((_, sz)) => *sz < buffer_size,
            None => true,
        };

        if needs_alloc {
            let buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("presentation_staging_buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.staging_buffer = Some((buf, buffer_size));
        }

        &self.staging_buffer.as_ref().unwrap().0
    }

    /// Downloads the output of `present` from a GPU storage buffer into a
    /// CPU-side `Rgba16Image`. Reuses a cached `MAP_READ` staging buffer across
    /// calls to avoid the per-call OS allocation cost (~15–30 ms for a 192 MB
    /// buffer on Apple Silicon).
    ///
    /// Use `bdip_core::gpu::texture::download_presentation_buffer` for one-shot
    /// callers (headless CLI, tests) that do not hold a `Renderer` across calls.
    pub fn download(
        &mut self,
        engine: &GpuEngine,
        src_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Result<crate::Rgba16Image, crate::error::BdipError> {
        // Tightly packed: 4 channels × 2 bytes = 8 bytes per pixel, no row padding.
        let buffer_size = width as u64 * height as u64 * 8;

        let staging_buffer = self.ensure_staging_buffer(engine, buffer_size);

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(src_buffer, 0, staging_buffer, 0, buffer_size);
        engine.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..buffer_size);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        engine
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

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

        crate::Rgba16Image::from_raw(width, height, pixel_vec)
            .ok_or_else(|| crate::error::BdipError::Gpu("Presentation buffer size mismatch".into()))
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

    /// Applies a single `Transformation` to `src_texture` and returns a new
    /// `Rgba16Float` texture in linear light. The correct pipeline is compiled
    /// on first use and cached for subsequent calls with the same transform kind.
    pub fn apply(
        &mut self,
        engine: &GpuEngine,
        src_texture: &wgpu::Texture,
        transformation: &Transformation,
    ) -> wgpu::Texture {
        let kind = TransformKind::from(transformation);
        let cached = self.pipeline_cache.get_or_create(&engine.device, kind);

        let (width, height, depth) = (
            src_texture.width(),
            src_texture.height(),
            src_texture.depth_or_array_layers(),
        );

        let dst_texture = engine.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("apply_dst_texture"),
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
            label: Some("Apply Texture Bind Group"),
            layout: &cached.texture_bind_group_layout,
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

        let params_buffer = match transformation {
            Transformation::Brightness(val) => {
                let p = BrightnessParams {
                    brightness_offset: *val,
                    _padding: [0.0; 3],
                };
                engine
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Apply Params Buffer"),
                        contents: bytemuck::cast_slice(&[p]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    })
            }
            Transformation::Saturation(val) => {
                let p = SaturationParams {
                    saturation_offset: *val,
                    _padding: [0.0; 3],
                };
                engine
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Apply Params Buffer"),
                        contents: bytemuck::cast_slice(&[p]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    })
            }
            Transformation::Contrast(val) => {
                let p = ContrastParams {
                    contrast_offset: *val,
                    _padding: [0.0; 3],
                };
                engine
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Apply Params Buffer"),
                        contents: bytemuck::cast_slice(&[p]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    })
            }
            Transformation::Grayscale => {
                let p = GrayscaleParams { _unused: [0.0; 4] };
                engine
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Apply Params Buffer"),
                        contents: bytemuck::cast_slice(&[p]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    })
            }
            Transformation::Invert => {
                let p = InvertParams { _unused: [0.0; 4] };
                engine
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Apply Params Buffer"),
                        contents: bytemuck::cast_slice(&[p]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    })
            }
        };

        let params_bind_group = engine.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Apply Params Bind Group"),
            layout: &cached.params_bind_group_layout,
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
            cpass.set_pipeline(&cached.pipeline);
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
    use crate::Transformation;
    use crate::gpu::texture::{download_presentation_buffer, upload_texture};

    // ========== Helpers ==========

    fn make_solid_image(w: u32, h: u32, r: u16, g: u16, b: u16) -> crate::Rgba16Image {
        let mut img = crate::Rgba16Image::new(w, h);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([r, g, b, 65535]);
        }
        img
    }

    fn roundtrip(
        renderer: &mut Renderer,
        engine: &GpuEngine,
        img: &crate::Rgba16Image,
        transforms: &[Transformation],
    ) -> crate::Rgba16Image {
        let (w, h) = (img.width(), img.height());
        let upload = upload_texture(&engine.device, &engine.queue, img);
        let mut current = renderer.ingest(engine, &upload);
        for t in transforms {
            current = renderer.apply(engine, &current, t);
        }
        let buf = renderer.present(engine, &current);
        download_presentation_buffer(&engine.device, &engine.queue, &buf, w, h).unwrap()
    }

    // ========== Existing brightness tests (updated to use apply()) ==========

    #[test]
    fn test_brightness_shader_positive() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% gray in sRGB (32767/65535 ≈ 0.500 sRGB → ~0.214 linear)
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Brightness(0.5)],
        );

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
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_brightness_shader_negative() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% gray in sRGB → ~0.214 linear; 0.214 - 0.6 = -0.386 → clamped to 0
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Brightness(-0.6)],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[0], 0);
            assert_eq!(pixel[1], 0);
            assert_eq!(pixel[2], 0);
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_brightness_shader_zero() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 10794, 25700, 51400);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Brightness(0.0)],
        );

        // sRGB → linear → sRGB is a mathematical identity; differences are f16 rounding.
        for pixel in out_img.pixels() {
            assert!((pixel[0] as i32 - 10794).abs() <= 64);
            assert!((pixel[1] as i32 - 25700).abs() <= 64);
            assert!((pixel[2] as i32 - 51400).abs() <= 64);
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_shader_chaining() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 51400/65535 ≈ 0.784 sRGB → ~0.577 linear
        let img = make_solid_image(2, 2, 51400, 51400, 51400);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transformation::Brightness(-0.2),
                Transformation::Brightness(0.5),
            ],
        );

        // 0.577 - 0.2 = 0.377; 0.377 + 0.5 = 0.877 linear → sRGB ≈ 0.944 → u16 ≈ 61880
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
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_shader_headroom_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32767/65535 ≈ 0.500 sRGB → ~0.214 linear
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transformation::Brightness(0.8),
                Transformation::Brightness(-0.8),
            ],
        );

        // 0.214 + 0.8 = 1.014 (above 1.0, held in headroom); 1.014 - 0.8 = 0.214 linear
        // linear_to_srgb(0.214) ≈ 0.500 sRGB → u16 ≈ 32767; allow ±64 for f16 precision
        for pixel in out_img.pixels() {
            assert!((pixel[0] as i32 - 32767).abs() <= 64);
            assert!((pixel[1] as i32 - 32767).abs() <= 64);
            assert!((pixel[2] as i32 - 32767).abs() <= 64);
            assert_eq!(pixel[3], 65535);
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
            assert_eq!(pixel[3], 65535);
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

    #[test]
    fn test_present_tiling_one_row_per_tile() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);

        // 4×4 image with distinct per-row colors to detect row-offset bugs.
        let mut img = crate::Rgba16Image::new(4, 4);
        let row_values: [u16; 4] = [0, 16384, 49152, 65535];
        for y in 0..4u32 {
            for x in 0..4u32 {
                let v = row_values[y as usize];
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let upload = upload_texture(&engine.device, &engine.queue, &img);
        let linear = renderer.ingest(&engine, &upload);

        // Force 1 row per tile: limit = width * 8 bytes (one row exactly).
        let one_row = 4u64 * 8;
        let tiled_buf = renderer.present_with_max_binding(&engine, &linear, one_row);
        let tiled_img =
            download_presentation_buffer(&engine.device, &engine.queue, &tiled_buf, 4, 4).unwrap();

        // Compare against the non-tiled path (uses real device limit → single tile).
        let ref_buf = renderer.present(&engine, &linear);
        let ref_img =
            download_presentation_buffer(&engine.device, &engine.queue, &ref_buf, 4, 4).unwrap();

        for y in 0..4u32 {
            for x in 0..4u32 {
                let tp = tiled_img.get_pixel(x, y);
                let rp = ref_img.get_pixel(x, y);
                assert_eq!(tp, rp, "Pixel ({x},{y}) mismatch: tiled={tp:?}, ref={rp:?}");
            }
        }
    }

    #[test]
    fn test_present_tiling_uneven_rows() {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);

        // 3×5 image — 5 rows don't divide evenly into 2-row tiles.
        let mut img = crate::Rgba16Image::new(3, 5);
        for y in 0..5u32 {
            let v = (y * 16384).min(65535) as u16;
            for x in 0..3u32 {
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let upload = upload_texture(&engine.device, &engine.queue, &img);
        let linear = renderer.ingest(&engine, &upload);

        // Force 2 rows per tile: tiles of 2, 2, 1 rows.
        let two_rows = 3u64 * 2 * 8;
        let tiled_buf = renderer.present_with_max_binding(&engine, &linear, two_rows);
        let tiled_img =
            download_presentation_buffer(&engine.device, &engine.queue, &tiled_buf, 3, 5).unwrap();

        let ref_buf = renderer.present(&engine, &linear);
        let ref_img =
            download_presentation_buffer(&engine.device, &engine.queue, &ref_buf, 3, 5).unwrap();

        for y in 0..5u32 {
            for x in 0..3u32 {
                let tp = tiled_img.get_pixel(x, y);
                let rp = ref_img.get_pixel(x, y);
                assert_eq!(tp, rp, "Pixel ({x},{y}) mismatch: tiled={tp:?}, ref={rp:?}");
            }
        }
    }

    // ========== Saturation correctness tests ==========

    #[test]
    fn test_saturation_zero_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Use a non-gray, non-neutral color so saturation has values to act on.
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Saturation(0.0)],
        );

        // saturation_offset=0 → scale=1.0 → identity; only f16 rounding applies.
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 16384).abs() <= 64,
                "G: expected ~16384, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 8192).abs() <= 64,
                "B: expected ~8192, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_saturation_negative_one_produces_grayscale() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure red: R=65535, G=0, B=0 in sRGB. After desaturation, all channels
        // equal Rec.709 luminance of the linear values: lum = 0.2126*1.0 = 0.2126.
        let img = make_solid_image(2, 2, 65535, 0, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Saturation(-1.0)],
        );

        for pixel in out_img.pixels() {
            // All three channels should be equal — the defining property of grayscale.
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "R and G should be equal after full desaturation: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[1] as i32 - pixel[2] as i32).abs() <= 64,
                "G and B should be equal after full desaturation: G={}, B={}",
                pixel[1],
                pixel[2]
            );
            // The gray value should reflect luminance, not zero or the original red.
            assert!(
                pixel[0] > 0 && pixel[0] < 65535,
                "Desaturated value should be a mid-tone, got {}",
                pixel[0]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_saturation_positive_increases_color() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Input: warm color where R > G > B.
        // After positive saturation: R should increase (it's above luminance),
        // G and B should decrease (they're below luminance).
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Saturation(0.5)],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[0] > 32767,
                "R should increase with positive saturation: got {}",
                pixel[0]
            );
            assert!(
                pixel[1] < 16384,
                "G should decrease with positive saturation: got {}",
                pixel[1]
            );
            assert!(
                pixel[2] < 8192,
                "B should decrease with positive saturation: got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    // ========== PipelineCache behavior tests ==========

    #[test]
    fn test_pipeline_cache_returns_same_pipeline() {
        let engine = GpuEngine::new().unwrap();
        let mut cache = PipelineCache::new();

        // Calling get_or_create twice for the same kind must return the same
        // cached entry — no recompilation on the second call.
        let p1 =
            cache.get_or_create(&engine.device, TransformKind::Brightness) as *const CachedPipeline;
        let p2 =
            cache.get_or_create(&engine.device, TransformKind::Brightness) as *const CachedPipeline;
        assert!(
            std::ptr::eq(p1, p2),
            "same TransformKind should return the same cached pipeline pointer"
        );
    }

    #[test]
    fn test_pipeline_cache_different_kinds() {
        let engine = GpuEngine::new().unwrap();
        let mut cache = PipelineCache::new();

        // Brightness and Saturation must occupy separate cache entries.
        let pb =
            cache.get_or_create(&engine.device, TransformKind::Brightness) as *const CachedPipeline;
        let ps =
            cache.get_or_create(&engine.device, TransformKind::Saturation) as *const CachedPipeline;
        assert!(
            !std::ptr::eq(pb, ps),
            "different TransformKinds should return different pipeline pointers"
        );
    }

    // ========== Chaining tests ==========

    #[test]
    fn test_chained_brightness_then_saturation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);

        // Apply each transform independently to establish baselines.
        let brightness_only = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Brightness(0.3)],
        );
        let saturation_only = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Saturation(-0.5)],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transformation::Brightness(0.3),
                Transformation::Saturation(-0.5),
            ],
        );

        // The chained result must differ from either single-transform result.
        let r_chain = chained.get_pixel(0, 0)[0] as i32;
        let r_bright = brightness_only.get_pixel(0, 0)[0] as i32;
        let r_sat = saturation_only.get_pixel(0, 0)[0] as i32;
        assert!(
            (r_chain - r_bright).abs() > 64,
            "chained result should differ from brightness-only: chain={r_chain}, bright={r_bright}"
        );
        assert!(
            (r_chain - r_sat).abs() > 64,
            "chained result should differ from saturation-only: chain={r_chain}, sat={r_sat}"
        );
    }

    #[test]
    fn test_chained_saturation_then_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);

        // Note: brightness (uniform additive offset) and saturation (linear scaling around
        // luminance) commute exactly when Rec.709 coefficients sum to 1.0 — which they do
        // (0.2126 + 0.7152 + 0.0722 = 1.0). Both orderings produce algebraically identical
        // results. This test verifies the chaining mechanism works regardless of order and
        // that the outputs are consistent.
        let bright_then_sat = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transformation::Brightness(0.3),
                Transformation::Saturation(-0.5),
            ],
        );
        let sat_then_bright = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transformation::Saturation(-0.5),
                Transformation::Brightness(0.3),
            ],
        );

        for y in 0..2u32 {
            for x in 0..2u32 {
                let a = bright_then_sat.get_pixel(x, y);
                let b = sat_then_bright.get_pixel(x, y);
                assert!(
                    (a[0] as i32 - b[0] as i32).abs() <= 64,
                    "R at ({x},{y}): order A={}, order B={}",
                    a[0],
                    b[0]
                );
                assert!(
                    (a[1] as i32 - b[1] as i32).abs() <= 64,
                    "G at ({x},{y}): order A={}, order B={}",
                    a[1],
                    b[1]
                );
                assert!(
                    (a[2] as i32 - b[2] as i32).abs() <= 64,
                    "B at ({x},{y}): order A={}, order B={}",
                    a[2],
                    b[2]
                );
            }
        }
    }

    #[test]
    fn test_multiple_same_transform() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // ~50% gray in sRGB → ~0.214 linear. Each +0.3 brightness step accumulates.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);

        let once = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Brightness(0.3)],
        );
        let twice = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transformation::Brightness(0.3),
                Transformation::Brightness(0.3),
            ],
        );

        // 0.214 + 0.3 = 0.514 linear → sRGB ≈ 0.744 → u16 ≈ 48777
        // 0.214 + 0.6 = 0.814 linear → sRGB ≈ 0.913 → u16 ≈ 59840
        // Applying brightness twice must produce a strictly brighter result.
        let r_once = once.get_pixel(0, 0)[0];
        let r_twice = twice.get_pixel(0, 0)[0];
        assert!(
            r_twice > r_once,
            "two brightness passes should accumulate: once={r_once}, twice={r_twice}"
        );
    }

    // ========== Contrast correctness tests ==========

    #[test]
    fn test_contrast_zero_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Use a non-neutral color so the shader has meaningful values to act on.
        let img = make_solid_image(2, 2, 10794, 25700, 51400);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Contrast(0.0)],
        );

        // contrast_offset=0 → scale=1.0 → identity; only f16 rounding applies.
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 10794).abs() <= 64,
                "R: expected ~10794, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 25700).abs() <= 64,
                "G: expected ~25700, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 51400).abs() <= 64,
                "B: expected ~51400, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_contrast_max_positive_clamps_below_midpoint_to_black() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% gray sRGB (≈0.214 linear) is below the 0.5 linear midpoint.
        // contrast=1.0 → scale=2.0 → (0.214 - 0.5)*2.0 + 0.5 = -0.072 → clamped to 0.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Contrast(1.0)],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[0], 0, "R: below-midpoint pixel should clamp to 0");
            assert_eq!(pixel[1], 0, "G: below-midpoint pixel should clamp to 0");
            assert_eq!(pixel[2], 0, "B: below-midpoint pixel should clamp to 0");
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_contrast_max_positive_pushes_above_midpoint_brighter() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 51400/65535 ≈ 0.784 sRGB → ≈0.577 linear (above 0.5 midpoint).
        // contrast=1.0 → (0.577 - 0.5)*2.0 + 0.5 = 0.655 linear → sRGB ≈ 0.829 → u16 ≈ 54366.
        let img = make_solid_image(2, 2, 51400, 51400, 51400);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Contrast(1.0)],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[0] > 51400,
                "R: above-midpoint pixel should brighten with positive contrast, got {}",
                pixel[0]
            );
            assert!(
                (pixel[0] as i32 - 54366).abs() <= 128,
                "R: expected ~54366, got {}",
                pixel[0]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_contrast_max_negative_flattens_to_neutral_gray() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // contrast=-1.0 → scale=0.0 → all channels become 0.5 linear regardless of input.
        // 0.5 linear → sRGB ≈ 0.735 → u16 ≈ 48184.
        let img = make_solid_image(2, 2, 0, 0, 0); // pure black input
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Contrast(-1.0)],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 48184).abs() <= 128,
                "R: expected neutral gray ~48184, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 48184).abs() <= 128,
                "G: expected neutral gray ~48184, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 48184).abs() <= 128,
                "B: expected neutral gray ~48184, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_contrast_preserves_alpha() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Verify that the alpha channel is untouched at max contrast.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Contrast(1.0)],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by contrast");
        }
    }

    #[test]
    fn test_contrast_chained_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32767/65535 ≈ 0.500 sRGB → ≈0.214 linear.
        // brightness +0.3 → 0.514 linear.
        // contrast +0.5 → scale=1.5 → (0.514-0.5)*1.5 + 0.5 = 0.521 linear.
        // sRGB(0.521) ≈ 0.749 → u16 ≈ 49097.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transformation::Brightness(0.3),
                Transformation::Contrast(0.5),
            ],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 49097).abs() <= 128,
                "R: expected ~49097, got {}",
                pixel[0]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    // ========== Grayscale correctness tests ==========

    #[test]
    fn test_grayscale_produces_equal_rgb_channels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Colored input: channels are distinct, so any non-trivial operation is detectable.
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(&mut renderer, &engine, &img, &[Transformation::Grayscale]);

        // After grayscale, R, G, and B must all equal the luminance value.
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "R and G should be equal: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[1] as i32 - pixel[2] as i32).abs() <= 64,
                "G and B should be equal: G={}, B={}",
                pixel[1],
                pixel[2]
            );
        }
    }

    #[test]
    fn test_grayscale_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(&mut renderer, &engine, &img, &[Transformation::Grayscale]);

        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by grayscale");
        }
    }

    #[test]
    fn test_grayscale_all_black_stays_black() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure black: all channels 0 linear → luminance = 0 → output stays 0.
        let img = make_solid_image(2, 2, 0, 0, 0);
        let out_img = roundtrip(&mut renderer, &engine, &img, &[Transformation::Grayscale]);

        for pixel in out_img.pixels() {
            assert_eq!(
                pixel[0], 0,
                "R: black input should produce 0, got {}",
                pixel[0]
            );
            assert_eq!(
                pixel[1], 0,
                "G: black input should produce 0, got {}",
                pixel[1]
            );
            assert_eq!(
                pixel[2], 0,
                "B: black input should produce 0, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_grayscale_all_white_stays_white() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure white: all channels 1.0 linear → luminance = 0.2126+0.7152+0.0722 = 1.0 → white.
        let img = make_solid_image(2, 2, 65535, 65535, 65535);
        let out_img = roundtrip(&mut renderer, &engine, &img, &[Transformation::Grayscale]);

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 64,
                "R: white input should stay white, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 65535).abs() <= 64,
                "G: white input should stay white, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 65535).abs() <= 64,
                "B: white input should stay white, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_grayscale_chained_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Apply brightness first to shift the values, then grayscale.
        // The result must still have equal R=G=B channels.
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Brightness(0.2), Transformation::Grayscale],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "R and G should be equal after brightness+grayscale: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[1] as i32 - pixel[2] as i32).abs() <= 64,
                "G and B should be equal after brightness+grayscale: G={}, B={}",
                pixel[1],
                pixel[2]
            );
        }
    }

    // ========== Invert correctness tests ==========

    #[test]
    fn test_invert_shader() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% gray in sRGB -> inverted should still be valid. Let's use custom color.
        // 10000 / 65535, inverted is 55535 / 65535.
        // Note: linear-light invert means 1.0 - linear_value.
        let img = make_solid_image(2, 2, 0, 65535, 32767);
        let out_img = roundtrip(&mut renderer, &engine, &img, &[Transformation::Invert]);

        for pixel in out_img.pixels() {
            // R: 0 -> inverted -> 65535
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 100,
                "R: expected ~65535, got {}",
                pixel[0]
            );
            // G: 65535 -> inverted -> 0
            assert!(pixel[1] <= 100, "G: expected ~0, got {}", pixel[1]);
            // Alpha preserved
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_double_invert_restores_original() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 10794, 25700, 51400);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transformation::Invert, Transformation::Invert],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 10794).abs() <= 128,
                "R: expected ~10794, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 25700).abs() <= 128,
                "G: expected ~25700, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 51400).abs() <= 128,
                "B: expected ~51400, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    // ========== Performance benchmark ==========

    /// Times the GPU-critical path on a 24 MP synthetic image — the primary target
    /// size from perf_goal_1.md.
    ///
    /// Two runs are measured to isolate warm-pipeline performance from one-time
    /// startup costs:
    ///
    /// - **Run 1 (cold)**: upload → ingest → apply → present → readback.
    ///   Includes shader compilation and initial staging buffer allocation.
    /// - **Run 2 (warm)**: apply → present → readback, reusing the ingested
    ///   base texture and the cached staging buffer from run 1. This matches
    ///   the interactive editing path, where upload+ingest happen once and are
    ///   cached across slider changes.
    ///
    /// Run 2 is the number to compare against the 8–20 ms target.
    ///
    /// This test is ignored by default so it does not run in CI. Run it manually:
    ///   cargo perf-test
    ///
    /// Measurements on Apple M4 Pro (2026-04):
    ///   gpu upload:              ~75.58 ms  (CPU f16 conversion loop; see tech_debt.md)
    ///   run 1 execute:           ~1.71 ms
    ///   run 1 readback:          ~64.36 ms  (staging buffer alloc + memcpy)
    ///   run 1 critical path:     ~66.07 ms
    ///   run 2 execute:           ~0.76 ms
    ///   run 2 readback:          ~34.28 ms (used previously alloced buffer)
    ///   run 2 critical path:     ~35.04 ms
    ///
    /// Target once all known bottlenecks are resolved (warm, interactive): 8–20 ms.
    /// When the target is reliably met, add an assertion and remove #[ignore].
    #[test]
    #[ignore = "performance benchmark: run manually to avoid slowing CI"]
    fn test_perf_gpu_roundtrip_24mp() {
        use std::time::Instant;

        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 5000×4800 = 24,000,000 pixels (~24 MP), matching the primary benchmark
        // target in perf_goal_1.md. Generated synthetically — no test asset needed.
        let img = make_solid_image(5000, 4800, 32767, 32767, 32767);

        let t_upload = Instant::now();
        let uploaded = upload_texture(&engine.device, &engine.queue, &img);
        let upload_ms = t_upload.elapsed().as_secs_f64() * 1000.0;

        // --- Run 1: cold (shader compilation + initial staging buffer allocation) ---
        let t_execute_1 = Instant::now();
        let ingested = renderer.ingest(&engine, &uploaded);
        let transformed_1 = renderer.apply(&engine, &ingested, &Transformation::Brightness(0.1));
        let present_buf_1 = renderer.present(&engine, &transformed_1);
        let execute_ms_1 = t_execute_1.elapsed().as_secs_f64() * 1000.0;

        let t_readback_1 = Instant::now();
        let _result_1 = renderer
            .download(&engine, &present_buf_1, img.width(), img.height())
            .unwrap();
        let readback_ms_1 = t_readback_1.elapsed().as_secs_f64() * 1000.0;

        // --- Run 2: warm (shaders compiled, staging buffer reused) ---
        // Reuses `ingested` directly — matching interactive editing where the base
        // texture is cached in VRAM across slider changes.
        let t_execute_2 = Instant::now();
        let transformed_2 = renderer.apply(&engine, &ingested, &Transformation::Brightness(0.1));
        let present_buf_2 = renderer.present(&engine, &transformed_2);
        let execute_ms_2 = t_execute_2.elapsed().as_secs_f64() * 1000.0;

        let t_readback_2 = Instant::now();
        let _result_2 = renderer
            .download(&engine, &present_buf_2, img.width(), img.height())
            .unwrap();
        let readback_ms_2 = t_readback_2.elapsed().as_secs_f64() * 1000.0;

        eprintln!("--- 24 MP GPU roundtrip ---");
        eprintln!("  gpu upload:                   {:>8.2} ms", upload_ms);
        eprintln!(
            "  run 1 execute (ingest+apply+present): {:>8.2} ms",
            execute_ms_1
        );
        eprintln!("  run 1 readback:               {:>8.2} ms", readback_ms_1);
        eprintln!(
            "  run 1 critical path:          {:>8.2} ms",
            execute_ms_1 + readback_ms_1
        );
        eprintln!(
            "  run 2 execute (apply+present):        {:>8.2} ms",
            execute_ms_2
        );
        eprintln!("  run 2 readback:               {:>8.2} ms", readback_ms_2);
        eprintln!(
            "  run 2 critical path:          {:>8.2} ms  (target: 8–20 ms warm)",
            execute_ms_2 + readback_ms_2
        );
        eprintln!("----------------------------------");
    }
}
