use crate::gpu::engine::GpuEngine;
use crate::gpu::shaders::{PassInput, PassOutput, Transform, registry_by_id};
use std::collections::{HashMap, hash_map::Entry};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, ComputePipeline, ComputePipelineDescriptor,
    PipelineLayoutDescriptor, ShaderStages, StorageTextureAccess, TextureFormat,
    TextureViewDescriptor, TextureViewDimension, util::DeviceExt,
};

// ========== Uniform structs ==========

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PresentParams {
    width: u32,
    y_offset: u32,
    tile_height: u32,
    _padding: u32, // WebGPU uniforms require 16-byte alignment
}

// ========== CompiledPass ==========

struct CompiledPass {
    pipeline: ComputePipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
}

// ========== ShaderPassesCache ==========

/// Lazily compiles and caches transform pipelines on first use.
struct ShaderPassesCache {
    cache: HashMap<&'static str, Vec<CompiledPass>>,
}

impl ShaderPassesCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Returns a reference to the compiled pipeline for `shader_id`, compiling
    /// it on first access and caching the result for all subsequent calls.
    fn get_or_create(&mut self, device: &wgpu::Device, shader_id: &'static str) -> &[CompiledPass] {
        self.cache
            .entry(shader_id)
            .or_insert_with(|| Self::compile(device, shader_id))
    }

    fn compile(device: &wgpu::Device, shader_id: &'static str) -> Vec<CompiledPass> {
        let reg =
            registry_by_id(shader_id).unwrap_or_else(|| panic!("Unknown shader ID: '{shader_id}'"));
        let meta = &reg.meta;

        let mut pipelines = Vec::new();

        for pass in meta.passes {
            let shader_label = format!("{} Shader ({})", meta.display_name, pass.label);
            let pipeline_label = format!("{} Pipeline ({})", meta.display_name, pass.label);
            let texture_bgl_label = format!("{} Texture BGL ({})", meta.display_name, pass.label);
            let params_bgl_label = format!("{} Params BGL ({})", meta.display_name, pass.label);
            let pl_label = format!("{} Pipeline Layout ({})", meta.display_name, pass.label);

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&shader_label),
                source: wgpu::ShaderSource::Wgsl(pass.wgsl_source.into()),
            });

            let texture_bind_group_layout =
                build_pass_bind_group_layout(device, pass.inputs.len() as u32, &texture_bgl_label);

            let params_bind_group_layout =
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some(&params_bgl_label),
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
                label: Some(&pl_label),
                bind_group_layouts: &[
                    Some(&texture_bind_group_layout),
                    Some(&params_bind_group_layout),
                ],
                immediate_size: 0,
            });

            let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&pipeline_label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            pipelines.push(CompiledPass {
                pipeline,
                texture_bind_group_layout,
                params_bind_group_layout,
            });
        }

        pipelines
    }
}

// ========== Helpers ==========

fn build_pass_bind_group_layout(
    device: &wgpu::Device,
    input_count: u32,
    label: &str,
) -> wgpu::BindGroupLayout {
    let mut entries = Vec::new();
    for i in 0..input_count {
        entries.push(BindGroupLayoutEntry {
            binding: i,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
    }
    entries.push(BindGroupLayoutEntry {
        binding: input_count,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba16Float,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    });
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

// ========== ScratchPool ==========

struct ScratchPool {
    width: u32,
    height: u32,
    textures: Vec<wgpu::Texture>,
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
    passes_cache: ShaderPassesCache,

    scratch_pool: ScratchPool,

    // Cached staging buffer for readback. Reused across `download` calls to
    // avoid per-call OS allocation of a large MAP_READ buffer.
    staging_buffer: Option<(wgpu::Buffer, u64)>, // (buffer, byte_size)

    // We considered caching 3 `present buffers` but only ended up caching the
    // `present_tile_buffer`. It can be reused across calls to avoid per-call
    // GPU buffer allocation and is only reallocated when image dimensions grow.
    //
    // `present_tile_buffer`: the per-tile STORAGE buffer sized for the worst-case
    //   tile (`max_rows * width * 8` bytes). Reused across tiles in the same
    //   call (tiles are processed sequentially) and across calls (the result is
    //   copied into the fresh output buffer each time).
    //
    // The output buffer returned by `present` is allocated fresh each call
    // because callers hold it alive (for `download`) while the next `present`
    // call would otherwise overwrite the same GPU memory.
    //
    // The params uniform (16 bytes) is not cached: it is created fresh per tile
    // via `create_buffer_init` because `queue.write_buffer` is submitted
    // immediately (before the command encoder's compute passes run), so a single
    // cached buffer updated in a loop would cause all tiles to read the last
    // tile's params. Per-tile allocation of 16 bytes is essentially free.
    present_tile_buffer: Option<(wgpu::Buffer, u64)>, // (buffer, byte_size)

    // Reusable pixel buffer for the interactive readback path. `download_slice`
    // copies GPU data into this Vec rather than allocating a fresh one each frame,
    // keeping the hot path allocation-free. `download` (used for file saving)
    // takes a slice of this Vec and copies it into an owned Vec as needed.
    pixel_vec: Vec<u16>,
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
            build_pass_bind_group_layout(&engine.device, 1, "Ingest Texture BGL");

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
            passes_cache: ShaderPassesCache::new(),
            scratch_pool: ScratchPool {
                width: 0,
                height: 0,
                textures: Vec::new(),
            },
            staging_buffer: None,
            present_tile_buffer: None,
            pixel_vec: Vec::new(),
        }
    }

    /// Converts sRGB-encoded values in `src_texture` to linear light, returning
    /// a new `Rgba16Float` texture. The source texture must be `Rgba16Unorm` as
    /// produced by `upload_texture`. Must be the first pass after `upload_texture`.
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
    pub fn present(&mut self, engine: &GpuEngine, src_texture: &wgpu::Texture) -> wgpu::Buffer {
        let max_binding = engine.device.limits().max_storage_buffer_binding_size;
        self.present_with_max_binding(engine, src_texture, max_binding)
    }

    /// Inner implementation of `present` that accepts an explicit binding-size
    /// limit. Tests use this to force tiling on small images.
    pub(crate) fn present_with_max_binding(
        &mut self,
        engine: &GpuEngine,
        src_texture: &wgpu::Texture,
        max_binding_size: u64,
    ) -> wgpu::Buffer {
        let (width, height) = (src_texture.width(), src_texture.height());
        let pixel_size: u64 = 8; // 4 channels × 2 bytes
        let full_size = width as u64 * height as u64 * pixel_size;

        let max_rows = (max_binding_size / (width as u64 * pixel_size)).min(height as u64) as u32;
        // The tile buffer must be large enough for the worst-case tile (max_rows rows).
        // The last tile may be smaller, but we only dispatch/copy `tile_size` bytes from
        // offset 0, so any unused space beyond `tile_size` is harmless.
        let max_tile_size = width as u64 * max_rows as u64 * pixel_size;
        let tile_buffer =
            Self::ensure_tile_buffer(&mut self.present_tile_buffer, engine, max_tile_size);

        // The output buffer is allocated fresh each call. The caller holds a
        // reference to it across the `present` + `download` sequence, so reusing
        // it across calls would overwrite memory the caller is still reading.
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

            // The params buffer is created fresh each tile. Using a cached buffer
            // updated via `queue.write_buffer` would not work here: `write_buffer`
            // is submitted immediately (before the command encoder's compute passes
            // run), so all tiles would see the last write's values. Per-tile
            // allocation of 16 bytes is cheap.
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
                tile_buffer,
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

    /// Ensures the cached tile buffer is allocated and large enough for the worst-case
    /// tile size of the current pass. Reused across tiles within a call and across
    /// subsequent calls to minimize GPU buffer allocation overhead.
    fn ensure_tile_buffer<'a>(
        present_tile_buffer: &'a mut Option<(wgpu::Buffer, u64)>,
        engine: &GpuEngine,
        max_tile_size: u64,
    ) -> &'a wgpu::Buffer {
        let needs_tile_alloc = match present_tile_buffer {
            Some((_, sz)) => *sz < max_tile_size,
            None => true,
        };

        if needs_tile_alloc {
            *present_tile_buffer = Some((
                engine.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("present_tile_buffer"),
                    size: max_tile_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                max_tile_size,
            ));
        }

        &present_tile_buffer.as_ref().unwrap().0
    }

    /// Ensures the cached staging buffer is allocated and large enough. Reused across
    /// `download_slice` calls to avoid per-call OS allocation of a large MAP_READ buffer.
    fn ensure_staging_buffer<'a>(
        staging_buffer: &'a mut Option<(wgpu::Buffer, u64)>,
        engine: &GpuEngine,
        buffer_size: u64,
    ) -> &'a wgpu::Buffer {
        let needs_alloc = match staging_buffer {
            Some((_, sz)) => *sz < buffer_size,
            None => true,
        };

        if needs_alloc {
            *staging_buffer = Some((
                engine.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("presentation_staging_buffer"),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                buffer_size,
            ));
        }

        &staging_buffer.as_ref().unwrap().0
    }

    /// Reads the output of `present` from a GPU storage buffer into `self.pixel_vec`,
    /// returning a borrowed `&[u16]` slice. No heap allocation occurs on the interactive
    /// path after the first call (the Vec capacity is retained across frames).
    ///
    /// The returned slice is borrowed from `self` and is valid until the next call that
    /// mutates `pixel_vec` (i.e., the next `download_slice` or `download` call).
    ///
    /// Use `bdip_core::gpu::texture::download_presentation_buffer` for one-shot callers
    /// (headless CLI, tests) that do not hold a `Renderer` across calls.
    pub fn download_slice(
        &mut self,
        engine: &GpuEngine,
        src_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Result<&[u16], crate::error::BdipError> {
        // Tightly packed: 4 channels × 2 bytes = 8 bytes per pixel, no row padding.
        let buffer_size = width as u64 * height as u64 * 8;

        let staging_buf =
            Self::ensure_staging_buffer(&mut self.staging_buffer, engine, buffer_size);

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(src_buffer, 0, staging_buf, 0, buffer_size);
        engine.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buf.slice(..buffer_size);
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
        let u16_data = bytemuck::cast_slice::<u8, u16>(&data);
        let pixel_count = u16_data.len();

        // Reuse the Vec's capacity across calls. `clear` retains the allocation;
        // `reserve` only reallocates if the new image is larger than any seen before.
        // `self.pixel_vec` is a separate field from `self.staging_buffer`; the
        // split-borrow above ensures the compiler accepts both borrows simultaneously.
        self.pixel_vec.clear();
        self.pixel_vec.reserve(pixel_count);
        self.pixel_vec.extend_from_slice(u16_data); // single memcpy, no allocation

        drop(data);
        staging_buf.unmap();

        Ok(&self.pixel_vec)
    }

    /// Downloads the output of `present` from a GPU storage buffer into an owned
    /// `Rgba16Image`. Delegates to `download_slice` and copies the slice into a
    /// fresh `Vec<u16>` — one allocation at call time, but isolated to code paths
    /// (file saving) where that cost is acceptable.
    ///
    /// For the interactive preview path, prefer `download_slice` to avoid the
    /// allocation entirely.
    pub fn download(
        &mut self,
        engine: &GpuEngine,
        src_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Result<crate::Rgba16Image, crate::error::BdipError> {
        let pixel_vec = self
            .download_slice(engine, src_buffer, width, height)?
            .to_vec();
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

    pub fn apply(
        &mut self,
        engine: &GpuEngine,
        src_texture: &wgpu::Texture,
        transform: &Transform,
    ) -> wgpu::Texture {
        let reg = registry_by_id(transform.shader_id)
            .unwrap_or_else(|| panic!("Unknown shader ID: '{}'", transform.shader_id));
        self.apply_passes(engine, src_texture, transform, reg, reg.meta.passes)
    }

    fn apply_passes(
        &mut self,
        engine: &GpuEngine,
        src_texture: &wgpu::Texture,
        transform: &Transform,
        reg: &'static crate::gpu::shaders::ShaderRegistration,
        passes: &[crate::gpu::shaders::PassDef],
    ) -> wgpu::Texture {
        let (width, height, depth) = (
            src_texture.width(),
            src_texture.height(),
            src_texture.depth_or_array_layers(),
        );

        self.sync_scratch_pool_dims(width, height);

        let borrowed_scratches = self.allocate_scratch_textures(
            engine,
            width,
            height,
            depth,
            transform.shader_id,
            passes,
        );

        let final_texture = engine.device.create_texture(&wgpu::TextureDescriptor {
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

        let params_buffer = Self::create_params_buffer(engine, reg, &transform.values);

        let cached_pipelines = self
            .passes_cache
            .get_or_create(&engine.device, transform.shader_id);

        Self::encode_transform_passes(
            engine,
            cached_pipelines,
            passes,
            src_texture,
            &final_texture,
            &borrowed_scratches,
            &params_buffer,
        );

        self.return_scratch_textures(borrowed_scratches);

        final_texture
    }

    fn sync_scratch_pool_dims(&mut self, width: u32, height: u32) {
        if self.scratch_pool.width != width || self.scratch_pool.height != height {
            self.scratch_pool.textures.clear();
            self.scratch_pool.width = width;
            self.scratch_pool.height = height;
        }
    }

    fn allocate_scratch_textures(
        &mut self,
        engine: &GpuEngine,
        width: u32,
        height: u32,
        depth: u32,
        shader_id: &str,
        passes: &[crate::gpu::shaders::PassDef],
    ) -> HashMap<&'static str, wgpu::Texture> {
        let mut borrowed = HashMap::new();
        for pass in passes {
            if let PassOutput::Scratch(name) = pass.output
                && let Entry::Vacant(e) = borrowed.entry(name)
            {
                let tex = self.scratch_pool.textures.pop().unwrap_or_else(|| {
                    let label = format!("{}::{}", shader_id, name);
                    engine.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(&label), // Tier 1: relabel on borrow (here at allocation)
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
                            | wgpu::TextureUsages::STORAGE_BINDING,
                        view_formats: &[],
                    })
                });
                e.insert(tex);
            }
        }
        borrowed
    }

    fn return_scratch_textures(&mut self, textures: HashMap<&'static str, wgpu::Texture>) {
        for (_, tex) in textures {
            self.scratch_pool.textures.push(tex);
        }
    }

    fn create_params_buffer(
        engine: &GpuEngine,
        reg: &crate::gpu::shaders::ShaderRegistration,
        values: &[f32],
    ) -> wgpu::Buffer {
        let uniform_bytes = (reg.make_uniform)(values);
        engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Apply Params Buffer"),
                contents: &uniform_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn encode_transform_passes(
        engine: &GpuEngine,
        pipelines: &[CompiledPass],
        passes: &[crate::gpu::shaders::PassDef],
        src_texture: &wgpu::Texture,
        final_texture: &wgpu::Texture,
        scratch_textures: &HashMap<&'static str, wgpu::Texture>,
        params_buffer: &wgpu::Buffer,
    ) {
        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let src_view = src_texture.create_view(&TextureViewDescriptor::default());
        let final_view = final_texture.create_view(&TextureViewDescriptor::default());

        let mut borrowed_views = HashMap::new();
        for (name, tex) in scratch_textures {
            borrowed_views.insert(*name, tex.create_view(&TextureViewDescriptor::default()));
        }

        let (width, height) = (src_texture.width(), src_texture.height());

        for (pass_idx, pass) in passes.iter().enumerate() {
            let pipeline = &pipelines[pass_idx];

            let mut texture_bind_group_entries = Vec::new();
            for (i, input) in pass.inputs.iter().enumerate() {
                let view = match input {
                    PassInput::Source => &src_view,
                    PassInput::Scratch(name) => borrowed_views.get(name).unwrap(),
                };
                texture_bind_group_entries.push(BindGroupEntry {
                    binding: i as u32,
                    resource: BindingResource::TextureView(view),
                });
            }

            let out_view = match pass.output {
                PassOutput::Final => &final_view,
                PassOutput::Scratch(name) => borrowed_views.get(name).unwrap(),
            };

            texture_bind_group_entries.push(BindGroupEntry {
                binding: pass.inputs.len() as u32,
                resource: BindingResource::TextureView(out_view),
            });

            let texture_bind_group = engine.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Apply Texture Bind Group"),
                layout: &pipeline.texture_bind_group_layout,
                entries: &texture_bind_group_entries,
            });

            let params_bind_group = engine.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Apply Params Bind Group"),
                layout: &pipeline.params_bind_group_layout,
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
                cpass.set_pipeline(&pipeline.pipeline);
                cpass.set_bind_group(0, &texture_bind_group, &[]);
                cpass.set_bind_group(1, &params_bind_group, &[]);
                cpass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
            }
        }

        engine.queue.submit(Some(encoder.finish()));
    }

    #[cfg(test)]
    pub(crate) fn scratch_pool_info(&self) -> ((u32, u32), usize) {
        (
            (self.scratch_pool.width, self.scratch_pool.height),
            self.scratch_pool.textures.len(),
        )
    }

    /// Returns the pool's current dimensions and the texture at `index` in the
    /// free list. Used by tests to assert both the active pool size and that the
    /// same physical GPU allocation is reused across runs (pointer identity is a
    /// stronger claim than "a texture of the same shape exists").
    #[cfg(test)]
    pub(crate) fn scratch_pool_handle(&self, index: usize) -> ((u32, u32), Option<&wgpu::Texture>) {
        let dims = (self.scratch_pool.width, self.scratch_pool.height);
        let tex = self.scratch_pool.textures.get(index);
        (dims, tex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shaders::Transform;
    use crate::gpu::test_util::{make_solid_image, roundtrip};
    use crate::gpu::texture::{download_presentation_buffer, upload_texture};

    // ========== Chaining tests ==========

    #[test]
    fn test_same_shader_chaining() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 51400/65535 ≈ 0.784 sRGB → ~0.577 linear
        let img = make_solid_image(2, 2, 51400, 51400, 51400);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "brightness",
                    values: vec![-0.2],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.5],
                },
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
    fn test_cross_shader_chaining() {
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
                Transform {
                    shader_id: "brightness",
                    values: vec![0.3],
                },
                Transform {
                    shader_id: "contrast",
                    values: vec![0.5],
                },
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

    // ========== Ingest/Present roundtrip and tiling tests ==========

    #[test]
    fn test_ingest_present_roundtrip() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

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
        let mut renderer = Renderer::new(&engine);

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
        let mut renderer = Renderer::new(&engine);

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
        let mut renderer = Renderer::new(&engine);

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
        let mut renderer = Renderer::new(&engine);

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

    // ========== Scratch pool behavior tests ==========

    /// A single-pass shader must never allocate from the scratch pool; after `apply`
    /// the free list must still be empty.
    #[test]
    fn test_single_pass_skips_scratch_pool() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let _ = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "brightness",
                values: vec![0.0],
            }],
        );

        let (pool_dims, pool_len) = renderer.scratch_pool_info();
        assert_eq!(pool_dims, (4, 4), "pool dims should match the image dims");
        assert_eq!(
            pool_len, 0,
            "single-pass shader must not allocate scratch textures"
        );
    }

    // ========== ShaderPassesCache behavior tests ==========

    #[test]
    fn test_passes_cache_returns_same_pipeline() {
        let engine = GpuEngine::new().unwrap();
        let mut cache = ShaderPassesCache::new();

        // Calling get_or_create twice for the same shader_id must return the same
        // cached entry — no recompilation on the second call.
        let p1 = cache.get_or_create(&engine.device, "brightness").as_ptr();
        let p2 = cache.get_or_create(&engine.device, "brightness").as_ptr();
        assert!(
            std::ptr::eq(p1, p2),
            "same shader_id should return the same cached pipeline pointer"
        );
    }

    #[test]
    fn test_passes_cache_different_kinds() {
        let engine = GpuEngine::new().unwrap();
        let mut cache = ShaderPassesCache::new();

        // Brightness and Saturation must occupy separate cache entries.
        let pb = cache.get_or_create(&engine.device, "brightness").as_ptr();
        let ps = cache.get_or_create(&engine.device, "saturation").as_ptr();
        assert!(
            !std::ptr::eq(pb, ps),
            "different shader IDs should return different pipeline pointers"
        );
    }

    /// A single-pass shader must compile to a `Vec<CompiledPass>` of length 1.
    /// A second `get_or_create` call must return the same cached vec (pointer-stable).
    #[test]
    fn test_pipeline_cache_compiles_per_pass() {
        let engine = GpuEngine::new().unwrap();
        let mut cache = ShaderPassesCache::new();

        let passes_first = cache.get_or_create(&engine.device, "brightness");
        assert_eq!(
            passes_first.len(),
            1,
            "single-pass shader must compile to 1 CompiledPass"
        );

        // Second call must return the same cached slice — no recompilation.
        let ptr_first = passes_first.as_ptr();
        let passes_second = cache.get_or_create(&engine.device, "brightness");
        assert!(
            std::ptr::eq(ptr_first, passes_second.as_ptr()),
            "second get_or_create must return the same cached vec"
        );
    }

    /// A multi-pass shader (Clarity, 3 passes) must compile to a `Vec<CompiledPass>`
    /// of length 3.
    #[test]
    fn test_pipeline_cache_compiles_per_pass_multi() {
        let engine = GpuEngine::new().unwrap();
        let mut cache = ShaderPassesCache::new();

        let passes = cache.get_or_create(&engine.device, "clarity");
        assert_eq!(
            passes.len(),
            3,
            "Clarity must compile to exactly 3 CompiledPass entries"
        );
    }

    /// Changing image dimensions must drop all scratch textures from the pool.
    #[test]
    fn test_multi_pass_image_resize_drops_pool() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let small = make_solid_image(4, 4, 32767, 32767, 32767);
        let _ = roundtrip(
            &mut renderer,
            &engine,
            &small,
            &[Transform {
                shader_id: "clarity",
                values: vec![0.0],
            }],
        );
        // Pool should hold the 2 scratch textures from Clarity's pass list.
        let (pool_dims, pool_len) = renderer.scratch_pool_info();
        assert_eq!(
            pool_dims,
            (4, 4),
            "pool dims must match image dims after first run"
        );
        assert_eq!(
            pool_len, 2,
            "pool must hold 2 scratch textures after first run"
        );

        // Run at a different image size — the pool must be reset.
        let large = make_solid_image(8, 8, 32767, 32767, 32767);
        let _ = roundtrip(
            &mut renderer,
            &engine,
            &large,
            &[Transform {
                shader_id: "clarity",
                values: vec![0.0],
            }],
        );
        let (pool_dims, pool_len) = renderer.scratch_pool_info();
        assert_eq!(
            pool_dims,
            (8, 8),
            "pool dims must update to new image dims after resize"
        );
        assert_eq!(
            pool_len, 2,
            "new pool dims must hold 2 scratch textures after resize"
        );
    }

    /// Running a multi-pass shader twice at the same image dimensions must reuse
    /// the same physical scratch textures. After each run the pool's free list holds
    /// exactly 2 textures, and the set of pointer addresses must be identical on the
    /// second run (same GPU allocations, not new ones).
    #[test]
    fn test_multi_pass_scratch_pool_reuses_across_runs() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "clarity",
            values: vec![0.0],
        };
        let dims = (8u32, 8u32);

        roundtrip(
            &mut renderer,
            &engine,
            &img,
            std::slice::from_ref(&transform),
        );
        let (pool_dims, pool_len) = renderer.scratch_pool_info();
        assert_eq!(
            pool_dims, dims,
            "pool dims must match image dims after first run"
        );
        assert_eq!(
            pool_len, 2,
            "pool must hold 2 scratch textures after first run"
        );

        let (_, tex0_after_first) = renderer.scratch_pool_handle(0);
        let (_, tex1_after_first) = renderer.scratch_pool_handle(1);
        let ptr0_after_first = tex0_after_first.unwrap() as *const wgpu::Texture;
        let ptr1_after_first = tex1_after_first.unwrap() as *const wgpu::Texture;

        roundtrip(
            &mut renderer,
            &engine,
            &img,
            std::slice::from_ref(&transform),
        );
        let (pool_dims, pool_len) = renderer.scratch_pool_info();
        assert_eq!(
            pool_dims, dims,
            "pool dims must match image dims after second run"
        );
        assert_eq!(
            pool_len, 2,
            "pool must still hold 2 scratch textures after second run"
        );

        let (_, tex0_after_second) = renderer.scratch_pool_handle(0);
        let (_, tex1_after_second) = renderer.scratch_pool_handle(1);
        let ptr0_after_second = tex0_after_second.unwrap() as *const wgpu::Texture;
        let ptr1_after_second = tex1_after_second.unwrap() as *const wgpu::Texture;

        // The set of pointer values must be the same (same physical textures
        // reused), though the pool order may differ between runs.
        let ptrs_first: std::collections::HashSet<_> = [ptr0_after_first, ptr1_after_first].into();
        let ptrs_second: std::collections::HashSet<_> =
            [ptr0_after_second, ptr1_after_second].into();
        assert_eq!(
            ptrs_first, ptrs_second,
            "same physical scratch textures must be reused on the second run"
        );
    }

    // ========== Performance benchmark ==========

    /// Times the GPU-critical path on a 24 MP synthetic image — the primary target
    /// size from perf_goal_1.md.
    ///
    /// Two runs are measured to isolate warm-pipeline performance from one-time
    /// startup costs:
    ///
    /// - **Run 1 (cold)**: upload → ingest → apply → present → download.
    ///   Includes shader compilation and initial staging buffer allocation.
    ///   Uses `download` (returns owned `Rgba16Image`) since this is the
    ///   first call and primes both the staging buffer and pixel_vec caches.
    /// - **Run 2 (warm)**: apply → present → download_slice, reusing the
    ///   ingested base texture, cached staging buffer, and pixel_vec from
    ///   run 1. This matches the interactive editing path
    ///   (`presentation_to_handle` → `download_slice`), where upload+ingest
    ///   happen once and are cached across slider changes.
    ///
    /// Run 2 is the number to compare against the 8–20 ms target.
    ///
    /// This test is ignored by default so it does not run in CI. Run it manually:
    ///   cargo perf-test
    ///
    /// Measurements on Apple M4 Pro (2026-04) after PR 4 (GPU Upload Conversion):
    ///   gpu upload:              ~14.81 ms  (Raw u16 upload; no CPU conversion)
    ///   run 1 execute:           ~1.61 ms
    ///   run 1 readback:          ~70.85 ms
    ///   run 1 critical path:     ~72.46 ms
    ///   run 2 execute:           ~0.42 ms
    ///   run 2 readback:          ~16.77 ms (reused staging buffer + pixel_vec)
    ///   run 2 critical path:     ~17.19 ms
    ///
    /// Target once all known bottlenecks are resolved (warm, interactive): <20 ms.
    /// When the target is reliably met, add an assertion and remove #[ignore].
    #[test]
    #[ignore]
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
        let transformed_1 = renderer.apply(
            &engine,
            &ingested,
            &Transform {
                shader_id: "brightness",
                values: vec![0.1],
            },
        );
        let present_buf_1 = renderer.present(&engine, &transformed_1);
        let execute_ms_1 = t_execute_1.elapsed().as_secs_f64() * 1000.0;

        let t_readback_1 = Instant::now();
        let _result_1 = renderer
            .download(&engine, &present_buf_1, img.width(), img.height())
            .unwrap();
        let readback_ms_1 = t_readback_1.elapsed().as_secs_f64() * 1000.0;
        let critical_path_1 = execute_ms_1 + readback_ms_1;

        // --- Run 2: warm (shaders compiled, staging buffer reused) ---
        // Reuses `ingested` directly — matching interactive editing where the base
        // texture is cached in VRAM across slider changes.
        let t_execute_2 = Instant::now();
        let transformed_2 = renderer.apply(
            &engine,
            &ingested,
            &Transform {
                shader_id: "brightness",
                values: vec![0.1],
            },
        );
        let present_buf_2 = renderer.present(&engine, &transformed_2);
        let execute_ms_2 = t_execute_2.elapsed().as_secs_f64() * 1000.0;

        let t_readback_2 = Instant::now();
        let _result_2 = renderer
            .download_slice(&engine, &present_buf_2, img.width(), img.height())
            .unwrap();
        let readback_ms_2 = t_readback_2.elapsed().as_secs_f64() * 1000.0;
        let critical_path_2 = execute_ms_2 + readback_ms_2;

        eprintln!("--- 24 MP GPU roundtrip ---");
        eprintln!("  gpu upload:                      {:>8.2} ms", upload_ms);
        eprintln!(
            "  run 1 execute (ingest+apply+present): {:>8.2} ms",
            execute_ms_1
        );
        eprintln!(
            "  run 1 readback (download):       {:>8.2} ms",
            readback_ms_1
        );
        eprintln!(
            "  run 1 critical path:             {:>8.2} ms",
            critical_path_1
        );
        eprintln!(
            "  run 2 execute (apply+present):   {:>8.2} ms",
            execute_ms_2
        );
        eprintln!(
            "  run 2 readback (download_slice): {:>8.2} ms",
            readback_ms_2
        );
        eprintln!(
            "  run 2 critical path:             {:>8.2} ms  (target: <25 ms warm)",
            critical_path_2
        );
        eprintln!("----------------------------------");

        assert!(
            upload_ms < 25.0,
            "Upload time exceeded 25ms target: {:.2}ms",
            upload_ms
        );
        assert!(
            critical_path_1 < 80.0,
            "Run 1 (cold) critical path exceeded 80ms target: {:.2}ms",
            critical_path_1
        );
        assert!(
            critical_path_2 < 25.0,
            "Run 2 (warm) critical path exceeded 25ms target: {:.2}ms",
            critical_path_2
        );
    }

    /// Times the GPU critical path on a 24 MP image with the Clarity multi-pass
    /// shader (3 passes: blur_h, blur_v, combine). Run 2 (warm) is the interactive
    /// editing latency target — the number to compare against the 25 ms budget.
    ///
    /// Run manually via:
    ///   cargo perf-test
    ///   cargo test --release -p bdip_core -- --ignored test_perf_gpu_roundtrip_24mp_clarity
    #[test]
    #[ignore]
    fn test_perf_gpu_roundtrip_24mp_clarity() {
        use std::time::Instant;

        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(5000, 4800, 32767, 32767, 32767);

        let uploaded = upload_texture(&engine.device, &engine.queue, &img);

        // --- Run 1: cold (shader compilation + initial buffer allocation) ---
        let t_execute_1 = Instant::now();
        let ingested = renderer.ingest(&engine, &uploaded);
        let transformed_1 = renderer.apply(
            &engine,
            &ingested,
            &Transform {
                shader_id: "clarity",
                values: vec![0.5],
            },
        );
        let present_buf_1 = renderer.present(&engine, &transformed_1);
        let execute_ms_1 = t_execute_1.elapsed().as_secs_f64() * 1000.0;

        let t_readback_1 = Instant::now();
        let _result_1 = renderer
            .download(&engine, &present_buf_1, img.width(), img.height())
            .unwrap();
        let readback_ms_1 = t_readback_1.elapsed().as_secs_f64() * 1000.0;
        let critical_path_1 = execute_ms_1 + readback_ms_1;

        // --- Run 2: warm (pipelines compiled, staging buffer reused) ---
        let t_execute_2 = Instant::now();
        let transformed_2 = renderer.apply(
            &engine,
            &ingested,
            &Transform {
                shader_id: "clarity",
                values: vec![0.5],
            },
        );
        let present_buf_2 = renderer.present(&engine, &transformed_2);
        let execute_ms_2 = t_execute_2.elapsed().as_secs_f64() * 1000.0;

        let t_readback_2 = Instant::now();
        let _result_2 = renderer
            .download_slice(&engine, &present_buf_2, img.width(), img.height())
            .unwrap();
        let readback_ms_2 = t_readback_2.elapsed().as_secs_f64() * 1000.0;
        let critical_path_2 = execute_ms_2 + readback_ms_2;

        eprintln!("--- 24 MP GPU roundtrip — Clarity ---");
        eprintln!(
            "  run 1 execute (ingest+apply+present): {:>8.2} ms",
            execute_ms_1
        );
        eprintln!(
            "  run 1 readback:                  {:>8.2} ms",
            readback_ms_1
        );
        eprintln!(
            "  run 1 critical path:             {:>8.2} ms",
            critical_path_1
        );
        eprintln!(
            "  run 2 execute (apply+present):   {:>8.2} ms",
            execute_ms_2
        );
        eprintln!(
            "  run 2 readback:                  {:>8.2} ms",
            readback_ms_2
        );
        eprintln!(
            "  run 2 critical path:             {:>8.2} ms  (target: <25 ms warm)",
            critical_path_2
        );
        eprintln!("--------------------------------------");

        assert!(
            critical_path_2 < 25.0,
            "Clarity warm critical path exceeded 25 ms target: {:.2} ms",
            critical_path_2
        );
    }

    /// Times the GPU critical path on a 24 MP image with the Cartoon multi-pass
    /// shader (5 passes: smooth_h, smooth_v, quantize, edges, combine). Run 2 (warm)
    /// is the interactive editing latency target — compare against the 25 ms budget.
    ///
    /// Run manually via:
    ///   cargo perf-test
    ///   cargo test --release -p bdip_core -- --ignored test_perf_gpu_roundtrip_24mp_cartoon
    #[test]
    #[ignore]
    fn test_perf_gpu_roundtrip_24mp_cartoon() {
        use std::time::Instant;

        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(5000, 4800, 32767, 32767, 32767);

        let uploaded = upload_texture(&engine.device, &engine.queue, &img);

        // Cartoon default params: strength=0.0, levels=8.0, edge_threshold=0.15,
        // edge_softness=0.10, edge_darkness=1.0. Using defaults exercises the full
        // 5-pass pipeline under realistic conditions without assuming specific output.
        let cartoon_values = vec![0.0f32, 8.0, 0.15, 0.10, 1.0];

        // --- Run 1: cold (shader compilation + initial buffer allocation) ---
        let t_execute_1 = Instant::now();
        let ingested = renderer.ingest(&engine, &uploaded);
        let transformed_1 = renderer.apply(
            &engine,
            &ingested,
            &Transform {
                shader_id: "cartoon",
                values: cartoon_values.clone(),
            },
        );
        let present_buf_1 = renderer.present(&engine, &transformed_1);
        let execute_ms_1 = t_execute_1.elapsed().as_secs_f64() * 1000.0;

        let t_readback_1 = Instant::now();
        let _result_1 = renderer
            .download(&engine, &present_buf_1, img.width(), img.height())
            .unwrap();
        let readback_ms_1 = t_readback_1.elapsed().as_secs_f64() * 1000.0;
        let critical_path_1 = execute_ms_1 + readback_ms_1;

        // --- Run 2: warm (pipelines compiled, staging buffer reused) ---
        let t_execute_2 = Instant::now();
        let transformed_2 = renderer.apply(
            &engine,
            &ingested,
            &Transform {
                shader_id: "cartoon",
                values: cartoon_values,
            },
        );
        let present_buf_2 = renderer.present(&engine, &transformed_2);
        let execute_ms_2 = t_execute_2.elapsed().as_secs_f64() * 1000.0;

        let t_readback_2 = Instant::now();
        let _result_2 = renderer
            .download_slice(&engine, &present_buf_2, img.width(), img.height())
            .unwrap();
        let readback_ms_2 = t_readback_2.elapsed().as_secs_f64() * 1000.0;
        let critical_path_2 = execute_ms_2 + readback_ms_2;

        eprintln!("--- 24 MP GPU roundtrip — Cartoon ---");
        eprintln!(
            "  run 1 execute (ingest+apply+present): {:>8.2} ms",
            execute_ms_1
        );
        eprintln!(
            "  run 1 readback:                  {:>8.2} ms",
            readback_ms_1
        );
        eprintln!(
            "  run 1 critical path:             {:>8.2} ms",
            critical_path_1
        );
        eprintln!(
            "  run 2 execute (apply+present):   {:>8.2} ms",
            execute_ms_2
        );
        eprintln!(
            "  run 2 readback:                  {:>8.2} ms",
            readback_ms_2
        );
        eprintln!(
            "  run 2 critical path:             {:>8.2} ms  (target: <25 ms warm)",
            critical_path_2
        );
        eprintln!("--------------------------------------");

        assert!(
            critical_path_2 < 25.0,
            "Cartoon warm critical path exceeded 25 ms target: {:.2} ms",
            critical_path_2
        );
    }
}
