# Current Performance Analysis

This document analyses the performance characteristics of `bdip` in its headless mode benchmark against the theoretical limits established in `specs/perf_goal_1.md`. 

Sample data was collected the branch `temp-perf` by running:

```
time cargo headless-release Images/vivid-P9220006.tif --output out.tif --apply saturation:0.5
```

## 1. How far are we from perf goals?

The performance goal specifies **~8–20ms total latency** for the critical path of transformation compute dispatch + readback on Apple Silicon for a 24MP image. 

The measured values from the latest headless release run (`vivid-P9220006.tif` with `saturation:0.5`):

| Category | Measured Time | Target (`perf_goal_1.md`) | Variance |
|----------|---------------|---------------------------|----------|
| **GPU Pipeline Execution** | 45.78 ms | 5–15 ms | **~3× to ~9× too slow** |
| **Buffer Readback** | 28.22 ms | 1–4 ms | **~7× to ~28× too slow** |
| **Critical Path Total** | **74.00 ms** | **8–20 ms** | **~3.7× to 9.2× too slow** |

Note: The `Texture Upload` (62.69 ms) and disk I/O are not strictly counted against the interactive editing latency goal because images are generally cached in VRAM for live editing, but they nonetheless represent a significant bottleneck in our overall CLI throughput.

## 2. Are the performance goals feasible or did we make bad assumptions?

**The goals are completely feasible and our UMA (Unified Memory Architecture) assumptions are correct.** Apple Silicon is easily capable of hitting 5-15ms for simple pixel maths over 24MP, and < 4ms for synchronising mapped memory.

Our divergence comes entirely from naive implementation details, not hardware or WebGPU limitations. The headless CLI benchmark performs a strictly **"cold" execution**, meaning it pays all initialisation, compilation, and allocation costs up front during the timed run. By contrast, achieving the theoretical interactive `8–20ms` limit requires a "warm" execution loop where memory is pre-allocated and shaders are pre-compiled.

## 3. Obvious shortcomings in our current implementation

The codebase currently treats every pass as a one-off execution, paying massive hidden overheads. 

### A. Hot-Loop Buffer & Texture Allocation (Impacts Execution & Readback)
In the GPU pipeline, `wgpu::Texture` and `wgpu::Buffer` backing objects are allocated dynamically on the fly:
- `apply()` creates a new `wgpu::Texture` for its output on every call.
- `present()` allocates a 192MB `wgpu::Buffer` (`output_buffer`) and sub-buffers (`tile_buffer`) on every call.
- `download_presentation_buffer()` allocates another 192MB `wgpu::Buffer` (`staging_buffer`) for readback mapping, requesting new memory from the OS every time.

**Fix:** A warm pipeline should allocate a pool of ping-pong VRAM textures and fixed staging buffers when the image is first loaded, and reuse them directly for every dispatch.

### B. Cold-Start Shader Compilation (Impacts Execution)
In `bdip_core/src/gpu/pipeline.rs`, we use a lazy `PipelineCache`. The very first time `Renderer::apply()` is called for `Saturation`, it compiles the WGSL shader into a Metal pipeline synchronously. 
Parsing WGSL and compiling the pipeline easily accounts for ~20-35ms of the observed 45ms "Execution" overhead in headless mode. 

**Fix:** This overhead disappears naturally in a UI loop after the first slider interaction (it becomes warm), but for headless throughput and fast UI initialization, we should evaluate eager/background precompilation or pipeline caching (`wgpu` pipeline caches).

### C. CPU Copying on Readback (Impacts Readback)
`specs/perf_goal_1.md` correctly notes UMA readback is near-free, but our implementation in `download_presentation_buffer` explicitly breaks zero-copy:
```rust
let pixel_vec: Vec<u16> = bytemuck::cast_slice::<u8, u16>(&data).to_vec();
```
`.to_vec()` allocates a fresh 192MB buffer in CPU memory and performs a synchronous memory block copy across it. This CPU RAM allocation + memcpy + wgpu sync overhead is what drives the 28ms readback time.

**Fix:** We shouldn't use `to_vec()`. We need an abstraction where `Rgba16Image` can leverage the mapped `wgpu` memory directly, or where the `MAP_READ` buffer provides memory straight to the UI/disk without interim CPU allocations.

### D. Single-Threaded CPU Pre-Processing (Impacts Upload)
While not part of the interactive preview goal, the `Texture Upload` takes an abysmal 62ms. We loop over 24 million pixels on a single CPU thread to perform floating-point divisions (`pixel[n] as f32 / 65535.0`), cast to `f16`, and construct a massive `Vec<PixelF16>` before calling `write_texture`.

**Fix:** 
1. Use `rayon` to trivially parallelise CPU conversion loops if they are required.
2. Even better, upload raw `u16` buffers and modify the `ingest` compute shader to perform the `u16` -> `f16` conversion natively on the GPU where the bandwidth and ALUs are vastly superior.
