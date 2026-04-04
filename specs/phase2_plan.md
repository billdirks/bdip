# Phase 2: GPU Pipeline & Headless CLI — Implementation Plan

This plan advances the `bdip` application to Phase 2 as detailed in the `specification.md` documentation. The goal is to establish the WebGPU rendering framework and a headless command-line interface capable of executing the first transformation.

## Scope & Goals

**Goals:**
*   **Performance First:** Per the specification, GPU execution and CPU-to-GPU transfers must be hyper-optimized. Minimizing buffer copies and ensuring high-efficiency parallel execution are primary goals of this phase.

**In scope:** Headless WebGPU (`wgpu`) initialization, CPU-to-GPU texture memory transfers, the first WGSL compute shader (Brightness), `clap` argument parsing, and an end-to-end processing pipeline executing from the terminal.
**Out of scope:** `iced` UI integration, region selection, and remaining V1 filters (Contrast, Saturation, etc. will easily follow the pattern established here).

---

## 1. Dependency Definitions

### `bdip_core/Cargo.toml`
Add WebGPU and buffer-casting dependencies:
```toml
wgpu = "0.20"
pollster = "0.3" # For blocking async GPU setup code in synchronous headless contexts
bytemuck = { version = "1.0", features = ["derive"] } # Safely cast structs to byte buffers for GPU uniform data
```

### `bdip/Cargo.toml`
Add CLI parsing framework:
```toml
clap = { version = "4", features = ["derive"] }
```

---

## 2. Core Library: Headless GPU Engine (`bdip_core/src/gpu`)

We will create a new nested module `bdip_core/src/gpu/` and expose it.

### [NEW] `bdip_core/src/gpu/engine.rs`
**Responsibility:** Managing the WebGPU connection context securely.
- Expose a `GpuEngine` struct containing the `wgpu::Instance`, `wgpu::Adapter`, `wgpu::Device`, and `wgpu::Queue`.
- Implement `GpuEngine::new()` using `pollster::block_on` to request the device (preferring high-performance backends, e.g., Metal).

### [NEW] `bdip_core/src/gpu/texture.rs`
**Responsibility:** CPU↔GPU Data Transcoding. 
- Implement mapping functions that take the `image::RgbaImage` provided by Phase 1's `io` module, and encode it into a `wgpu::Texture` object using WebGPU's strict buffer copying requirements.
- The textures will be requested with `Rgba16Float` format for high dynamic range precision as mandated by the architectural spec.

### [NEW] `bdip_core/src/gpu/shader.wgsl`
**Responsibility:** WebGPU Shader Language Compute Program.
- Write a highly parallelized 2D compute shader that reads an input texture, decodes the pixel coordinates, applies the brightness offset uniformly to the RGB channels (ignoring Alpha), and writes to the output texture.

### [NEW] `bdip_core/src/gpu/pipeline.rs`
**Responsibility:** The execution conductor.
- Implements `Renderer::apply_brightness(texture, brightness_value)`.
- Defines the `wgpu::ComputePipeline`, binds the pipeline layouts, pushes the `uniform` parameter (the `f32` normalized brightness value), and dispatches workgroups optimally based on the image bounds.

---

## 3. Application Binary: CLI Routing (`bdip/src`)

### [NEW] `bdip/src/cli.rs`
**Responsibility:** Using `clap` to rigorously parse the terminal command.
- `input`: The positional argument specifying the source image.
- `--headless`: Flag specifying the application should skip `iced` window generation.
- `--output` (`-o`): Destination filepath.
- `--apply` (`-a`): Allow repeated values (e.g. `--apply brightness:0.5 -a invert`).
- `--pipeline` (`-p`): Target a text file containing line-by-line parameters.

### [MODIFY] `bdip/src/main.rs`
**Responsibility:** Execute the coordinated workflow.
- Map the parsed flags from `cli.rs`.
- Read the image using `bdip_core::io::load_image()`.
- Spawn `GpuEngine`.
- Parse the `--apply` payload into `Transformation::Brightness(0.5)`. Upload texture, run pipeline.
- Save out the results via `bdip_core::io::save_image()`.

---

## 4. Verification Plan

It is requested that unit tests accompany precisely all new code logic alongside an end-to-end verifiable test.

### 4.1. Unit Testing Strategy

**`bdip_core` tests (The Math and GPU)**:
1. `test_gpu_headless_init`: Initialize the `wgpu` engine completely headlessly. Prove it succeeds returning a valid context and doesn't fatally crash without a window server.
2. `test_brightness_shader_positive`: Passes `brightness: 0.5` against a 50% gray `[127, 127, 127, 255]` 2x2 image. Asserts red, green, and blue clamp successfully to maximum brightness.
3. `test_brightness_shader_negative`: Passes `brightness: -0.5` against a 50% gray image. Asserts values correctly pull down mathematically to pure black `[0, 0, 0, 255]`.
4. `test_brightness_shader_zero`: Passes `brightness: 0.0`. Asserts the output buffer perfectly matches the input buffer (identity/no-op).
5. `test_shader_chaining`: Applies `Brightness(-0.2)` followed by `Brightness(0.5)` sequentially on the GPU buffer without downloading to CPU in between, verifying that pipelined transformations persist intermediate floating point data natively.

**`bdip` tests (The Binary)**:
1. `test_cli_argument_parser`: Use `clap::CommandFactory::debug_assert` and pass synthetic command line arrays ensuring arguments map securely into the struct.

### 4.2. End-to-End Output Assessment (Verified When Criteria)

As designated by the Phase 2 specification ("*Produces a visibly brighter output image*"), we will add a robust end-to-end integration test (`tests/e2e_cli_pipeline.rs`) to the `bdip` crate.

**Test Procedure Mechanism**:
Following the testing pyramid, parameter permutations will be handled via the isolated core unit tests. We will use a single robust E2E test to prove the complete binary pipeline natively connects CLI parsing to file output:
1. Programmatically write a completely black (`rgb(0,0,0)`) 16x16 dummy test file (`test_in.jpg`) to a temporary directory.
2. Spawn the binary executable via `std::process::Command` executing the stacked pipeline: 
   `$ bdip --headless test_in.jpg --output test_out.png --apply brightness:-0.2 --apply brightness:0.5`
3. Wait on the child process and assert the exit code is `0`.
4. Assert that `test_out.png` was safely encoded mechanistically to disk.
5. Decode the `test_out.png` result, calculate its average luminance, and assert rigorously that it holds higher values compared to the initial payload (proving a net positive brightness shift survived the end-to-end multi-transform pipeline).
