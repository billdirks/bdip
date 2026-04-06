# Technical Debt & Refactoring Tracker

This document tracks known architectural shortcuts, generic naming, and structural issues that were accepted during early iterations (Phase 1 & 2) to maintain velocity. 

> [!CAUTION]
> **To Future AI Contexts / Agents:** The remediations proposed below are *suggestions* based on the state of the codebase at the time they were documented. Do NOT execute them blindly. Always cross-reference these suggestions with the *current* state of the code and evaluate if a better, more modern pattern is appropriate before executing refactors.

## Generalization Blockers (GPU Pipeline)

### Generic Parameter Structs
- **Location:** `bdip_core/src/gpu/pipeline.rs`
- **Current Pattern:** The GPU parameter mapping struct is generically named `ParamsUniform`.
- **Refactor Goal:** Prevent a "Monster Struct" pattern where a single giant struct holds unused parameters for *all* possible transformations, which wastes GPU memory alignment space and causes developer confusion.
- **Suggested Remediation:** Rename `ParamsUniform` to `BrightnessUniform` (or `BrightnessParams`). Moving forward, create dedicated, tightly packed, and specifically named parameter structs for *each* discrete transformation (e.g., `ContrastUniform`, `SaturationUniform`).

### Generic Shader Naming
- **Location:** `bdip_core/src/gpu/shader.wgsl`
- **Current Pattern:** The core compute shader file is generically named `shader.wgsl` despite only handling the Brightness algorithm.
- **Refactor Goal:** Avoid confusion and filename collision when managing multiple shaders handling different calculations on the GPU.
- **Suggested Remediation:** Rename `shader.wgsl` to something explicitly descriptive like `brightness.wgsl`. Future shaders should follow this specific isolated naming convention (`contrast.wgsl`, `saturation.wgsl`) to keep the pipeline renderer modular.

### Monolithic Pipeline Initialization
- **Location:** `bdip_core/src/gpu/pipeline.rs` (`Renderer::new`)
- **Current Pattern:** The `Renderer::new` constructor eagerly compiles the shader module and statically generates the `ComputePipeline` directly inside the startup sequence.
- **Refactor Goal:** Prevent a "Monster Initialization" bottleneck. As the application grows to support dozens of transformations, globally compiling all shaders and building all pipelines at startup will aggressively bottleneck application launch times and waste GPU resources for transformations the user may never actually click.
- **Suggested Remediation:** Refactor the `Renderer` (or create a dedicated `PipelineCache`) to implement a **Lazy-Loading** pattern. Shaders and their corresponding pipelines should safely JIT (Just-In-Time) compile upon their first invocation during an `apply_*` call, and then be stored in a HashMap/cache for rapid re-use.

## UI Responsiveness (GPU Pipeline)

### Synchronous Readback Blocks UI Thread
- **Location:** `bdip_core/src/gpu/texture.rs` (`download_texture`), call sites in `bdip/src/`
- **Current Pattern:** `download_texture()` calls `device.poll(wait_indefinitely)`, which blocks the
  calling thread until the GPU finishes and the PCIe transfer (on discrete GPUs) completes. If called
  on the UI thread, the window freezes for the duration of the operation.
- **Risk:** On Apple Silicon (UMA) this is 1–4 ms and imperceptible. On discrete GPU hardware with
  large images (24MP+), it can approach 15–20 ms per edit — noticeable as stutter during rapid edits.
- **Suggested Remediation:** Move the full pipeline invocation (compute shader dispatch +
  `download_texture`) onto a background thread using `std::thread::spawn`. `wgpu::Device` and
  `wgpu::Queue` are `Send + Sync` and can be moved freely. Send the resulting `RgbaImage` back to
  the UI thread via an `std::sync::mpsc` channel. The UI thread polls the channel and refreshes the
  image widget when the result arrives, remaining fully responsive during processing. No `async/await`
  or external async runtime is required.
- **Priority:** Low for V1 (Apple Silicon target, typical image sizes are fast). Revisit before
  shipping on non-Apple or supporting very large (50MP+) images.

## Image I/O (Precision)

### [IMMEDIATE FIX] 8-bit Downsampling Trap
- **Location:** `bdip_core/src/io.rs` (`load_image`)
- **Current Pattern:** The `load_image` function explicitly calls `img.to_rgba8()`.
- **Risk:** This is a "silent quality killer." High-end DSLR and mirrorless camera exports (16-bit
  TIFFs) are immediately downsampled to 8-bit integers upon loading. This destroys the extra
  precision and dynamic range before the GPU pipeline even starts, defeating the purpose of our
  internal `Rgba16Float` engine.
- **Suggested Remediation:** Update `load_image` to use `to_rgba16()`. This will require updating
  the `bdip_core` GPU texture upload logic to accept `u16` buffers and correctly map them to the
  `wgpu` texture format without narrowing the data.
- **Priority:** **High / Immediate.** This should be resolved before Phase 4 full UI integration to
  ensure we are actually delivering the promised "commercial-beating" image quality.

## Future Considerations
*(Add new items here as they are discovered during development)*
