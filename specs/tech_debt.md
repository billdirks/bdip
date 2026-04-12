# Technical Debt & Refactoring Tracker

This document tracks known architectural shortcuts, generic naming, and structural issues that were accepted during early iterations (Phase 1 & 2) to maintain velocity. 

> [!CAUTION]
> **To Future AI Contexts / Agents:** The remediations proposed below are *suggestions* based on the state of the codebase at the time they were documented. Do NOT execute them blindly. Always cross-reference these suggestions with the *current* state of the code and evaluate if a better, more modern pattern is appropriate before executing refactors.

## Generalization Blockers (GPU Pipeline)

### Generic Parameter Structs — **[RESOLVED: Phase 4, PR 1]**
- **Resolution:** `ParamsUniform` renamed to `BrightnessParams` in both
  `pipeline.rs` (Rust struct) and `brightness.wgsl` (WGSL struct).

### Generic Shader Naming — **[RESOLVED: Phase 4, PR 1]**
- **Resolution:** `shader.wgsl` renamed to `brightness.wgsl`. Future shaders
  follow this naming convention (`saturation.wgsl`, `contrast.wgsl`, etc.).

### Monolithic Pipeline Initialization — **[RESOLVED: Phase 4, PR 2]**
- **Location:** `bdip_core/src/gpu/pipeline.rs` (`Renderer::new`)
- **Current Pattern:** The `Renderer::new` constructor eagerly compiles the shader module and
  statically generates the `ComputePipeline` directly inside the startup sequence.
- **Refactor Goal:** Prevent a "Monster Initialization" bottleneck. As the application grows to
  support dozens of transformations, globally compiling all shaders and building all pipelines at
  startup will bottleneck application launch times and waste GPU resources for transformations the
  user may never actually use.
- **Suggested Remediation:** Introduce a `PipelineCache` with lazy-loading. Shaders and their
  corresponding pipelines JIT-compile upon first invocation and are stored in a HashMap for
  re-use. Ingest/present pipelines remain eagerly initialized (always needed).
- **Resolution Plan:** `specs/2shader_plan.md`, Step 2.

## UI Responsiveness

### [PHASE 4 REQUIREMENT] Synchronous Disk I/O Blocks UI Initialization
- **Location:** `bdip/src/ui_spike.rs` (`SpikeApp::new`)
- **Current Pattern:** The `io::load_image` function is called directly within the application
  initialization logic on the main thread.
- **Risk:** Decoding massive DSLR imagery (e.g. 100+ MB TIFFs) is CPU-intensive. Running it
  synchronously entirely blocks the UI framework event loop from launching. The user sees no window 
  and the app appears "frozen" or unresponsive for up to a full second while booting/loading an image.
- **Suggested Remediation:** Return the `iced::Application` state immediately with an empty "Loading" 
  UI (e.g., `None` for the `image_handle`). Dispatch the heavy `io::load_image` execution as a 
  non-blocking background `iced::Task`. Once the background worker yields the loaded image, pass it 
  back to the event loop natively via an `Update` message to render the scene.
- **Priority:** **High.** Mandatory for delivering a seamless, native-feeling user experience.

### [PHASE 4 REQUIREMENT] Panic-on-Failure in Application Constructor
- **Location:** `bdip/src/ui_spike.rs` (`SpikeApp::new`)
- **Current Pattern:** The constructor uses `expect()` for all fallible operations (image loading,
  GPU init, texture download). The `iced` application constructor returns `(Self, Task<Message>)`,
  not a `Result`, so there is no way to propagate errors without panicking.
- **Risk:** An invalid file path, missing GPU, or readback failure crashes the process with a stack
  trace before the window appears. Acceptable in a throwaway spike, unacceptable in a shipping app.
- **Suggested Remediation:** When building the Phase 4 `iced` app, the constructor should return
  immediately with an empty/loading state. All fallible work (I/O, GPU init, pipeline execution)
  should be dispatched via `iced::Task` and results communicated back through `Message` variants
  that carry `Result` payloads. Errors should be surfaced via a UI error dialog, not a panic.
- **Priority:** **High.** Must be addressed in the Phase 4 app design, not retroactively in the
  spike.

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

### CPU Upload Pixel Conversion Loop
- **Location:** `bdip_core/src/gpu/texture.rs` (`upload_texture`)
- **Current Pattern:** The upload function uses a manual `for` loop on the CPU to iterate over every
  pixel, converting `u16` sRGB-normalized values to `f16` before calling `queue.write_texture`.
- **Risk:** Similar to the readback bottleneck, this puts significant linear work on the CPU before
  the GPU can start processing. While less complex than the readback padding constraints, it
  still wastes battery and impacts debug-mode performance for large images.
- **Suggested Remediation:** Upload raw `u16` buffers directly using `TextureFormat::Rgba16Unorm`.
  Refactor the "Ingest" compute shader to read this `Unorm` texture; the GPU hardware will
  automatically handle the conversion to floating point, allowing the CPU to simply use a
  raw memory copy during upload.
- **Priority:** **Medium.** Completes the "zero-CPU-loop" architecture and improves startup latency.

## API Design

### Missing High-Level Pipeline Execution Method
- **Location:** `bdip_core/src/gpu/pipeline.rs` (`Renderer`)
- **Current Pattern:** Consumers must manually assemble the full GPU pipeline sequence
  themselves: `upload_texture` → `renderer.ingest` → `renderer.apply_*` (one per
  transformation) → `renderer.present` → `download_texture`. Both `bdip/src/main.rs` and
  `bdip/src/ui_spike.rs` duplicate this plumbing.
- **Risk:** Every new consumer of `bdip_core` must understand GPU pipeline mechanics
  (ingest/present ordering, texture lifetime) to use the library correctly. This is
  implementation detail leakage — the spec describes the core library as something that
  "consumes a base image buffer and a list of declarative `Transformation` instructions
  and outputs a formatted texture," which implies a single call, not manual orchestration.
  Omitting `ingest` or `present`, or calling them in the wrong order, produces silently
  incorrect color output with no error.
- **Suggested Remediation:** Add a method to `Renderer` (or a free function in
  `bdip_core`) with a signature along the lines of:
  ```rust
  pub fn apply(
      &self,
      engine: &GpuEngine,
      img: &Rgba16Image,
      transforms: &[Transformation],
  ) -> Result<Rgba16Image, BdipError>
  ```
  This method owns the complete sequence internally (upload → ingest → per-transform
  dispatch → present → download) and returns a plain CPU image. The low-level primitives
  (`upload_texture`, `ingest`, `present`, `download_texture`) remain public. This is critical
  for interactive consumers like the UI, which need to upload and `ingest` the full-resolution
  image *once* and cache the resulting intermediate `wgpu::Texture` in GPU memory. During a live
  slider drag, the UI can then just call `apply` and `present` directly on that cached texture,
  skipping the expensive PCIe upload and ensuring smooth responsiveness. The new high-level
  `apply` method covers the common one-shot case (like the CLI) without exposing GPU concepts.
- **Priority:** Medium. Not blocking for V1 given there are only two call sites today,
  but should be addressed before `bdip_core` is used by additional consumers or exposed
  as a library API.

## Future Considerations
*(Add new items here as they are discovered during development)*
