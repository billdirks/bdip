# Technical Debt & Refactoring Tracker

This document tracks known architectural shortcuts, generic naming, and structural issues that were accepted during early iterations (Phase 1 & 2) to maintain velocity. 

> [!CAUTION]
> **To Future AI Contexts / Agents:** The remediations proposed below are *suggestions* based on the state of the codebase at the time they were documented. Do NOT execute them blindly. Always cross-reference these suggestions with the *current* state of the code and evaluate if a better, more modern pattern is appropriate before executing refactors.

## UI Responsiveness

### [PHASE 5 REQUIREMENT] Synchronous Disk I/O Blocks UI Initialization
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

### [PHASE 5 REQUIREMENT] Panic-on-Failure in Application Constructor
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
- **Location:** `bdip_core/src/gpu/texture.rs` (`download_presentation_buffer`), call sites in
  `bdip/src/ui/app.rs`
- **Current Pattern:** `download_presentation_buffer()` calls `device.poll(wait_indefinitely)`, which
  blocks the calling thread until the GPU finishes and the PCIe transfer completes. This is currently
  called on the UI thread during both the initial image load (to generate the UI handle) and during
  transform adjustments.
- **Risk:** While fast on Apple Silicon (1–4ms), on discrete GPUs or with very large images (40MP+),
  this blocking call causes the UI to freeze momentarily. This is particularly noticeable after the
  initial file load completes, where the app stays "busy" for another few hundred milliseconds while
  the GPU prepares the preview.
- **Suggested Remediation:** Move the full pipeline invocation (compute shader dispatch +
  `download_presentation_buffer`) into an `iced::Task`. Once the readback completes, send the
  resulting `Rgba16Image` back to the UI state via a `Message`. This keeps the UI event loop fully
  responsive with a "loading" state instead of freezing.
- **Priority:** **Medium.** Important for maintaining a premium feel on high-resolution imagery.

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

## GPU Pipeline Extensibility

### Shader Isolation — Per-Shader Touch Points in `pipeline.rs`
- **Location:** `bdip_core/src/gpu/pipeline.rs`
- **Current Pattern:** Adding a new transform shader requires modifying 4 locations in
  `pipeline.rs`: a params struct, a `TransformKind` variant (+ `From` impl), a
  `PipelineCache::compile()` match arm, and a `Renderer::apply()` match arm. The
  `TransformKind::from()` and `apply()` functions panic at runtime on unhandled variants.
- **Risk:** As the shader count grows, `pipeline.rs` becomes a long file of mechanically
  similar match arms. New shader authors must read through unrelated shader definitions to
  find insertion points. Missing a match arm is a runtime panic, not a compile-time error.
- **Suggested Remediation:** Extract each shader into a self-contained module implementing a
  `TransformShader` trait, with a single registration point for dispatch. See
  `specs/isolating_shaders_plan.md` for the full design. The current process is documented
  in `specs/adding_a_shader.md`.
- **Priority:** Low. The current architecture works well at 2 shaders. Becomes worthwhile at
  4-5 shaders or when shader authors should not need to understand pipeline internals.

## Future Considerations
*(Add new items here as they are discovered during development)*
