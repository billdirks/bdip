# Technical Debt & Refactoring Tracker

This document tracks known architectural shortcuts, generic naming, and structural issues that were accepted during early iterations (Phase 1 & 2) to maintain velocity. 

> [!CAUTION]
> **To Future AI Contexts / Agents:** The remediations proposed below are *suggestions* based on the state of the codebase at the time they were documented. Do NOT execute them blindly. Always cross-reference these suggestions with the *current* state of the code and evaluate if a better, more modern pattern is appropriate before executing refactors.

## UI Responsiveness

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

## Potential Improvements

### Proxy Resolution During Live Preview
- **Goal:** Processing a downscaled proxy during interaction and applying the full-resolution pipeline only on slider release to reduce live-preview latency.
- **Rationale:** While full-resolution processing hits our targets on Apple Silicon, very large images (50MP+) or discrete GPUs with slower PCIe readback may benefit from a 4-10x latency reduction.
- **Priority:** **Lowest.** (We will see if we can hit sub-20ms goals on primary target hardware without this complexity).
- **Suggested Remediation:** 
  During active slider interaction (the `is_previewing` state), the pipeline should be dispatched against a downscaled proxy texture (e.g., a display-resolution proxy of ~2–5MP) rather than the full-resolution source (e.g., 24MP+). On slider release, the full-resolution result is computed once to ensure the final preview and history entries are high-fidelity. The internal `Renderer::apply` and `Renderer::present` methods are already resolution-independent, so this primarily requires managing the lifecycle of the proxy texture in `BdipApp`.
