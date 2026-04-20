# Technical Debt & Refactoring Tracker

This document tracks known architectural shortcuts, generic naming, and structural issues that were accepted during early iterations (Phase 1 & 2) to maintain velocity. 

> [!CAUTION]
> **To Future AI Contexts / Agents:** The remediations proposed below are *suggestions* based on the state of the codebase at the time they were documented. Do NOT execute them blindly. Always cross-reference these suggestions with the *current* state of the code and evaluate if a better, more modern pattern is appropriate before executing refactors.

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

### Cartoon (sRGB-quantization variant)
- **Location:** `bdip_core/src/gpu/shaders/cartoon/` (`quantize.wgsl`)
- **Current Pattern:** Cartoon's `quantize` pass applies `floor(c * L) / (L - 1)` in
  linear-light space, consistent with the rest of the pipeline (`Rgba16Float`, linear
  throughout). Banding boundaries fall at energy-uniform intervals. The `quantize.wgsl`
  carries an inline comment calling this out.
- **Risk:** Users familiar with Photoshop's Posterize, GIMP's Posterize, or other tools
  that quantize in sRGB-gamma space will see visibly different band placement for the
  same `levels` value. On a linear ramp, linear quantization puts more bands near black
  (where human perception already has finer discrimination) and fewer near white.
  sRGB-space quantization is the opposite — perceptually-even bands that match user
  intuition from those tools. This is a stylistic difference, not a correctness bug.
- **Suggested Remediation:** If PR 3 review or user feedback indicates the linear
  banding feels "wrong" for the toon aesthetic, ship a second `cartoon_srgb` shader
  that mirrors `cartoon`'s pass list but with an sRGB-encode / linear-decode around the
  quantize step:
  ```wgsl
  // In quantize.wgsl (variant)
  let srgb = pow(smoothed.rgb, vec3<f32>(1.0 / 2.2));
  let q    = floor(srgb * L) / (L - 1.0);
  let quantized_rgb = pow(clamp(q, 0.0, 1.0), vec3<f32>(2.2));
  ```
  The variant ships as a parallel shader (not a toggle inside `cartoon`) so users can
  pick the aesthetic directly and both stay testable with their own fixtures. Reuses
  the same smooth/edges/combine passes unchanged.
- **Priority:** Low. Only worth building if users report the linear-quantized aesthetic
  is a dealbreaker. A comment in `quantize.wgsl` explaining the choice is enough
  until then.

## Potential Improvements

### Proxy Resolution During Live Preview
- **Goal:** Processing a downscaled proxy during interaction and applying the full-resolution pipeline only on slider release to reduce live-preview latency.
- **Rationale:** While full-resolution processing hits our targets on Apple Silicon, very large images (50MP+) or discrete GPUs with slower PCIe readback may benefit from a 4-10x latency reduction.
- **Priority:** **Lowest.** (We will see if we can hit sub-20ms goals on primary target hardware without this complexity).
- **Suggested Remediation:** 
  During active slider interaction (the `is_previewing` state), the pipeline should be dispatched against a downscaled proxy texture (e.g., a display-resolution proxy of ~2–5MP) rather than the full-resolution source (e.g., 24MP+). On slider release, the full-resolution result is computed once to ensure the final preview and history entries are high-fidelity. The internal `Renderer::apply` and `Renderer::present` methods are already resolution-independent, so this primarily requires managing the lifecycle of the proxy texture in `BdipApp`.

### Pipeline Latency Investigation (Profiling)
- **Location:** `bdip_core/src/gpu/pipeline.rs` (`test_perf_gpu_roundtrip_24mp`)
- **Goal:** Profile the warm editing path to identify bottlenecks and understand if there are opportunities for further speed improvements on 24MP+ images.
- **Suggested Profiling Tools:**
  - **Xcode Instruments (Metal System Trace):** Measures exact GPU execution timing and command submission overhead on macOS/Metal.
  - **RenderDoc:** Provides a frame-level view of all commands, bindings, and state transitions sent to the GPU.
  - **wgpu Timestamp Queries:** Measures the duration of specific compute passes directly on the GPU, helping isolate execution time from CPU driver overhead.
- **Priority:** Low. (Currently meeting targets, but serves as a placeholder for analysis after the multi-pass architecture is stable).

## User Interface & Metadata

### Missing Parameter Descriptions
- **Location:** `bdip_core/src/gpu/shaders/mod.rs` (`SliderDef`), `bdip/src/ui/sidebar.rs`
- **Current Pattern:** Shader parameters (sliders) only have a name and numeric bounds.
  Descriptions are not defined in the shader metadata.
- **Risk:** Users may not understand the specific effect of a parameter without experimentation.
  Lack of inline documentation/tooltips makes the application less accessible.
- **Suggested Remediation:**
  1.  Extend `SliderDef` in `bdip_core` with a `description: &'static str` field.
  2.  Update all shader `META` definitions to include concise, helpful descriptions.
  3.  Implement hover tooltips in the `bdip` UI (via `iced` or similar) to show these descriptions
      when hovering over parameter names in the sidebar.
- **Priority:** Low. Beneficial for UX and discoverability as the shader library grows.
