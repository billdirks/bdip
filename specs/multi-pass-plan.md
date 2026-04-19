# Multi-Pass Shader Support — Implementation Plan

This plan executes **Option C** from `specs/multi-pass-research.md` (declarative multi-pass
in `ShaderRegistration`) and ships two new transforms on top of it: **Clarity** and
**Cartoon**. Read `specs/multi-pass-research.md` first — this plan assumes the option
tradeoffs, migration-to-E analysis, and bind-group discipline discussed there.

Also required reading before coding:

- `specs/execution_model.md` — Clean Slate Replay invariant; no intermediate CPU readback.
- `specs/adding_a_shader.md` — the current single-pass shader registration pattern, which
  stays the authoritative doc for single-pass shaders. This plan adds a sibling pattern.
- `specs/some_shaders.md` — Clarity formula (`C_hp = C_in - C_blurred`, combined with a
  midtone weight).
- `specs/film_grain_plan.md` — FilmGrainFBM discussion (see "Scope" below for why FBM is
  deliberately *not* part of this plan).

---

## Goals

1. **Enable multi-pass compute for a single user-facing `Transform`** without breaking the
   Clean Slate Replay model, the zero-touch shader registry, or any history/undo invariant.
2. **Ship Clarity** as the first multi-pass shader — separable Gaussian blur + combine.
3. **Ship Cartoon** as the second multi-pass shader — smoothing + edge detection + 3-input
   combine. Cartoon is the concrete 3-input-combine case that validates the
   position-indexed binding discipline in production code.
4. **Keep single-pass shaders unchanged at the user-visible layer.** Migrating each
   existing shader to the new registration shape is mechanical and must not change
   behavior or require per-shader tuning.
5. **Preserve the 24 MP warm-path performance budget** (~20 ms critical path). Each new
   compute pass costs ~0.3–0.5 ms; Clarity adds ~1 ms, Cartoon ~2 ms. Both fit inside the
   readback-dominated frame.

## Non-goals

- **No render graph.** A general DAG (Option E) is out of scope. The position-indexed
  binding discipline this plan enforces is the one structural choice that keeps a future
  migration to E mechanical; nothing else in this plan anticipates E.
- **No auxiliary-texture bindings.** Blue-noise textures, LUTs, and other external
  resources are tracked separately in `film_grain_plan.md` (FilmGrainBlue) and are
  orthogonal to multi-pass.
- **No multi-pass FilmGrainFBM.** Procedural FBM via an in-shader octave loop is
  mathematically equivalent to multi-pass octave summing when both use procedural noise.
  Multi-pass FBM would only raise quality if combined with baked-noise-texture sampling
  (auxiliary-texture territory), which is out of scope. FilmGrainFBM ships separately as a
  single-pass shader per `film_grain_plan.md` option A.
- **No dynamic pass counts.** Every multi-pass shader declares a fixed, compile-time pass
  list. Variable-depth pyramids (Bloom, Laplacian filters) are the first workloads that
  would force Option E, and they are not on this plan's roadmap.
- **No scratch-pool LRU.** The pool drops all scratch textures on image-size change. A
  smarter eviction strategy is a follow-up only if profiling shows it matters.

---

## Architecture decisions

Carried from `specs/multi-pass-research.md` § "Option C" and § "The one gotcha":

1. **A `Transform` may resolve to 1..N compute passes.** One shader_id, one history entry,
   one user-visible transform. Passes are an internal implementation detail.
2. **Passes are declared, not orchestrated.** Each multi-pass shader supplies a static
   `PassDef` array listing its passes in order. The engine walks the array and dispatches;
   shaders do not call into `Renderer`.
3. **Passes name their inputs and output with typed identifiers.** `PassInput::Source` is
   the Transform's input texture (output of the previous Transform in the stack).
   `PassInput::Scratch("name")` is the output of an earlier pass in the same Transform.
   `PassOutput::Scratch("name")` is a recycled intermediate; `PassOutput::Final` is the
   Transform's output texture.
4. **Bindings are position-indexed, derived from declared arity.** A pass declaring N
   inputs binds them to `@group(0) @binding(0)` through `@group(0) @binding(N-1)`. The
   output storage texture binds to `@group(0) @binding(N)`. The uniform buffer stays on
   `@group(1) @binding(0)`. This is the single most important architectural commitment in
   the plan — it is what makes a future Option E migration touch zero existing WGSL.
5. **Scratch textures are owned by `Renderer` and recycled.** Keyed by
   `(shader_id, scratch_name, width, height)`. Pool is dropped on image-resize. Mirrors
   existing patterns (`present_tile_buffer`, `staging_buffer`).
6. **Single-pass shaders remain first-class.** `ShaderProgram::Single` is the existing
   path, preserved unchanged behind a new enum variant. `ShaderProgram::MultiPass` is the
   new path.
7. **One WGSL file per pass.** A multi-pass shader is a directory with one `mod.rs` and N
   `.wgsl` files. `include_str!` picks each up at compile time.
8. **Scratch textures are `Rgba16Float`** — same format as the main pipeline. No precision
   loss between passes; full headroom preserved.

---

## Core abstractions

### New types in `bdip_core/src/gpu/shaders/mod.rs`

```rust
/// Which resource a pass reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassInput {
    /// The Transform's input texture (output of the previous Transform).
    Source,
    /// Output of a prior pass in this same Transform, by name.
    Scratch(&'static str),
}

/// Where a pass writes its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassOutput {
    /// Intermediate — engine allocates/recycles a scratch texture by this name.
    Scratch(&'static str),
    /// Final output of the Transform.
    Final,
}

/// Declarative description of one compute pass inside a multi-pass shader.
#[derive(Debug, Clone, Copy)]
pub struct PassDef {
    /// Debug label (e.g., "blur_h"). Used for GPU object labels and error messages.
    pub label: &'static str,
    /// WGSL source for this pass. Must declare bindings per the position-indexed
    /// contract (see `specs/multi-pass-plan.md` § "Bind-group contract").
    pub wgsl_source: &'static str,
    /// Inputs in declared order. Index `i` binds to `@group(0) @binding(i)` in WGSL.
    /// An empty slice is allowed for passes that synthesize their output from
    /// uniforms only (rare; included for completeness).
    pub inputs: &'static [PassInput],
    /// Where this pass writes.
    pub output: PassOutput,
}

/// Replaces the single `wgsl_source: &'static str` field on `ShaderMeta`.
#[derive(Debug, Clone, Copy)]
pub enum ShaderProgram {
    /// Existing single-pass behavior. WGSL string unchanged from today's shaders.
    Single(&'static str),
    /// Ordered list of passes. Engine dispatches each in sequence.
    MultiPass(&'static [PassDef]),
}
```

### Updated `ShaderMeta`

```rust
pub struct ShaderMeta {
    pub id: &'static str,
    pub display_name: &'static str,
    pub program: ShaderProgram,   // replaces `wgsl_source: &'static str`
    pub param: ParamKind,
}
```

### Bind-group contract (multi-pass passes)

Each pass's bind group 0 is built from the declared `inputs` slice plus one output slot.
Group 1 continues to carry the uniform buffer.

| Group | Binding    | Resource                                                            |
|-------|------------|---------------------------------------------------------------------|
| 0     | 0..N-1     | Input textures in declared order (N = `inputs.len()`)               |
| 0     | N          | Destination storage texture (`rgba16float, write`)                  |
| 1     | 0          | Uniform buffer (same shader-wide params for all passes)             |

**All passes in one shader share the same uniform buffer.** Parameters are shader-level,
not pass-level. A Clarity pass and Cartoon pass each have one params struct; internal
passes read whichever fields they need. Keeping uniforms shader-scoped (not pass-scoped)
avoids a second uniform-buffer allocation per pass for data that never changes between
passes within a single Transform.

### Single-pass backwards compatibility

`ShaderProgram::Single(wgsl)` is equivalent to a `MultiPass` program with one pass that has
`inputs: &[PassInput::Source]` and `output: PassOutput::Final`. The engine can implement
`Single` as a literal alias of that, or keep the existing fast path — either is correct.
Single-pass WGSL files stay exactly as they are today (source at `@binding(0)`, dest at
`@binding(1)`, uniform at `@group(1) @binding(0)`).

---

## Renderer changes

All changes are in `bdip_core/src/gpu/pipeline.rs`. Nothing else in `bdip_core` needs to
change.

### `PipelineCache`

Today: `HashMap<&'static str, CachedPipeline>`.
Change to: `HashMap<&'static str, Vec<CachedPipeline>>` (indexed by pass index; single-pass
shaders have `Vec` of length 1).

Compilation is still lazy per shader_id. When compiling a multi-pass shader, each
`PassDef` gets its own `ComputePipeline`, its own bind-group layout (derived from
`inputs.len()`), and its own shader module (from the pass's `wgsl_source`).

### Scratch pool

New field on `Renderer`:

```rust
// Keyed by (shader_id, scratch_name, width, height). All scratch textures are
// Rgba16Float — same format as the main pipeline, so no precision loss.
scratch_pool: HashMap<(&'static str, &'static str, u32, u32), wgpu::Texture>,
```

Scratch textures are allocated on first miss and retained for subsequent invocations at
the same `(shader_id, scratch_name, dims)` key. When `Renderer::apply` sees a new image
size for a key already in the pool at different dims, the old entry is dropped and
replaced. For V1, **any `apply` call at dims different from the majority of pool entries
triggers a full pool reset** (simpler than per-key tracking, and image resize is rare).

### `Renderer::apply` dispatch

Today `Renderer::apply` is a single compute dispatch. Change it to:

```rust
pub fn apply(&mut self, engine: &GpuEngine, src_texture: &wgpu::Texture, transform: &Transform) -> wgpu::Texture {
    let reg = registry_by_id(transform.shader_id).expect(...);
    match reg.meta.program {
        ShaderProgram::Single(_) => self.apply_single_pass(engine, src_texture, transform, reg),
        ShaderProgram::MultiPass(passes) => self.apply_multi_pass(engine, src_texture, transform, reg, passes),
    }
}
```

`apply_single_pass` is the existing code, unchanged.

`apply_multi_pass`:

1. Look up or lazily create the scratch texture for every `PassOutput::Scratch(name)` in
   the pass list. All are the same size as the Transform's input.
2. Allocate the `Final` destination texture (same size as input), just like single-pass
   does today.
3. Build the uniform buffer once from `transform.values`.
4. For each pass in declaration order:
   - Resolve each `PassInput` to a concrete `wgpu::TextureView` (Source = input texture,
     Scratch = pool entry).
   - Resolve `PassOutput` to a concrete destination view (Scratch = pool entry, Final =
     output texture).
   - Build bind group 0 from `inputs.len()` input views plus the destination view.
   - Build bind group 1 from the shared uniform buffer.
   - Dispatch the pass's pipeline.
5. Submit one command encoder containing all passes (single submission per Transform —
   matches single-pass today).
6. Return the `Final` texture.

All scratch textures stay in the pool after the call returns; the next Clarity invocation
at the same dims reuses them.

---

## Migrating existing single-pass shaders

Every existing shader replaces `wgsl_source: include_str!("foo.wgsl")` with
`program: ShaderProgram::Single(include_str!("foo.wgsl"))`. This is mechanical across:

```
bdip_core/src/gpu/shaders/
├── brightness/mod.rs
├── contrast/mod.rs
├── exposure/mod.rs
├── grayscale/mod.rs
├── highlights/mod.rs
├── invert/mod.rs
├── saturation/mod.rs
├── shadows/mod.rs
├── temperature/mod.rs
├── tint/mod.rs
└── vignette/mod.rs
```

No WGSL file changes. No test changes. No behavioral change. Every existing test passes
unchanged.

---

## Motivating examples

### Clarity

Per `specs/some_shaders.md`:

```
C_hp = C_in - C_blurred
C_out = C_in + (C_hp * u_Clarity * W_mid)
```

where `W_mid` is a midtone luminance weight (peaks at mid-gray, falls off toward both
endpoints), and `C_blurred` is a Gaussian-smoothed copy of `C_in`.

**Pass list (3 passes):**

| Index | Label       | Inputs                            | Output           | Purpose                              |
|-------|-------------|-----------------------------------|------------------|--------------------------------------|
| 0     | `blur_h`    | `[Source]`                        | `Scratch("h")`   | Horizontal separable Gaussian blur   |
| 1     | `blur_v`    | `[Scratch("h")]`                  | `Scratch("v")`   | Vertical separable Gaussian blur     |
| 2     | `combine`   | `[Source, Scratch("v")]`          | `Final`          | High-pass extract + midtone-weighted |

**Params (single shared uniform across all 3 passes):**

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClarityParams {
    pub amount: f32,          // u_Clarity ∈ [-1.0, 1.0]
    pub blur_sigma: f32,      // fixed default e.g., 2.0% of image diagonal
    pub _padding: [f32; 2],
}
```

For V1, `blur_sigma` is not exposed — it is a fixed value set by `from_values(amount)`.
Exposing it as a second slider is a later, separate decision.

**Blur kernel size.** A 1D Gaussian with σ=2% of image width (e.g., ~100 px on a 24 MP
image) truncated at 3σ gives ~600 taps per invocation per pass. This is the professional
range. On 24 MP M4 Pro, ~0.4 ms per pass × 2 blur passes + ~0.05 ms combine = ~0.85 ms.
Verifiable against the warm perf test.

### Cartoon

Standard toon-filter pipeline: edge-preserving smoothing → quantization → edge detection →
combine.

**Pass list (5 passes):**

| Index | Label       | Inputs                                                   | Output             | Purpose                                  |
|-------|-------------|----------------------------------------------------------|--------------------|------------------------------------------|
| 0     | `smooth_h`  | `[Source]`                                               | `Scratch("sh")`    | Horizontal blur (larger σ than Clarity)  |
| 1     | `smooth_v`  | `[Scratch("sh")]`                                        | `Scratch("smooth")`| Vertical blur → smoothed image           |
| 2     | `quantize`  | `[Scratch("smooth")]`                                    | `Scratch("quant")` | Posterize to N levels per channel        |
| 3     | `edges`     | `[Source]`                                               | `Scratch("edges")` | Sobel magnitude → single-channel mask    |
| 4     | `combine`   | `[Source, Scratch("quant"), Scratch("edges")]`           | `Final`            | Mix `quant` with `Source` by `u_Strength`; overlay edges |

The `combine` pass is the 3-input-combine case. It binds three input textures to
`@binding(0)`, `@binding(1)`, `@binding(2)`, writes output to `@binding(3)`, reads
uniforms from `@group(1) @binding(0)`.

**Params:**

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CartoonParams {
    pub strength: f32,         // [0.0, 1.0] — 0 = original, 1 = full cartoon
    pub levels: f32,           // posterization levels per channel ∈ [2.0, 16.0]
    pub edge_threshold: f32,   // Sobel-magnitude cutoff ∈ [0.0, 1.0]
    pub edge_darkness: f32,    // how black the overlaid edges are ∈ [0.0, 1.0]
}
```

Four sliders: Strength, Levels, Edge Threshold, Edge Darkness. Matches the 4 sliders other
multi-param shaders (Shadows, Highlights) already expose.

### FilmGrainFBM — why it's not here

Procedural FBM = sum of N Perlin octaves with halving amplitude, doubling frequency.
Whether you sum them by looping in one shader invocation or by running N shader passes that
each add to an accumulator, the *result is identical* when the noise source is procedural
Perlin. Multi-pass gives zero quality improvement.

Multi-pass FBM raises quality only when combined with baked-noise-texture sampling (at
different mip levels), which requires auxiliary-texture binding — a separate architectural
axis owned by `film_grain_plan.md` (FilmGrainBlue). FilmGrainFBM continues to ship as a
single-pass shader via the in-shader octave loop called out as "Option A" in that spec.

---

## Test strategy

### Infrastructure tests (PR 1)

Tests live in `bdip_core/src/gpu/pipeline.rs` and `bdip_core/src/gpu/shaders/mod.rs`.

- `test_shader_program_single_round_trips` — an existing single-pass shader wrapped in
  `ShaderProgram::Single` still produces bit-identical output to before (pick brightness
  with a known parameter; compare to the existing roundtrip assertion).
- `test_shader_program_multipass_scratch_recycling` — a 2-pass test shader whose second
  pass copies its scratch input to `Final`. Run `apply` twice at the same dims; assert the
  scratch texture handle is reused (read back pool state for the assertion, or an
  observable side effect via labels).
- `test_shader_program_multipass_image_resize_drops_pool` — `apply` once at 4×4, then once
  at 8×8; assert the pool has no lingering 4×4 entries (check pool size).
- `test_shader_program_multipass_final_output_correctness` — a 2-pass identity shader
  (pass 0: Source → Scratch, pass 1: Scratch → Final, both plain copies) returns
  pixel-identical output to the input after `ingest`→`apply`→`present` roundtrip.
- `test_pipeline_cache_multi_pass_compiles_per_pass` — `get_or_create` on a multi-pass
  shader returns a `Vec<CachedPipeline>` whose length equals the pass count, and a second
  call returns the same vec entries by pointer equality.
- `test_position_indexed_bindings_three_inputs` — a test shader with a 3-input pass
  (`@binding(0)`, `@binding(1)`, `@binding(2)` for inputs; `@binding(3)` for output)
  correctly reads all three and writes the expected combination. This is the explicit
  regression guard against reverting to hardcoded binding slots.

Each test follows `AGENTS.md` single-behavior rule.

### Shader-level tests (PRs 2 & 3)

**Clarity** (mirrors existing shader-test style from `vignette/mod.rs`):

| Test name                                      | Setup                                              | Assertion                                                        |
|------------------------------------------------|----------------------------------------------------|------------------------------------------------------------------|
| `test_clarity_registry_entry_exists`           | —                                                  | `registry_by_id("clarity").is_some()`                            |
| `test_clarity_registry_metadata`               | —                                                  | display_name, `Sliders`, `ShaderProgram::MultiPass` with 3 passes|
| `test_clarity_make_uniform_known_value`        | `reg.make_uniform(&[0.5])`                         | bytes equal `bytemuck::bytes_of(&ClarityParams { amount: 0.5, .. })` |
| `test_clarity_zero_amount_is_identity`         | 16×16 solid mid-gray, `amount=0.0`                 | every pixel within ±64 of input                                  |
| `test_clarity_positive_amount_increases_contrast_on_edge` | synthetic step image (left half gray, right half white); `amount=0.5` | pixels just inside the edge on each side diverge further from the mean than with `amount=0.0` |
| `test_clarity_negative_amount_softens_edge`    | same step image; `amount=-0.5`                     | edge transition is softer than at `amount=0.0`                   |
| `test_clarity_alpha_preserved`                 | 4×4 solid mid-gray, `amount=0.5`                   | every output pixel's alpha == 65535                              |
| `test_clarity_deterministic`                   | 16×16 solid mid-gray; run twice at `amount=0.5`    | outputs pixel-identical                                          |
| `test_clarity_scratch_pool_reuses_across_runs` | run Clarity twice at same dims                     | pool has 2 entries ("h", "v") both times; same texture pointers  |

**Cartoon** (mirrors the same structure):

| Test name                                        | Setup                                             | Assertion                                                        |
|--------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------|
| `test_cartoon_registry_entry_exists`             | —                                                 | `registry_by_id("cartoon").is_some()`                            |
| `test_cartoon_registry_metadata`                 | —                                                 | name, 4 sliders, `MultiPass` with 5 passes                       |
| `test_cartoon_make_uniform_known_value`          | `reg.make_uniform(&[0.5, 8.0, 0.2, 0.8])`         | bytes equal `bytemuck::bytes_of(&CartoonParams { .. })`          |
| `test_cartoon_zero_strength_is_identity`         | solid gradient image, `strength=0.0`              | output within ±128 of input per channel                          |
| `test_cartoon_full_strength_reduces_unique_colors` | smooth gradient, `strength=1.0`, `levels=4`     | unique pixel values in output < unique values in input (posterization works) |
| `test_cartoon_edges_darken_high_gradient_pixels` | image with sharp black/white edge, `edge_darkness=1.0` | pixels along the edge are darker in the output than in the input|
| `test_cartoon_no_edges_below_threshold`          | smooth gradient with no edges, `edge_threshold=0.1` | no pixel differs from the pure-posterized version by more than ±64 |
| `test_cartoon_alpha_preserved`                   | 4×4 solid mid-gray                                | every output pixel's alpha == 65535                              |
| `test_cartoon_deterministic`                     | same image, run twice with same params            | outputs pixel-identical                                          |
| `test_cartoon_three_input_combine_pass_binds_correctly` | synthetic inputs distinguishable per channel | combine output shows contributions from all 3 inputs (regression guard on binding positions) |

### Cross-shader integration tests (PR 4)

Add to `bdip_core/src/gpu/shaders/cross_shader_tests.rs`:

- `test_brightness_then_clarity` — Brightness 0.2 → Clarity 0.5 produces a deterministic
  output; mean pixel brightness > mean of Brightness-alone output (Clarity does not cancel
  Brightness).
- `test_clarity_then_vignette` — stacking Clarity with Vignette composes without crashing
  and preserves alpha.
- `test_cartoon_then_saturation` — Cartoon then Saturation composes; unique-color count
  after the stack is close to the Cartoon-alone count (Saturation does not restore colors
  that Cartoon quantized away).

### Performance budget test

Extend `test_perf_gpu_roundtrip_24mp` (or add `test_perf_gpu_roundtrip_24mp_clarity`) to
measure Clarity and Cartoon cold + warm critical paths on the 24 MP synthetic image, with
assertions:

- Clarity warm critical path: < 22 ms (20 ms baseline + ~1 ms Clarity overhead + ~1 ms
  slack).
- Cartoon warm critical path: < 24 ms (20 ms baseline + ~2 ms Cartoon overhead + ~2 ms
  slack).

These assertions are *soft* ceilings — if the measurement drifts, the assertion fires
before regressions reach production. The benchmark is `#[ignore]`-gated, same as today's
perf test.

---

## PR breakdown

Each PR is a discrete, reviewable unit. No PR leaves the codebase in a broken state.

### PR 1 — Multi-pass infrastructure + existing-shader migration

**Scope:** All architectural changes needed to support multi-pass, with zero new shaders.
Every existing shader migrates to `ShaderProgram::Single(...)` in the same PR so
`main` always has a consistent shape.

**Files to add:** none.

**Files to modify:**

- `bdip_core/src/gpu/shaders/mod.rs`:
  - Add `PassInput`, `PassOutput`, `PassDef`, `ShaderProgram`.
  - Change `ShaderMeta.wgsl_source: &'static str` to `ShaderMeta.program: ShaderProgram`.
- `bdip_core/src/gpu/pipeline.rs`:
  - `PipelineCache` map value → `Vec<CachedPipeline>`.
  - `Renderer::scratch_pool` field.
  - `Renderer::apply` splits into `apply_single_pass` + `apply_multi_pass`.
  - New private `PassBindGroupLayout` helper that builds a layout from declared input arity.
- `bdip_core/src/gpu/shaders/{brightness,contrast,exposure,grayscale,highlights,invert,saturation,shadows,temperature,tint,vignette}/mod.rs`:
  wrap `include_str!(...)` in `ShaderProgram::Single(...)` (11 files, 1-line change each).
- `specs/adding_a_shader.md`: add a "Multi-pass shaders" section describing the `PassDef`
  pattern and position-indexed binding contract. Single-pass guidance stays unchanged.

**Tests shipped:** the Infrastructure tests listed above, plus a test-only 2-pass "copy
shader" fixture in `pipeline.rs` so the infrastructure has an integration surface without
needing a real new shader. All ~50+ existing shader tests continue to pass unchanged.

**Review focus:** the `PassDef` / `ShaderProgram` shape (this is the public contract),
position-indexed bind-group construction, scratch-pool lifecycle.

**Rollback characteristics:** if this PR is reverted, nothing on `main` ships multi-pass.
No user-visible change either way.

### PR 2 — Clarity shader

**Scope:** First real multi-pass shader on top of PR 1's infrastructure.

**Files to add:**

- `bdip_core/src/gpu/shaders/clarity/mod.rs`
- `bdip_core/src/gpu/shaders/clarity/blur_h.wgsl`
- `bdip_core/src/gpu/shaders/clarity/blur_v.wgsl`
- `bdip_core/src/gpu/shaders/clarity/combine.wgsl`

**Files to modify:**

- `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod clarity;`.
- `specs/some_shaders.md` — update the Clarity row to note it ships as multi-pass
  (separable Gaussian + combine) and reference this plan.

**Tests shipped:** the Clarity shader-level test matrix above.

**Review focus:** WGSL correctness of the Gaussian kernel (σ, kernel width, normalization),
midtone-weight formula, visual result on a real photo (include before/after screenshots
in the PR description).

### PR 3 — Cartoon shader

**Scope:** Second multi-pass shader; exercises the 3-input combine path.

**Files to add:**

- `bdip_core/src/gpu/shaders/cartoon/mod.rs`
- `bdip_core/src/gpu/shaders/cartoon/smooth_h.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/smooth_v.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/quantize.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/edges.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/combine.wgsl`

**Files to modify:**

- `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod cartoon;`.
- `specs/transformations_reference.md` — add a Cartoon section under a new "Stylization"
  heading referencing the XDoG paper and this plan.

**Tests shipped:** the Cartoon shader-level test matrix above, including the explicit
3-input binding regression guard.

**Review focus:** Sobel kernel, posterize math, combine formula, and **the WGSL binding
indices in `combine.wgsl`** — this is the first 3-input pass on `main` and is the
production validation of the position-indexed discipline.

### PR 4 — Cross-shader integration + performance guardrails

**Scope:** Stitch multi-pass into the broader test story. Small, safe.

**Files to modify:**

- `bdip_core/src/gpu/shaders/cross_shader_tests.rs` — add the three cross-shader chain
  tests listed above.
- `bdip_core/src/gpu/pipeline.rs` — extend `test_perf_gpu_roundtrip_24mp` (or add
  siblings) with Clarity + Cartoon perf assertions.

**Tests shipped:** cross-shader chain tests + perf guardrails.

**Rollback characteristics:** tests-only PR; reverting loses coverage but not behavior.

---

## Risks and open questions

- **Blur kernel portability across GPUs.** Large Gaussian kernels loop many texture reads;
  shader compilers may unroll differently. Validated on M4 Pro in PR 2's perf test; must
  also be spot-checked on a discrete NVIDIA GPU before V1 ship (not blocking for PR
  merges but must be on the V1 checklist).
- **Scratch pool growth if users stack multiple multi-pass shaders.** Each multi-pass
  shader contributes 1–4 scratch textures per image size. 24 MP `Rgba16Float` = ~185 MB
  per texture. A stack of 5 multi-pass shaders = ~1 GB scratch VRAM. Acceptable on
  typical discrete GPUs; tight on integrated GPUs. Mitigation if it bites: share a pool of
  *generic* scratch textures keyed only by dims, not shader_id — defer until profiling
  shows it matters.
- **Parameter coupling for Clarity.** V1 hardcodes blur sigma; a later PR may expose it as
  a slider. Decision is not blocking.
- **Cartoon parameter ranges.** The ranges above are informed estimates; will need tuning
  on real photos during PR 3 review. Not an architecture question.
- **`rustfmt` interaction with `ShaderProgram::Single(include_str!(...))`.** Long
  `include_str!` calls sometimes cause awkward line breaks. Non-blocking; can use
  `#[rustfmt::skip]` if needed on the affected lines.
