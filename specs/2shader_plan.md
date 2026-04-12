# Phase 4: Second Shader & Pipeline Generalization

## Goal
Add a Saturation compute shader alongside the existing Brightness shader, resolve the
Monolithic Pipeline Initialization tech debt, and make headless mode fully functional
with arbitrary ordered combinations of both transformations.

## Motivation
The current pipeline is built around a single shader. This masks architectural
assumptions that will break as soon as a second transform is introduced: monolithic
initialization, generic naming, a single `ParamsUniform` struct, and a hardcoded
`apply_brightness` method. Adding a second one-parameter shader (Saturation) is the
smallest step that forces these issues to the surface while the codebase is still
simple. It also validates the multi-shader pipeline before UI work begins in Phase 5.

## Prerequisites
- Phase 3 complete (headless brightness pipeline working end-to-end).

## Deliverables
1. Saturation compute shader (`saturation.wgsl`) operating in linear color space.
2. `PipelineCache` replacing monolithic `Renderer::new` initialization.
3. Renamed `shader.wgsl` to `brightness.wgsl` and `ParamsUniform` to
   `BrightnessParams`.
4. New `SaturationParams` uniform struct.
5. CLI `--apply saturation:<f32>` support.
6. Headless mode supports arbitrary ordered combinations:
   `--apply brightness:0.3 --apply saturation:-0.5 --apply brightness:-0.1`
7. Pipeline file support for the same combinations.
8. Unit and integration tests for the new shader and multi-shader dispatch.

---

## PR Strategy

Steps are grouped into PRs based on compilation boundaries. Steps within a PR
are tightly coupled — landing them separately would leave the code in a
non-compiling or non-functional state.

| PR | Steps | Description | Tests landing with PR |
|----|-------|-------------|---------------------|
| 1 | 1 | Rename generics (`shader.wgsl` → `brightness.wgsl`, `ParamsUniform` → `BrightnessParams`). Pure refactor — all existing tests pass unchanged. | Existing tests (updated references only) |
| 2 | 2, 3, 4 | `PipelineCache` + saturation shader + generalized `apply()`. These three are inseparable: the cache needs a second shader to be meaningful, and `apply()` needs the cache to dispatch. | Saturation correctness tests, cache behavior tests, chaining tests |
| 3 | 5, 6 | CLI multi-transform chaining + saturation parsing. Headless mode fully functional with both shaders. | Integration test (`test_headless_multi_apply`) |
| 4 | 7 | UI spike mechanical update (`apply_brightness` → `apply`). | Existing UI spike continues to compile and run |

**Why not 1 PR per step:** Steps 2–4 form an atomic unit. Step 2 (PipelineCache)
removes `apply_brightness` in favor of a generalized `apply`, but `apply` needs
Step 3 (saturation shader) to have a second pipeline to dispatch, and Step 4
(dispatch logic) is the `apply` method itself. Landing Step 2 alone would leave
`Renderer` with a cache but no method to use it, breaking compilation for all
callers. Similarly, Steps 5–6 are a natural pair: the chaining loop (Step 5) is
trivial without a second transform to chain, and the CLI parser (Step 6) is
useless without the chaining loop.

---

## Implementation Steps

### Step 1: Rename Existing Generics (Tech Debt Cleanup)
**PR 1** — Pure refactor, no behavioral changes.

**Files:** `bdip_core/src/gpu/pipeline.rs`, `bdip_core/src/gpu/shader.wgsl`

- Rename `shader.wgsl` to `brightness.wgsl`.
- Update the `include_str!` path in `pipeline.rs`.
- Rename `ParamsUniform` to `BrightnessParams`.
- Update all references in `pipeline.rs` and tests.

**Resolves:** "Generic Parameter Structs" and "Generic Shader Naming" items in
`specs/tech_debt.md`.

**Verification:** All existing tests pass. `cargo clippy` clean.

### Step 2: Introduce PipelineCache (Lazy-Loading Refactor)
**PR 2** — Lands together with Steps 3 and 4.

**Files:** `bdip_core/src/gpu/pipeline.rs`

Replace the monolithic `Renderer` struct with a `PipelineCache` that JIT-compiles
pipelines on first use.

**Design:**
```
Renderer {
    // Ingest/present pipelines stay eagerly initialized (always needed)
    ingest_pipeline: ComputePipeline,
    ingest_bind_group_layout: BindGroupLayout,
    present_pipeline: ComputePipeline,
    present_texture_bind_group_layout: BindGroupLayout,
    present_params_bind_group_layout: BindGroupLayout,

    // Transform pipelines are lazily initialized
    pipeline_cache: PipelineCache,
}

PipelineCache {
    cache: HashMap<TransformKind, CachedPipeline>,
}

CachedPipeline {
    pipeline: ComputePipeline,
    texture_bind_group_layout: BindGroupLayout,
    params_bind_group_layout: BindGroupLayout,
}
```

- `TransformKind` is a lightweight discriminant enum (e.g., `Brightness`,
  `Saturation`) derived from `Transformation` — it identifies which pipeline to
  use without carrying parameter values.
- `PipelineCache::get_or_create(device, kind)` compiles the shader and pipeline on
  first access, returning a reference to the cached entry.
- `Renderer::new` no longer compiles any transform pipelines. It initializes only
  the ingest and present pipelines (which are always needed) and an empty
  `PipelineCache`.

**Resolves:** "Monolithic Pipeline Initialization" item in `specs/tech_debt.md`.

### Step 3: Write Saturation Shader
**PR 2** — Lands together with Steps 2 and 4.

**File:** `bdip_core/src/gpu/saturation.wgsl`

Saturation adjustment in linear color space. The shader computes luminance using
Rec. 709 coefficients (`0.2126 * R + 0.7152 * G + 0.0722 * B`), then linearly
interpolates between the luminance gray and the original color:

```
result.rgb = mix(vec3(luminance), color.rgb, 1.0 + saturation_offset)
```

Where `saturation_offset` ranges from `-1.0` (full grayscale) to `1.0` (double
saturation). At `0.0`, the image is unchanged.

**Uniform struct:** `SaturationParams { saturation_offset: f32, _padding: [f32; 3] }`

The bind group layout matches the existing brightness contract:
- `@group(0) @binding(0)`: Source texture
  (`texture_storage_2d<rgba16float, read>`)
- `@group(0) @binding(1)`: Destination texture
  (`texture_storage_2d<rgba16float, write>`)
- `@group(1) @binding(0)`: Uniforms

### Step 4: Generalized Transform Dispatch
**PR 2** — Lands together with Steps 2 and 3.

**File:** `bdip_core/src/gpu/pipeline.rs`

Replace the single `apply_brightness` method with a generalized `apply` method:

```rust
pub fn apply(
    &mut self,
    engine: &GpuEngine,
    src_texture: &wgpu::Texture,
    transformation: &Transformation,
) -> wgpu::Texture
```

Internally, this method:
1. Derives `TransformKind` from the `Transformation` variant.
2. Calls `pipeline_cache.get_or_create(device, kind)` to obtain the pipeline.
3. Builds the appropriate uniform buffer (e.g., `BrightnessParams` or
   `SaturationParams`) based on the variant.
4. Creates bind groups, dispatches the compute pass, returns the output texture.

The old `apply_brightness` method is removed. All callers use `apply` instead.

**PR 2 tests:**
- `test_saturation_zero_is_identity` — saturation offset 0.0 produces unchanged
  output.
- `test_saturation_negative_one_produces_grayscale` — full desaturation matches
  luminance values.
- `test_saturation_positive_increases_color` — verify color channels diverge
  further from luminance.
- `test_pipeline_cache_returns_same_pipeline` — calling `get_or_create` twice
  for the same kind returns the cached entry (no recompilation).
- `test_pipeline_cache_different_kinds` — brightness and saturation get separate
  pipelines.
- `test_chained_brightness_then_saturation` — apply both in sequence, verify
  output differs from either alone.
- `test_chained_saturation_then_brightness` — verify order-dependent results
  (saturation then brightness differs from brightness then saturation on
  non-neutral images).
- `test_multiple_same_transform` — applying brightness twice accumulates.

**Verification:** All existing brightness tests pass through the new `apply()`
path. All new saturation and caching tests pass. `cargo clippy` clean.

### Step 5: Multi-Transform Pipeline Chaining
**PR 3** — Lands together with Step 6.

**File:** `bdip/src/main.rs`

Update the headless execution loop to iterate over the full `Vec<Transformation>`
and chain `apply` calls:

```rust
let mut current_texture = renderer.ingest(&engine, &uploaded_texture);
for transform in &transforms {
    current_texture = renderer.apply(&engine, &current_texture, transform);
}
let output_buffer = renderer.present(&engine, &current_texture);
```

This already supports arbitrary ordering and repetition by construction.

### Step 6: CLI Parsing for Saturation
**PR 3** — Lands together with Step 5.

**File:** `bdip/src/main.rs` (in `parse_transform`)

Add a `"saturation"` match arm in the transform parser, mirroring the brightness
pattern. Example usage:
```
bdip --headless --input in.png --output out.png \
    --apply brightness:0.3 --apply saturation:-0.5
```

Pipeline files also support `saturation:<f32>` lines.

**PR 3 tests:**
- `test_headless_multi_apply` — end-to-end CLI test with
  `--apply brightness:0.3 --apply saturation:-0.5`, verifying the output file is
  created and differs from the input.

**Verification:** Full headless pipeline works with arbitrary ordered combinations
of both transforms. Pipeline files with mixed lines work. `cargo clippy` clean.

### Step 7: Update UI Spike
**PR 4** — Standalone mechanical update.

**File:** `bdip/src/ui_spike.rs`

Update the UI spike to use the new `apply` method instead of `apply_brightness`.
No new UI controls are needed — this is a mechanical update to keep the spike
compiling.

**Verification:** UI spike compiles and displays a loaded image with brightness
applied. `cargo clippy` clean.

---

## Tech Debt Items Resolved

| Item | Status |
|------|--------|
| Generic Parameter Structs (`ParamsUniform`) | Resolved in PR 1 (Step 1) |
| Generic Shader Naming (`shader.wgsl`) | Resolved in PR 1 (Step 1) |
| Monolithic Pipeline Initialization | Resolved in PR 2 (Step 2) |

---

## Verification Criteria (Phase Complete)
- `cargo clippy` passes with no warnings.
- All existing tests continue to pass.
- All new tests pass.
- `bdip --headless --input test.jpg --output out.png --apply brightness:0.3
  --apply saturation:-0.5` produces a visibly brighter, less saturated image.
- `bdip --headless --input test.jpg --output out.png --apply saturation:-1.0`
  produces a grayscale image.
- Pipeline file with mixed brightness/saturation lines works correctly.
