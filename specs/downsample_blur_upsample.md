# Downsample-Blur-Upsample Optimization Plan

## Goal

Bring Clarity and Cartoon warm critical paths under 25 ms at 24 MP by replacing the
full-resolution Gaussian blur passes with a downsample → blur-at-reduced-resolution →
upsample sequence. This is option 4 from `specs/speed_up_gpu_time.md`.

## Context

The dominant cost in both shaders is the separable Gaussian blur. At 5000 px, Clarity
computes `sigma = 0.02 * 5000 = 100`, `radius = 300`, yielding 601 texture loads per
output pixel per blur pass. Two passes (h+v) over 24 M pixels is ~28.8 billion loads.

A Gaussian at sigma=100 removes all spatial frequencies above ~0.0016 cycles/pixel.
Nyquist for a 4× downsampled image is ~0.0004 cycles/pixel — well below the blur's
cutoff. The frequencies between 0.0004 and 0.0016 are already suppressed by >99.9% by
the Gaussian. Downsampling before blurring therefore produces a visually equivalent
result: the information the blur discards was never present in the downsampled image.

With a 4× downsample (the standard factor used by Lightroom and Photoshop for
large-radius Gaussians):

- Pixel count drops 16× (24 MP → 1.5 MP)
- Radius drops 4× (300 → 75 taps → 151 loads per pixel per pass)
- Combined reduction: ~96× less work in the blur passes

This should bring blur-pass GPU time from ~100+ ms to ~1-2 ms, well within the 25 ms
warm budget even after adding the cheap downsample and upsample passes.

## Architecture Changes

### 1. Add `PassScale` to `PassDef`

**File:** `bdip_core/src/gpu/shaders/mod.rs`

Add a new enum and a field to `PassDef`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassScale {
    Full,
    Down(u32), // 1/N of source dimensions
}

pub struct PassDef {
    pub label: &'static str,
    pub wgsl_source: &'static str,
    pub inputs: &'static [PassInput],
    pub output: PassOutput,
    pub output_scale: PassScale, // NEW
}
```

`output_scale` determines the output texture's dimensions:
- `Full` → same as source `(width, height)`
- `Down(4)` → `(width / 4, height / 4)` (integer division)

The dispatch workgroup count for each pass is derived from its output dimensions:
`(out_w.div_ceil(16), out_h.div_ceil(16), 1)`.

All 13 existing shader modules get `output_scale: PassScale::Full` on every pass. This
is a mechanical change that touches each `PASSES` definition.

### 2. Teach the scratch pool about mixed resolutions

**File:** `bdip_core/src/gpu/image_pipeline.rs` — `ScratchPool`

The pool currently stores textures at a single `(width, height)`. With mixed-resolution
passes, it needs to match textures by size.

Replace:

```rust
struct ScratchPool {
    width: u32,
    height: u32,
    textures: Vec<wgpu::Texture>,
}
```

With:

```rust
struct ScratchPool {
    source_width: u32,
    source_height: u32,
    textures: Vec<(wgpu::Texture, u32, u32)>, // (texture, width, height)
}
```

Behavior:
- `sync_scratch_pool_dims`: clears the pool when source image dimensions change (same
  policy as today).
- `allocate_scratch_textures`: for each scratch output, compute the needed `(w, h)` from
  the pass's `output_scale` and the source dims. Pop a matching-sized texture from the
  pool if one exists; allocate a new one otherwise.
- `return_scratch_textures`: push `(texture, w, h)` back to the pool.

### 3. Update `encode_passes_into` dispatch

**File:** `bdip_core/src/gpu/image_pipeline.rs` — `encode_passes_into`

Currently (line 1078):
```rust
cpass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
```

This assumes all passes share source dimensions. Change to compute per-pass output
dimensions from `pass.output_scale`:

```rust
let (out_w, out_h) = match pass.output_scale {
    PassScale::Full => (src_width, src_height),
    PassScale::Down(n) => (src_width / n, src_height / n),
};
cpass.dispatch_workgroups(out_w.div_ceil(16), out_h.div_ceil(16), 1);
```

`src_width` and `src_height` are the original source dimensions already available in
scope.

### 4. Update `allocate_scratch_textures`

Pass the source dimensions and each pass's `output_scale` through to the allocation
call so that scratch textures for `Down(N)` passes are created at reduced resolution
instead of full resolution.

### 5. Update `validate_pass_list`

No new structural rules are needed. The existing DAG validation (forward-only scratch
references, final-on-last-pass, no duplicate scratch names) is sufficient. Scale
correctness is ensured by the renderer — the scratch texture's actual dimensions are
set by `output_scale`, and shaders adapt via `textureDimensions()`.

## New WGSL Shaders

### Downsample (box filter)

One WGSL per parent shader (Clarity and Cartoon), because each must declare the parent's
full params struct for bind-group layout compatibility. The shader body is identical; only
the struct block differs. (~25 lines each.)

The shader derives the scale factor from the ratio of input to output texture dimensions,
so no hardcoded constant is needed in the WGSL — the Rust-side `PassScale::Down(N)`
controls the allocation and the shader adapts:

```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(output_texture);
    if gid.x >= out_dims.x || gid.y >= out_dims.y { return; }

    let in_dims = textureDimensions(input_texture);
    let scale   = vec2<f32>(in_dims) / vec2<f32>(out_dims);
    let base    = vec2<i32>(vec2<f32>(gid.xy) * scale);
    let block   = vec2<i32>(ceil(scale));

    var accum = vec4<f32>(0.0);
    var count = 0.0;
    for (var dy: i32 = 0; dy < block.y; dy = dy + 1) {
        for (var dx: i32 = 0; dx < block.x; dx = dx + 1) {
            let c = clamp(
                base + vec2<i32>(dx, dy),
                vec2<i32>(0),
                vec2<i32>(in_dims) - 1,
            );
            accum = accum + textureLoad(input_texture, c, 0);
            count = count + 1.0;
        }
    }

    textureStore(output_texture, vec2<i32>(gid.xy), accum / count);
}
```

**Files to create:**
- `bdip_core/src/gpu/shaders/clarity/downsample.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/downsample.wgsl`

### Upsample (bilinear interpolation)

Same duplication pattern. The shader derives the scale from texture dimensions and
performs manual bilinear interpolation from the 4 nearest input texels:

```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(output_texture);
    if gid.x >= out_dims.x || gid.y >= out_dims.y { return; }

    let in_dims = textureDimensions(input_texture);
    let scale   = vec2<f32>(in_dims) / vec2<f32>(out_dims);

    // Map output pixel center to fractional input coordinate.
    let src  = (vec2<f32>(gid.xy) + 0.5) * scale - 0.5;
    let p0   = vec2<i32>(floor(src));
    let frac = src - vec2<f32>(p0);

    let max_coord = vec2<i32>(in_dims) - 1;
    let c00 = textureLoad(input_texture, clamp(p0,                    vec2(0), max_coord), 0);
    let c10 = textureLoad(input_texture, clamp(p0 + vec2<i32>(1, 0), vec2(0), max_coord), 0);
    let c01 = textureLoad(input_texture, clamp(p0 + vec2<i32>(0, 1), vec2(0), max_coord), 0);
    let c11 = textureLoad(input_texture, clamp(p0 + vec2<i32>(1, 1), vec2(0), max_coord), 0);

    let top    = mix(c00, c10, frac.x);
    let bottom = mix(c01, c11, frac.x);
    textureStore(output_texture, vec2<i32>(gid.xy), mix(top, bottom, frac.y));
}
```

**Files to create:**
- `bdip_core/src/gpu/shaders/clarity/upsample.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/upsample.wgsl`

## Pipeline Changes

### Clarity (3 passes → 5 passes)

**File:** `bdip_core/src/gpu/shaders/clarity/mod.rs`

Current pass list:
```
blur_h:   Source          → Scratch("h")   [Full]
blur_v:   Scratch("h")   → Scratch("v")   [Full]
combine:  Source + Scratch("v") → Final    [Full]
```

New pass list:
```
down:     Source          → Scratch("down") [Down(4)]
blur_h:   Scratch("down")→ Scratch("h")    [Down(4)]
blur_v:   Scratch("h")   → Scratch("v")    [Down(4)]
up:       Scratch("v")   → Scratch("up")   [Full]
combine:  Source + Scratch("up") → Final    [Full]
```

The blur shaders need no changes — `textureDimensions()` adapts to the smaller input.
The combine pass reads Source (full-res) and the upsampled blur (full-res), so its
binding contract is unchanged.

### Cartoon (5 passes → 7 passes)

**File:** `bdip_core/src/gpu/shaders/cartoon/mod.rs`

Current pass list:
```
smooth_h: Source           → Scratch("sh")     [Full]
smooth_v: Scratch("sh")   → Scratch("smooth")  [Full]
quantize: Scratch("smooth")→ Scratch("quant")  [Full]
edges:    Source           → Scratch("edges")   [Full]
combine:  Source + Scratch("quant") + Scratch("edges") → Final [Full]
```

New pass list:
```
down:     Source            → Scratch("down")    [Down(4)]
smooth_h: Scratch("down")  → Scratch("sh")      [Down(4)]
smooth_v: Scratch("sh")    → Scratch("smooth")   [Down(4)]
quantize: Scratch("smooth")→ Scratch("quant")    [Down(4)]
up:       Scratch("quant") → Scratch("quant_up") [Full]
edges:    Source            → Scratch("edges")    [Full]
combine:  Source + Scratch("quant_up") + Scratch("edges") → Final [Full]
```

Quantize is included in the downsampled chain because it is a per-pixel operation that
does not depend on absolute resolution — the posterization levels produce the same bands
at any size. Bilinear upsample after quantize acts as natural anti-aliasing on the band
boundaries.

## Scratch Memory Budget (24 MP)

| Config     | Scratch textures              | Total scratch memory |
|------------|-------------------------------|----------------------|
| Current    | 2 × full-res (Clarity)        | 384 MB               |
|            | 4 × full-res (Cartoon)        | 768 MB               |
| Proposed   | 3 × 1/16 + 2 × full (Clarity)| 420 MB               |
|            | 4 × 1/16 + 2 × full (Cartoon)| 396 MB               |

Scratch memory decreases for Cartoon and increases slightly for Clarity (two additional
full-res textures for the down and up names, but the three blur scratches shrink to
1/16th). Net: roughly comparable. The quarter-res textures are ~12 MB each vs ~192 MB
at full-res.

## Testing Strategy

### Existing tests

All existing Clarity and Cartoon unit tests should continue to pass. Key considerations:

- **Identity tests** (`test_clarity_zero_amount_is_identity`,
  `test_cartoon_zero_strength_and_zero_edge_darkness_is_identity`): At amount=0 the
  combine formula ignores the blurred result entirely, so the downsample/upsample path
  is not visible in the output. These should pass unchanged.

- **Behavioral tests** (edge enhancement, softening, posterization): These assert
  directional properties (darker/brighter/fewer colors) rather than exact values. The
  downsample-blur approximation preserves these properties. Should pass unchanged.

- **Tolerance tests**: The ±64 u16 tolerance established for GPU roundtrip tests absorbs
  f16 rounding. The downsample/upsample approximation may add a few more LSBs of error
  on non-identity amounts. Monitor whether the tolerance needs widening. If so, document
  the cause.

### New tests (assigned to PRs)

1. **Scratch pool multi-resolution reuse** → PR 1
2. **Downsample-upsample roundtrip accuracy** → PR 2
3. **Blur equivalence at small image sizes** → PR 2 (optional)

### Perf tests

The existing assertions in `performance.rs` (`perf_gpu_roundtrip_24mp_clarity` and
`perf_gpu_roundtrip_24mp_cartoon`) assert `warm.critical_path_ms() < 25.0`. These
should start passing once the optimization lands. Run and record the new numbers.

## PRs

Three PRs, each producing a compiling, all-tests-passing state. Each builds on the
previous. Designed so that a Sonnet agent with no prior context can implement each one
from the prompt "implement PR N from `@specs/downsample_blur_upsample.md`".

---

### PR 1: Add `PassScale` infrastructure (no behavior change)

**Goal:** Extend the pass system and renderer to support mixed-resolution passes. Every
existing pass uses `Full`, so runtime behavior is unchanged — this PR is pure plumbing.

**Files to modify:**

1. `bdip_core/src/gpu/shaders/mod.rs`
   - Add the `PassScale` enum (see "Architecture Changes §1" above for the definition).
   - Add `output_scale: PassScale` field to `PassDef`.
   - Import/export `PassScale` alongside the existing `PassInput`/`PassOutput` types.

2. All 13 shader modules' `PASSES` definitions — add `output_scale: PassScale::Full`
   to every `PassDef` literal. The modules are:
   `brightness`, `cartoon`, `clarity`, `contrast`, `exposure`, `grayscale`,
   `highlights`, `invert`, `saturation`, `shadows`, `temperature`, `tint`, `vignette`.
   Each lives at `bdip_core/src/gpu/shaders/<name>/mod.rs`.

3. `bdip_core/src/gpu/image_pipeline.rs` — three changes:
   - **`ScratchPool`**: change from `(width, height, Vec<Texture>)` to
     `(source_width, source_height, Vec<(Texture, u32, u32)>)`. Update
     `sync_scratch_pool_dims` to clear on source-dimension change (same policy).
   - **`allocate_scratch_textures`**: for each pass's `PassOutput::Scratch`, compute
     the needed `(w, h)` from `pass.output_scale` and the source dimensions. When
     popping from the pool, match by `(w, h)`. When allocating fresh, use the computed
     dimensions.
   - **`encode_passes_into`**: replace the single `(width, height)` dispatch with
     per-pass output dimensions derived from `pass.output_scale` and source dims.
   - **`return_scratch_textures`**: push `(texture, w, h)` with the texture's actual
     dimensions.
   - **`scratch_pool_info` / `scratch_pool_handle`** (test helpers): update return
     types to include per-texture dimensions. Update any tests that call these methods.

4. `bdip_core/src/gpu/image_pipeline.rs` — **new test**:
   `test_multi_pass_scratch_pool_reuses_mixed_resolution`. Run Clarity (which is still
   all-`Full` at this point) twice and verify pool reuse still works with the new tagged
   pool. This exercises the new pool mechanics against the existing pass list.

**Verification:** `cargo test`, `cargo clippy`, `cargo fmt --all` — all must pass.
No runtime behavior change; no new WGSL files.

---

### PR 2: Clarity downsample-blur-upsample

**Goal:** Add downsample and upsample WGSL shaders for Clarity, wire them into the pass
list, and verify correctness + performance. This is the first shader to use the
`PassScale::Down` infrastructure from PR 1.

**Depends on:** PR 1 merged.

**Files to create:**

1. `bdip_core/src/gpu/shaders/clarity/downsample.wgsl` — Box-filter downsample. Must
   declare the full `ClarityParams` struct (same as the other Clarity WGSL files) for
   bind-group layout compatibility. The shader body derives the scale factor from
   `textureDimensions(input_texture) / textureDimensions(output_texture)` — see
   "New WGSL Shaders § Downsample" above for the reference implementation. Bindings:
   `@group(0) @binding(0)` input texture, `@group(0) @binding(1)` output storage
   texture, `@group(1) @binding(0)` params uniform.

2. `bdip_core/src/gpu/shaders/clarity/upsample.wgsl` — Bilinear upsample. Same params
   struct and binding layout as downsample. See "New WGSL Shaders § Upsample" above
   for the reference implementation.

**Files to modify:**

3. `bdip_core/src/gpu/shaders/clarity/mod.rs` — Change `PASSES` from 3 passes to 5:
   ```
   down:    Source           → Scratch("down")  [Down(4)]
   blur_h:  Scratch("down")  → Scratch("h")     [Down(4)]
   blur_v:  Scratch("h")     → Scratch("v")     [Down(4)]
   up:      Scratch("v")     → Scratch("up")    [Full]
   combine: Source + Scratch("up") → Final       [Full]
   ```
   The existing blur_h.wgsl, blur_v.wgsl, and combine.wgsl are not modified — they
   adapt to the smaller input via `textureDimensions()`. Update the metadata test
   (`test_clarity_registry_metadata`) to expect 5 passes instead of 3.

**New tests to add:**
- **Downsample-upsample roundtrip accuracy**: In `clarity/mod.rs` tests, create a
  two-pass shader-level test (or use `roundtrip` with the new 5-pass Clarity at
  `amount=0`) that confirms a smooth synthetic image survives the
  downsample → blur → upsample path with minimal error (±a few u16 for f16
  precision). This validates the new WGSL shaders produce correct output.
- **Blur equivalence at small image sizes** (optional): Run Clarity on a 64×64 image
  (downsample produces 16×16) and verify the behavioral tests still hold. This stress
  tests the degenerate-size edge case.

**Verification:**
- `cargo test -p bdip_core` — all Clarity unit tests must pass. The identity test
  (`test_clarity_zero_amount_is_identity`) should pass unchanged because `amount=0`
  makes the combine formula ignore the blur. Behavioral tests (edge enhancement,
  softening) assert directional properties, not exact values, so should also pass. If
  any tolerance tests fail, widen the u16 tolerance and document the cause.
- `cargo perf-test -- clarity` — run the Clarity perf benchmark and record the new
  warm `gpu_wait`. Expected: ~5-15 ms (down from ~234 ms).
- `cargo clippy`, `cargo fmt --all`.

---

### PR 3: Cartoon downsample-blur-upsample

**Goal:** Same as PR 2 but for Cartoon. Cartoon includes quantize in the downsampled
chain (it is a per-pixel operation independent of resolution), so upsample happens after
quantize rather than after blur_v.

**Depends on:** PR 1 merged. Independent of PR 2 (could be implemented in parallel on a
separate branch).

**Files to create:**

1. `bdip_core/src/gpu/shaders/cartoon/downsample.wgsl` — Same box-filter body as the
   Clarity version, but declares the full `CartoonParams` struct instead of
   `ClarityParams`. Copy the struct definition from any existing Cartoon WGSL file
   (e.g., `smooth_h.wgsl`). Binding layout: `@group(0) @binding(0)` input,
   `@group(0) @binding(1)` output, `@group(1) @binding(0)` params.

2. `bdip_core/src/gpu/shaders/cartoon/upsample.wgsl` — Same bilinear body as the
   Clarity version, with the `CartoonParams` struct. Same binding layout.

**Files to modify:**

3. `bdip_core/src/gpu/shaders/cartoon/mod.rs` — Change `PASSES` from 5 passes to 7:
   ```
   down:     Source            → Scratch("down")     [Down(4)]
   smooth_h: Scratch("down")   → Scratch("sh")       [Down(4)]
   smooth_v: Scratch("sh")     → Scratch("smooth")   [Down(4)]
   quantize: Scratch("smooth") → Scratch("quant")    [Down(4)]
   up:       Scratch("quant")  → Scratch("quant_up") [Full]
   edges:    Source            → Scratch("edges")     [Full]
   combine:  Source + Scratch("quant_up") + Scratch("edges") → Final [Full]
   ```
   The existing smooth_h.wgsl, smooth_v.wgsl, quantize.wgsl, edges.wgsl, and
   combine.wgsl are not modified. Update the metadata test
   (`test_cartoon_registry_metadata`) to expect 7 passes instead of 5.

**Verification:**
- `cargo test -p bdip_core` — all Cartoon unit tests must pass. Identity test
  (`test_cartoon_zero_strength_and_zero_edge_darkness_is_identity`) should pass
  unchanged. Behavioral tests (posterization count, edge darkening, softness ramp)
  assert directional properties. If tolerance tests fail, widen and document.
- `cargo perf-test -- cartoon` — record the new warm `gpu_wait`. Expected: ~5-15 ms
  (down from ~183 ms).
- `cargo clippy`, `cargo fmt --all`.

## Expected Performance Impact

| Shader  | Current warm `gpu_wait` | Expected warm `gpu_wait` | Reason             |
|---------|-------------------------|--------------------------|--------------------|
| Clarity | ~234 ms                 | ~5-15 ms                 | 96× less blur work |
| Cartoon | ~183 ms                 | ~5-15 ms                 | 96× less blur work |

The downsample and upsample passes are trivial per-pixel operations (~0.5 ms each at
24 MP). The blur passes at quarter resolution process 1/16th the pixels with 1/4 the
radius each. Quantize and edges are already fast and unaffected by the optimization.

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Bilinear upsample softens the blur boundary | Low — the result is a blur, softening is expected | Spot-check visually; switch to bicubic if needed |
| Posterize bands soften after bilinear upsample | Low — acts as free anti-aliasing | If sharp bands desired, use nearest-neighbor for quantize upsample only |
| Tolerance increase on existing tests | Medium | Widen tolerance with documented justification; the visual difference is imperceptible |
| Very small images (< 64 px) hit degenerate downsample | Low — Down(4) on a 16×16 image gives 4×4, still usable | Skip downsample for images where `max(w,h) / N < 32` and fall back to full-res blur |
| `PassDef` change touches every shader module | Certain but mechanical | Step 1 is a pure additive change with `Full` everywhere — easy to review |

## Notes

- The 4× downsample factor is a standard choice (Lightroom, Photoshop). It could be
  tuned per-shader — Cartoon could use Down(3) for slightly sharper smoothing — but 4×
  is a safe starting point.
- The blur shaders need zero changes. They derive sigma from
  `textureDimensions(input_texture)`, so sigma scales automatically with the
  downsample factor. This is the key property that makes the optimization clean.
- Future optimization: linear-sampling trick (option 3 from `speed_up_gpu_time.md`)
  stacks multiplicatively with this work — another 2× on the blur passes if needed.
