# Speeding Up the GPU Critical Path

## Context

After splitting the perf-test timer into `execute` (CPU encode + submit), `gpu_wait`
(time blocked in `device.poll(Wait)` after submission), and `readback`
(`copy_buffer_to_buffer` + map + memcpy), the warm critical path on a 24 MP image
breaks down as follows on Apple M4 Pro:

| Shader              | execute  | gpu_wait    | readback | warm total | target |
|---------------------|----------|-------------|----------|------------|--------|
| Brightness (1 pass) | ~0.4 ms  | ~12 ms      | ~5.5 ms  | ~18 ms     | <25 ms |
| Cartoon (5 passes)  | ~0.5 ms  | **~183 ms** | ~6 ms    | ~189 ms    | <25 ms |
| Clarity (3 passes)  | ~0.4 ms  | **~234 ms** | ~5.5 ms  | ~240 ms    | <25 ms |

`gpu_wait` is the wall-clock time the CPU spends blocked in `device.poll` waiting for
all submitted compute work to complete. It is the true GPU shader execution time. For
Cartoon and Clarity it is >95% of the warm critical path.

The dominant cost is the separable Gaussian blur in `clarity/blur_h.wgsl` and
`cartoon/smooth_h.wgsl`. With `SIGMA_FRACTION = 0.02` and a 5000 px image, sigma is
100 → radius is 300 → **601 `textureLoad` calls per output pixel per blur pass**.
Two blur passes (h+v) over 24 M pixels is ~28.8 billion texture loads. There is no
shared-memory caching, so neighboring threads in a workgroup re-read the same texels
~16× redundantly.

This document enumerates the realistic options for reducing `gpu_wait`, in rough
order of effort vs. payoff.

## Options

### 1. Downsample for the live preview (proxy rendering)

Already documented in `specs/tech_debt.md` (priority: Lowest). During slider drag,
process a 2-5 MP proxy of the source; on slider release, run the full-resolution
pipeline once for fidelity.

- **Expected speedup:** 4-10×. Brings warm critical path to a few ms for any shader.
- **Pros:**
  - No shader changes. Works for every current and future shader uniformly.
  - The renderer is already resolution-independent — most of the work is UI-side
    lifecycle plumbing for the proxy texture in `BdipApp`.
  - The blur radius scales with image size in the current shaders (`SIGMA_FRACTION
    * max(dims)`), so a 5× downsampled proxy renders the *same proportional* blur
    as the full-res original. Visually faithful.
- **Cons:**
  - Adds state to the UI layer: when to recompute the proxy, when to discard, how
    to swap in the full-res result on release without flicker.
  - Doesn't reduce the work of the final full-resolution apply (still needed for
    the canonical preview frame and any export). It postpones cost; it doesn't
    eliminate it.
  - On slider release the full-res run still takes hundreds of ms — felt as a
    delay between "let go of slider" and "image stops looking blurry".

### 2. Shared-memory tile caching in the blur shaders

Each workgroup cooperatively loads a tile + halo into `var<workgroup>` shared
memory once, then each thread reads from shared memory (an order of magnitude
faster than the texture cache). Standard pattern for separable convolutions.

- **Expected speedup:** 3-5× on blur passes (when feasible). Clarity might drop
  from ~234 ms to ~50-70 ms warm.
- **Pros:** Targets the actual bottleneck: redundant memory traffic.
- **Cons:**
  - WGSL shared memory is typically capped at ~16 KB. For a 16×16 tile + radius-300
    halo (332×332 of `vec4<f32>`), the working set is ~1.7 MB — orders of magnitude
    over budget.
  - To make it fit at large radii you must (a) shrink the tile, (b) split the blur
    into multiple smaller-radius passes, or (c) downsample first (see #4).
  - For radii in the 30-50 range it's a clean win; at radius 300 it forces
    additional structural changes to be useful.

### 3. Linear-sampling trick

Switch from `texture_2d<f32>` + `textureLoad` to a `sampler` binding +
`textureSampleLevel`. With careful tap-position math, each linearly-filtered sample
returns a weighted average of two adjacent texels for free, halving the tap count
of a Gaussian.

- **Expected speedup:** ~2×. Stacks multiplicatively with #2 and #4.
- **Pros:** Small, mechanical change. Independent of other optimizations.
- **Cons:**
  - Requires a sampler bind-group plumbing change (currently the shaders only bind
    textures and storage buffers).
  - Slight numerical precision difference at the half-texel offsets — visually
    indistinguishable for a Gaussian blur but worth noting in tests that compare
    against reference values.

### 4. Downsample → blur → upsample (the photo-editor classic)

For large-radius blurs, downsample by 4× or 8× first, do a small-radius blur on the
smaller image (visually equivalent because spatial frequencies above the downsample
threshold contribute negligibly to a Gaussian's output), then bilinear-upsample
back to full resolution.

For Clarity at radius 300: downsample 4× → blur with radius 75 on a 6 MP image →
upsample. **Order-of-magnitude speedup expected.** This is the technique used by
Lightroom, Photoshop's Gaussian Blur, and similar tools for large radii.

- **Expected speedup:** ~10×+. Likely brings warm Clarity/Cartoon under 25 ms even
  at full resolution.
- **Pros:**
  - Dramatic. Directly attacks the "kernel scales with image size" problem rather
    than working around it.
  - Helps full-resolution export, not just live preview — proxy rendering doesn't.
  - Combines well with #2 (shared memory becomes feasible at the smaller radius)
    and #3 (linear sampling on top).
- **Cons:**
  - Adds 2 passes per blur (down, up) — a wash given the work each one saves.
  - Some loss of very-high-frequency detail in the blurred result, which is fine
    because the result is a blur. Slight visual difference vs. true large-radius
    Gaussian, usually imperceptible. Needs spot-checking against the existing
    test fixtures.
  - Requires new downsample/upsample compute passes (small) and pipeline changes
    in `Renderer::apply_passes` to allow passes at non-source resolution. Today
    every pass operates at source dimensions; the scratch-pool allocator and pass
    encoder both assume that.

### 5. Profile first — wgpu timestamp queries

Right now we know `gpu_wait` is ~234 ms for Clarity but not the per-pass split.
`blur_h` vs `blur_v` may have very different costs due to memory access patterns
(row-major vs column-strided texture reads). `combine` may be unexpectedly cheap or
expensive. Without this data, optimizations 2-4 are speculative.

`wgpu` supports timestamp queries on `ComputePassDescriptor` via the
`TIMESTAMP_QUERY` feature. Add per-pass instrumentation behind a `#[cfg(test)]` or
debug-only flag, surface results in the perf reports.

- **Expected speedup:** zero on its own.
- **Pros:** Tells us where to spend effort. One afternoon of work. Makes #2/#3/#4
  targeted instead of guesswork. Becomes a permanent debugging asset for any
  future shader regression.
- **Cons:** Build cost only.

## Recommendation

Order by cheapest and highest signal first:

1. **Add timestamp queries (#5).** Confirms the bottleneck is the blur passes (most
   likely) and quantifies the split between `blur_h` / `blur_v` / `combine` for
   Clarity, and `smooth_h` / `smooth_v` / `quantize` / `edges` / `combine` for
   Cartoon. Cheap. Permanent value.
2. **Implement downsample-blur-upsample (#4)** for Clarity and Cartoon's smooth
   passes. This is almost certainly the right architectural fix for kernels whose
   radius scales with image size. Most likely brings the full-resolution warm path
   under 25 ms by itself, which would unblock the perf assertions and obviate the
   need for #1 and #2 entirely.
3. **Add linear sampling (#3)** as a 2× cherry on top if Clarity/Cartoon are close
   to budget but not quite under it.
4. **Proxy preview (#1)** as a defensive fallback: ship for users on weaker GPUs or
   significantly larger images (50 MP+). Keep on the roadmap, build after the
   shader-side wins land. Probably not needed on the primary target hardware once
   #4 ships.
5. **Skip shared memory (#2)** until after #4. The tile-budget math doesn't work at
   the current radii; #4 reduces the radii to a range where shared memory becomes
   tractable. If after #4 we still want more speed, revisit.

Shader-side fixes are the bigger lever because they help full-resolution export
too (where the user actually wants the full-res blur, not a proxy). Proxy rendering
is valuable as a defensive fallback but doesn't reduce the actual work — it just
postpones it.

## Recommnedations Implemented

1. [Done] Add timestamp queries
2. [Done] Downsample-blur-upsample

## Notes on measurement methodology

These numbers come from `cargo perf-test` after the timer split landed
(see `bdip_core/src/gpu/image_pipeline.rs::PhaseTiming`). Cold-run `gpu_wait`
includes one-time shader compilation; warm-run `gpu_wait` is the steady-state
interactive editing cost and is the number to compare against the 25 ms target.

The synchronous-readback issue tracked separately in `specs/tech_debt.md`
("Synchronous GPU Readback Blocks the Caller's Thread") is unrelated to GPU
compute time. Even if readback became fully asynchronous, Cartoon and Clarity
would still feel laggy — the GPU is the bottleneck, not the I/O API.
