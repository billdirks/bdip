# Fix Film Grain Blue Transform

## Problem Summary

The `film_grain_blue` transform is substantially correct but has two minor issues that should be
addressed.

### Moderate Issue

1. **Incorrect sampler filter** (mod.rs:43): Uses `AuxSamplerFilter::Linear` for blue noise texture.
   Blue noise should use nearest/point sampling to preserve its carefully crafted spectral
   properties. Linear filtering interpolates between texels, which can blur the noise and partially
   destroy the blue noise distribution that makes grain appear evenly distributed.

   **Source**: [NVIDIA - Rendering with Spatiotemporal Blue Noise](https://developer.nvidia.com/blog/rendering-in-real-time-with-spatiotemporal-blue-noise-textures-part-1/) -
   "read the texture with nearest neighbor point sampling"

### Minor Issue

2. **Misleading code comment** (film_grain_blue.wgsl:35-37): Comment says "grain is more visible
   in midtones/shadows" but the `sqrt(luma)` weighting actually applies MORE grain to highlights
   and LESS to shadows. The test `test_film_grain_blue_black_pixels_have_minimal_grain` confirms
   the intended behavior (black pixels get no grain), so the code is correct but the comment is
   misleading.

### No Issues Found

- Algorithm is correct for perceptual film grain emulation
- Parameter ranges are appropriate (amount 0.0-0.1 is reasonable for subtle grain)
- Blue noise tiling with variation offset is correctly implemented
- Test coverage is good (9 unit tests)

---

## Implementation Plan

### PR 1: Fix Sampler Filter and Comment

**Goal**: Change blue noise sampler to nearest-neighbor and fix misleading comment.

**Scope**:
- `bdip_core/src/gpu/shaders/film_grain_blue/mod.rs`: Change `AuxSamplerFilter::Linear` to
  `AuxSamplerFilter::Nearest`
- `bdip_core/src/gpu/shaders/film_grain_blue/film_grain_blue.wgsl`: Update comment to accurately
  describe the luminance weighting behavior

**Changes**:

1. In `mod.rs`, line 43, change:
   ```rust
   filter: AuxSamplerFilter::Linear,
   ```
   to:
   ```rust
   filter: AuxSamplerFilter::Nearest,
   ```

2. In `film_grain_blue.wgsl`, lines 35-37, change:
   ```wgsl
   // Rec. 709 luma-weighted grain — grain is more visible in midtones/shadows,
   // matching film emulsion behavior.
   ```
   to:
   ```wgsl
   // Rec. 709 luma-weighted grain — perceptual model where grain visibility
   // scales with luminance (zero in shadows, full in highlights).
   ```

**Tests to Add**: None required - existing tests cover the behavior adequately.

**Existing Tests**: All existing tests should continue to pass. The sampler change may cause minor
numerical differences due to sampling behavior, but the tests use tolerances that should accommodate
this.

---

## Validation Checklist

After PR is merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (all 9 film_grain_blue tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Grain effect still produces visible noise with amount > 0
- [ ] Black pixels still receive minimal/no grain
- [ ] Variation parameter still changes the grain pattern

---

## References

- [NVIDIA - Rendering in Real Time with Spatiotemporal Blue Noise Textures, Part 1](https://developer.nvidia.com/blog/rendering-in-real-time-with-spatiotemporal-blue-noise-textures-part-1/)
- [Free Blue Noise Textures](https://momentsingraphics.de/BlueNoise.html)
- [Dehancer - Film Grain](https://www.dehancer.com/learn/article/grain)
- [Film Grain Emulation 2025 Guide](https://vmoldo.com/film-grain-emulation-2025-guide/)
