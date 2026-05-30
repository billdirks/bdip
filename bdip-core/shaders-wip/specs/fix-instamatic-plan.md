# Fix Instamatic Transform

## Problem Summary

The Instamatic transform has solid algorithm design that correctly emulates cheap instant camera
characteristics (lifted shadows, warm tones, yellow-green midtone cast, desaturation, vignette).
However, it has one moderate issue that affects user experience.

### Moderate Issue

**Double strength application creates quadratic response curve**

The shader multiplies `params.strength` into each sub-effect AND uses it in the final mix:

| Location | Code |
|----------|------|
| Line 62 | `rgb + params.strength * shadow_weight * lift_target` |
| Line 72 | `1.0 - params.strength * 0.06` |
| Line 85 | `faded + params.strength * midtone_weight * midtone_cast` |
| Lines 92-94 | `1.0 + params.strength * 0.04` (etc.) |
| Line 102 | `mix(balanced, vec3<f32>(lum), params.strength * 0.08)` |
| Line 114 | `params.strength * 0.35 * vignette_mask` |
| Line 126 | `mix(rgb, vignetted, params.strength)` (final blend) |

**Effect:** At `strength=0.5`, each effect is applied at 50% intensity, then blended 50% with the
original — resulting in approximately 25% of the intended effect intensity. The response curve is
quadratic rather than linear.

**Evidence from codebase:** Other similar shaders (kodachrome, technicolor, cyberpunk, cross_process)
compute the full effect without strength multiplication, then apply strength only in the final mix:

```wgsl
// kodachrome.wgsl — correct pattern
let graded  = kodachrome_matrix(rgb);  // Full effect computed
let out_rgb = mix(rgb, graded, params.strength);  // Single strength application
```

**User impact:** Slider feels "sluggish" in the lower range. Users must set strength > 0.7 to see
a noticeable effect because the quadratic response compresses most of the visible change into the
upper portion of the slider range.

### What's Working Correctly

The algorithm design is sound and matches research on cheap instant camera characteristics:

1. **Shadow lift toward milky grey** — Matches the raised black floor of cheap instant film
2. **Highlight compression** — Creates the faded look from film not reaching pure white
3. **Yellow-green midtone cast** — Emulates uneven dye response in cheap film stock
4. **Warm channel balance** — Matches the warm, golden tones of Kodacolor-X era film
5. **Slight desaturation** — Produces the muted colors characteristic of consumer film
6. **Radial vignette** — Emulates simple plastic lens darkening

Sources consulted:
- [Kodacolor (still photography) - Wikipedia](https://en.wikipedia.org/wiki/Kodacolor_(still_photography))
- [Kodak Instamatic - Wikipedia](https://en.wikipedia.org/wiki/Instamatic)
- [Lift Gamma Gain - Filmic Worlds](http://filmicworlds.com/blog/minimal-color-grading-tools/)
- [Vintage Filter Guide - The Editing Studio](https://theeditingstudio.co/blog/vintage-clean-film-lightroom-preset-guide)

---

## Implementation Plan

### PR 1: Fix Double Strength Application

**Goal**: Remove strength from sub-effects and apply it only in the final mix for linear response.

**Scope**: `bdip_core/src/gpu/shaders/instamatic/instamatic.wgsl`

**Implementation**:

Replace the current approach where each sub-effect multiplies by `params.strength` with a pattern
that computes the full effect first, then blends with original:

```wgsl
// ── Shadow lift toward milky grey ────────────────────────────────────────
let lift_target    = vec3<f32>(0.055, 0.048, 0.038);
let shadow_weight  = clamp(1.0 - lum / 0.35, 0.0, 1.0);
let lifted         = rgb + shadow_weight * lift_target;  // Remove params.strength

// ── Highlight compression ────────────────────────────────────────────────
let highlight_scale = 0.94;  // Fixed value, not scaled by strength
let faded           = lifted * highlight_scale;

// ── Yellow-green midtone cast ────────────────────────────────────────────
let midtone_weight = (1.0 - abs(lum - 0.45) / 0.45) * clamp(lum / 0.1, 0.0, 1.0);
let midtone_cast   = vec3<f32>(0.04, 0.05, -0.06);
let cast_rgb       = faded + midtone_weight * midtone_cast;  // Remove params.strength

// ── Global warm channel balance ──────────────────────────────────────────
let channel_scale = vec3<f32>(1.04, 1.02, 0.90);  // Fixed values
let balanced = cast_rgb * channel_scale;

// ── Slight desaturation ──────────────────────────────────────────────────
let desaturated = mix(balanced, vec3<f32>(luminance(balanced)), 0.08);

// ── Radial vignette ──────────────────────────────────────────────────────
let center_dist   = length(uv - vec2<f32>(0.5));
let vignette_mask = smoothstep(0.25, 0.75, center_dist);
let vignette_amt  = 0.35 * vignette_mask;  // Remove params.strength
let vignetted     = desaturated * (1.0 - vignette_amt);

// ── Final blend with original ────────────────────────────────────────────
let out_rgb = mix(rgb, vignetted, params.strength);  // Single strength application
```

**Note on desaturation luminance**: The current implementation uses original `lum` for desaturation.
The fix above changes this to use `luminance(balanced)` to compute desaturation from the processed
color, which is more accurate. However, using original luminance is also acceptable — choose
whichever produces a better visual result during implementation.

**Update comment**: Remove or update the comment at lines 120-125 that justifies the double
application:

```wgsl
// ── Final blend with original ────────────────────────────────────────────
// At strength=0 the output equals the original (identity).
// At strength=1 the full Instamatic look is applied.
let out_rgb = mix(rgb, vignetted, params.strength);
```

**Tests to verify** (existing tests should continue to pass):
- `test_instamatic_zero_strength_is_identity` — Must still return original at strength=0
- `test_instamatic_full_strength_warms_image` — Must still warm at strength=1
- `test_instamatic_full_strength_lifts_shadows` — Must still lift shadows at strength=1
- `test_instamatic_full_strength_compresses_highlights` — Must still compress at strength=1
- `test_instamatic_vignette_darkens_corners` — Must still darken corners at strength=1
- `test_instamatic_alpha_preserved` — Must still preserve alpha

**New test to add**:

```rust
#[test]
fn test_instamatic_half_strength_produces_intermediate_result() {
    // Verify linear response: half strength should produce approximately
    // half the difference between original and full-strength output.
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    let img = make_solid_image(8, 8, 32767, 32767, 32767);
    
    let out_full = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "instamatic",
            values: vec![1.0],
        }],
    );
    
    let out_half = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "instamatic",
            values: vec![0.5],
        }],
    );
    
    // For each channel, half-strength should be approximately midway
    // between original (32767) and full-strength output.
    let orig_r = 32767i32;
    let full_r = out_full.get_pixel(4, 4)[0] as i32;
    let half_r = out_half.get_pixel(4, 4)[0] as i32;
    let expected_half_r = (orig_r + full_r) / 2;
    
    // Allow 15% tolerance for non-linear sub-effect interactions
    let tolerance = ((full_r - orig_r).abs() as f32 * 0.15) as i32;
    assert!(
        (half_r - expected_half_r).abs() <= tolerance,
        "half strength should be approximately midway: orig={}, half={}, full={}, expected_half={}±{}",
        orig_r, half_r, full_r, expected_half_r, tolerance
    );
}
```

---

## Test Specifications

### `test_instamatic_half_strength_produces_intermediate_result`

- **Purpose**: Verify the strength parameter has a linear response curve
- **Setup**: Create 8×8 mid-gray image (32767, 32767, 32767)
- **Actions**:
  1. Process with strength=1.0, record center pixel values
  2. Process with strength=0.5, record center pixel values
- **Assertions**: Half-strength output should be approximately midway between original and
  full-strength output (within 15% tolerance to account for non-linear sub-effect interactions)

---

## Validation Checklist

After PR is merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (all existing instamatic tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] At strength=0.5, the effect is visibly noticeable (not barely visible as before)
- [ ] At strength=1.0, the effect matches the previous full-strength appearance
- [ ] Slider feels responsive across its full range

---

## References

- [Kodacolor (still photography) - Wikipedia](https://en.wikipedia.org/wiki/Kodacolor_(still_photography))
- [Kodak Instamatic - Wikipedia](https://en.wikipedia.org/wiki/Instamatic)
- [Minimal Color Grading Tools - Filmic Worlds](http://filmicworlds.com/blog/minimal-color-grading-tools/)
- [Film Emulation - Wikipedia](https://en.wikipedia.org/wiki/Film_emulation)
- [Vintage Film Lightroom Preset Guide - The Editing Studio](https://theeditingstudio.co/blog/vintage-clean-film-lightroom-preset-guide)
