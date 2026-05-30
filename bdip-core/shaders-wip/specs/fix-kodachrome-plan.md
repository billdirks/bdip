# Kodachrome Transform Audit

## Problem Summary

The current kodachrome transform uses a straightforward 3x3 color matrix that captures the general warm/saturated
character of Kodachrome film. The implementation is mathematically correct and produces a recognizable Kodachrome-style
color grade, but it misses several defining characteristics of the film stock.

### Moderate Issues

1. **No shadow-specific warmth** (lines 36-39 in `kodachrome.wgsl`)

   The color matrix is applied uniformly across all luminance levels. Kodachrome is renowned for its warm shadows — a
   characteristic achieved in the film through its unique dye-coupler chemistry. Professional emulations typically
   apply luminance-dependent color shifts, similar to how `fade_1970s.wgsl` (lines 43-49) implements shadow lift.

   **Research source**: [Kodachrome Film Lightroom Preset Guide](https://theeditingstudio.co/blog/kodachrome-film-lightroom-preset-guide) —
   "warm reds, saturated cyan-blues, slightly crushed shadows"

### Minor Issues

2. **No contrast enhancement** (entire shader)

   Kodachrome had "punchy contrast" and "high micro-contrast" that contributed to its three-dimensional feel. The
   current implementation only adjusts color balance without any contrast curve.

   **Research source**: [Kodachrome - Wikipedia](https://en.wikipedia.org/wiki/Kodachrome) — "fairly contrasty nature
   that required very accurate exposures but lent the resulting slide a pleasing 'snap'"

3. **Blue channel doesn't shift toward cyan** (line 38 in `kodachrome.wgsl`)

   The blue row `(0.05, -0.10, 1.15)` boosts blue but doesn't introduce the cyan shift that Kodachrome blues are known
   for. Kodachrome blues are "vivid and slightly shifted toward cyan rather than the warmer blue of most color negative
   film."

   **Research source**: [DPReview Forums - Kodachrome characteristics](https://www.dpreview.com/forums/thread/2836740)

### Current Parameters

| Parameter | Range   | Issue                                           |
|-----------|---------|-------------------------------------------------|
| Strength  | 0.0–1.0 | OK, but no additional creative control          |

### Missing Parameters (Optional Enhancements)

- **Shadow Warmth**: Control over shadow-specific warm tint
- **Contrast**: Control over the punchy contrast characteristic

---

## Implementation Plan

### PR 1: Add Shadow Warmth

**Goal**: Apply luminance-dependent warm tint to shadows to better match Kodachrome's characteristic warm shadow tones.

**Scope**:
- Modify `kodachrome.wgsl` to compute luminance and apply shadow-weighted warm lift
- Keep existing color matrix as the primary color grade
- Apply shadow warmth after the matrix, similar to the approach in `fade_1970s.wgsl`

**Implementation details**:

```wgsl
// After applying the color matrix:
let lum = dot(graded, vec3<f32>(0.2126, 0.7152, 0.0722));

// Warm shadow lift target — a reddish-brown tone matching Kodachrome's shadow character.
// Less aggressive than fade_1970s since Kodachrome shadows are warm but not muddy.
let shadow_lift = vec3<f32>(0.025, 0.015, 0.005);

// Shadow weight: peaks at lum=0, falls to 0 at lum=0.25.
let shadow_weight = clamp(1.0 - lum / 0.25, 0.0, 1.0);
let with_shadow_warmth = graded + shadow_weight * shadow_lift;
```

**No new parameters required** — this integrates seamlessly with the existing strength parameter.

**Tests to add**:

1. `test_kodachrome_shadows_warmer_than_highlights`: On a gradient input, verify that the warm shift is stronger in
   dark pixels than bright pixels.

2. `test_kodachrome_shadow_warmth_zero_strength_identity`: Verify that at strength=0, no shadow lift is applied
   (existing identity test should still pass, but explicitly verify shadow behavior).

---

### PR 2: Add Contrast Enhancement (Optional)

**Goal**: Add a subtle S-curve contrast boost to reproduce Kodachrome's "punchy" look.

**Scope**:
- Add a contrast parameter to `KodachromeParams` (default 0.5 for moderate contrast)
- Apply a configurable S-curve after color grading
- Update slider definitions in `mod.rs`

**New parameter**:

```rust
pub struct KodachromeParams {
    pub strength: f32,
    pub contrast: f32,  // 0.0 = no contrast boost, 1.0 = full punchy contrast
    pub _padding: [f32; 2],
}
```

**Implementation details**:

```wgsl
// Soft S-curve: t³(6t² - 15t + 10) is smoother than cubic smoothstep.
// The intensity controls the blend between linear and S-curve.
fn contrast_curve(v: f32, intensity: f32) -> f32 {
    let t = clamp(v, 0.0, 1.0);
    let s = t * t * t * (t * (6.0 * t - 15.0) + 10.0);
    return mix(v, select(s, v, v > 1.0), intensity * 0.5);
}
```

Apply per-channel after the color matrix and shadow warmth.

**Tests to add**:

1. `test_kodachrome_contrast_zero_is_flat`: At contrast=0, verify the output matches the non-contrast version.

2. `test_kodachrome_contrast_increases_range`: At contrast=1.0, verify midtones are spread — shadows slightly darker,
   highlights slightly brighter than at contrast=0.

---

### PR 3: Shift Blues Toward Cyan (Optional)

**Goal**: Adjust the color matrix to produce the cyan-shifted blues characteristic of Kodachrome.

**Scope**:
- Modify the blue row coefficients in `kodachrome.wgsl`
- Update test comments/assertions to reflect new coefficients

**Implementation details**:

Current blue row: `(0.05, -0.10, 1.15)` — pure blue boost.

Proposed blue row: `(0.00, 0.08, 1.10)` — introduces green into blue output (creating cyan shift).

For a pure blue input (0, 0, 1):
- Current: B_out = 0 - 0 + 1.15 = 1.15 (pure blue, boosted)
- Proposed: B_out = 0 + 0.08 + 1.10 = 1.18, but G_out also gets positive contribution making it cyan-shifted

Alternative: Add a small green boost to blues by adjusting the green row to have positive blue contribution, e.g.,
change green row from `(-0.10, 0.90, 0.00)` to `(-0.10, 0.88, 0.02)`. This adds a slight green boost when blue is
present, creating the cyan shift.

**Tests to add**:

1. `test_kodachrome_blue_input_has_cyan_tint`: A pure blue input should output with measurably higher green than
   identity, indicating cyan shift.

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_kodachrome_shadows_warmer_than_highlights`

```rust
/// Verify that shadow pixels receive more warm lift than highlight pixels.
#[test]
fn test_kodachrome_shadows_warmer_than_highlights() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Create an image with dark and bright regions (grey gradient).
    // Left half: dark (8000), right half: bright (56000).
    let mut img = Rgba16Image::new(4, 2);
    for y in 0..2 {
        for x in 0..2 { img.put_pixel(x, y, Rgba([8000, 8000, 8000, 65535])); }
        for x in 2..4 { img.put_pixel(x, y, Rgba([56000, 56000, 56000, 65535])); }
    }

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "kodachrome",
            values: vec![1.0],
        }],
    );

    // Measure warm shift: (R - B) delta from input.
    // Dark pixels should have larger warm shift than bright pixels.
    let dark_pixel = out.get_pixel(0, 0);
    let bright_pixel = out.get_pixel(3, 0);

    let dark_warm_shift = (dark_pixel[0] as i32 - 8000) - (dark_pixel[2] as i32 - 8000);
    let bright_warm_shift = (bright_pixel[0] as i32 - 56000) - (bright_pixel[2] as i32 - 56000);

    assert!(
        dark_warm_shift > bright_warm_shift,
        "shadows should have more warm shift: dark={} bright={}",
        dark_warm_shift, bright_warm_shift
    );
}
```

---

## Validation Checklist

After PR 1 is merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (all existing + new tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Visual check: dark image regions show warm tint at full strength
- [ ] Visual check: bright image regions are less affected by shadow warmth
- [ ] Strength=0 still returns source unchanged (identity preserved)

---

## References

- [Kodachrome - Wikipedia](https://en.wikipedia.org/wiki/Kodachrome)
- [VISNS - How Good Was Kodachrome Color Film? A Colorimetric Analysis](https://visns.neocities.org/4x5LFphotography/HGWK)
- [Kodachrome Film Lightroom Preset Guide](https://theeditingstudio.co/blog/kodachrome-film-lightroom-preset-guide)
- [DPReview Forums - Kodachrome characteristics](https://www.dpreview.com/forums/thread/2836740)
- [From Code to Kodachrome: Film Emulation from Scratch](https://articles.alexcastronovo.com/article/2/from-code-to-kodachrome-film-emulation-from-scratch)
