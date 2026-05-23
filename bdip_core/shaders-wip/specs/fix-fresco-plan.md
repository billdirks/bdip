# Fix Fresco Transform

## Problem Summary

The fresco shader correctly implements matte desaturation, contrast reduction, and grain overlay, but
it has one critical issue that prevents it from producing an authentic Renaissance fresco look.

### Critical Issues

1. **Desaturation targets grayscale instead of warm earth tones** (lines 43-49): The matte parameter
   blends colors toward pure luminance (grayscale), but authentic frescoes were painted with natural
   earth pigments—ochres, umbers, siennas—which produce characteristic warm undertones. The shader
   should blend toward a warm earth-toned gray rather than neutral gray.

   **Current code:**
   ```wgsl
   let matte_rgb = mix(src.rgb, vec3<f32>(luma), params.matte);
   ```

   This produces a cold, generic "old photo" look rather than the warm, earthy quality of frescoes
   painted on lime plaster with iron-oxide pigments.

   **Reference**: According to [Britannica](https://www.britannica.com/art/fresco-painting) and
   [Natural Pigments](https://www.naturalpigments.com/artist-materials/history-technique-fresco-painting),
   fresco pigments were primarily earth colors—ochre (yellow-brown), umber (brown), sienna
   (red-brown)—mixed directly into wet lime plaster. Blues were problematic because neither azurite
   nor lapis lazuli worked well in the alkaline plaster environment.

### Minor Issues

2. **Inaccurate comment about contrast range** (line 56): The comment states the contrast softening
   "maps [0, 1] → [0.04, 0.96] at full strength" but the actual math produces [0.02, 0.98]:
   - At matte=1.0: soft_amount = 0.5
   - shadow_lift = 0.5 × 0.04 = 0.02
   - highlight_compress = 1.0 - 0.5 × 0.08 = 0.96
   - For pixel=0.0: 0.0 × 0.96 + 0.02 = 0.02
   - For pixel=1.0: 1.0 × 0.96 + 0.02 = 0.98

### Current Parameters

| Parameter     | Range     | Issue |
|---------------|-----------|-------|
| Strength      | 0.0–1.0   | OK    |
| Matte         | 0.0–1.0   | Desaturates to gray, not earth tones |
| Texture Scale | 0.5–4.0   | OK    |

### What Works Correctly

- Contrast softening formula (aside from comment)
- Grain overlay with fixed 0.25 blend weight
- Strength-based identity passthrough
- Alpha preservation
- Linear-light BT.709 luminance calculation
- Test coverage is comprehensive

---

## Implementation Plan

### PR 1: Add Warm Earth Tone Shift to Matte Desaturation

**Goal**: Make the matte parameter blend toward warm earth tones (ochre/sienna) rather than neutral
grayscale, producing the characteristic warm palette of fresco paintings.

**Scope**:
- Modify `fresco.wgsl` lines 43-49 to blend toward a warm earth-toned gray
- Fix the comment on line 56 to reflect the actual contrast range [0.02, 0.98]
- Update/add tests to verify warm color shift

**Implementation Details**:

The desaturation target should be a warm gray derived from luminance. Earth pigments like ochre and
sienna have approximately these ratios relative to neutral gray:
- Red: 5-10% boost
- Green: 0-5% reduction  
- Blue: 20-30% reduction

**New algorithm (pseudocode)**:

```wgsl
// ── 1. Matte desaturation toward warm earth tones ────────────────────────
//
// Unlike a standard grayscale desaturation, fresco pigments (ochre, umber,
// sienna) produce inherently warm tones. We blend toward a luminance-scaled
// warm tint rather than pure gray.
let luma = dot(src.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

// Warm earth tone coefficients: boost red, slightly reduce green, reduce blue
// These approximate the warmth of natural iron-oxide pigments
let earth_tone = vec3<f32>(luma * 1.08, luma * 0.98, luma * 0.78);
let matte_rgb = mix(src.rgb, earth_tone, params.matte);
```

The coefficients (1.08, 0.98, 0.78) are derived from the approximate color balance of raw sienna
pigment relative to neutral gray. They can be tuned based on visual testing.

**Comment fix** (line 56):

Change:
```wgsl
// The formula maps [0, 1] → [0.04, 0.96] at full strength.
```
To:
```wgsl
// The formula maps [0, 1] → [0.02, 0.98] at full strength.
```

**Tests to Add**:

1. `test_fresco_full_matte_produces_warm_tint`: Verify that a neutral gray input with full matte
   produces output where R > G > B (warm bias).

2. `test_fresco_matte_reduces_blue_channel`: Verify that the blue channel is reduced more than the
   red channel for a neutral gray input at matte=1.0.

**Existing Tests to Verify Still Pass**:

- `test_fresco_zero_strength_is_identity` — unchanged behavior
- `test_fresco_alpha_preserved` — unchanged behavior
- `test_fresco_full_matte_reduces_saturation` — may need threshold adjustment since warm tint
  still reduces saturation, just toward a warm target
- `test_fresco_texture_scale_changes_pattern` — unchanged behavior
- `test_fresco_chains_with_brightness` — unchanged behavior
- `test_fresco_deterministic` — unchanged behavior

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_fresco_full_matte_produces_warm_tint`

```rust
/// Full matte on neutral gray should produce warm output: R > G > B.
/// This verifies the earth-tone shift characteristic of fresco pigments.
#[test]
fn test_fresco_full_matte_produces_warm_tint() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    // Neutral 50% gray input
    let img = make_solid_image(4, 4, 32767, 32767, 32767);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "fresco",
            values: vec![1.0, 1.0, 1.0], // full strength, full matte
        }],
    );
    // Sample a pixel — all should have the same color on solid input
    // (grain overlay adds variation, so check mean or central pixel)
    let mut sum_r: i64 = 0;
    let mut sum_g: i64 = 0;
    let mut sum_b: i64 = 0;
    for pixel in out.pixels() {
        sum_r += pixel[0] as i64;
        sum_g += pixel[1] as i64;
        sum_b += pixel[2] as i64;
    }
    let count = (4 * 4) as i64;
    let avg_r = sum_r / count;
    let avg_g = sum_g / count;
    let avg_b = sum_b / count;
    
    assert!(
        avg_r > avg_g && avg_g > avg_b,
        "warm fresco tint should have R > G > B; got R={}, G={}, B={}",
        avg_r, avg_g, avg_b
    );
}
```

#### `test_fresco_matte_reduces_blue_channel`

```rust
/// Verify the blue channel is reduced proportionally more than red,
/// reflecting the warm earth-tone shift of fresco pigments.
#[test]
fn test_fresco_matte_reduces_blue_channel() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    // White input
    let img = make_solid_image(4, 4, 60000, 60000, 60000);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "fresco",
            values: vec![1.0, 1.0, 1.0], // full strength, full matte
        }],
    );
    
    // Compute average channels
    let mut sum_r: i64 = 0;
    let mut sum_b: i64 = 0;
    for pixel in out.pixels() {
        sum_r += pixel[0] as i64;
        sum_b += pixel[2] as i64;
    }
    let count = 16i64;
    let avg_r = sum_r / count;
    let avg_b = sum_b / count;
    
    // Blue should be reduced more than red relative to input
    // Input was equal R=B, so output should have R > B
    assert!(
        avg_r > avg_b + 1000,
        "blue should be reduced more than red; R={}, B={}, diff={}",
        avg_r, avg_b, avg_r - avg_b
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Fresco effect with matte=1.0 produces visibly warm (not cold gray) output
- [ ] Neutral gray input with full matte has R > G > B in output
- [ ] Blue channel is noticeably reduced compared to original on matte=1.0
- [ ] Strength=0 returns source unchanged (identity behavior preserved)
- [ ] Existing test `test_fresco_full_matte_reduces_saturation` still passes

---

## References

- [Fresco painting | Britannica](https://www.britannica.com/art/fresco-painting)
- [Fresco - Wikipedia](https://en.wikipedia.org/wiki/Fresco)
- [History and Technique of Fresco Painting | Natural Pigments](https://www.naturalpigments.com/artist-materials/history-technique-fresco-painting)
- [The Art of Fresco: Color - Lucia Wiley](https://www.muralist.org/fresco/color.html)
- [Base Historic Fresco Pigments List | FrescoShop.com](https://frescoshop.com/base-historic-fresco-pigments-list/)
- [Earth Colors for Oil Painting | Old Masters Academy](https://oldmasters.academy/old-masters-academy-art-lessons/earth-colors-for-oil-painting)
