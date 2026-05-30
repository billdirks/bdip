# Fix Daguerreotype Transform

## Problem Summary

The daguerreotype transform is largely well-implemented but has one algorithmic bug in the vignette
aspect ratio correction that affects non-square images.

### Critical Issues

1. **Vignette aspect ratio correction is inverted** (`daguerreotype_pass1.wgsl:71-73`): The formula
   multiplies the Y coordinate by aspect ratio instead of X, producing an incorrectly oriented
   elliptical vignette for non-square images.

   Current (incorrect):
   ```wgsl
   let d_sq = centered.x * centered.x + (centered.y * aspect) * (centered.y * aspect);
   ```

   For a 2:1 landscape image, this makes vertical distances count double, so the top/bottom edges
   receive more vignette darkening than the left/right edges. But physically, the left/right edges
   are further from the image center and should receive more vignette (assuming a circular lens
   vignette).

   Correct formula:
   ```wgsl
   let d_sq = (centered.x * aspect) * (centered.x * aspect) + centered.y * centered.y;
   ```

   **Note**: The tintype shader (`tintype_pass1.wgsl:41-43`) has the same bug and should be fixed
   separately.

### Moderate Issues

None identified. The core algorithm correctly implements:
- Rec. 709 luminance coefficients for desaturation
- S-curve contrast boost via cubic Hermite interpolation
- Blue-grey metallic tint (R×0.94, G×0.97, B×1.06)
- Luminance-weighted procedural grain (brighter areas get more grain, matching silver-halide
  physics where dense silver regions trap more particles)
- Proper strength-based blending with the original image

### Minor Issues

1. **Single parameter limits creative control**: The shader exposes only "Strength". Additional
   parameters could allow independent control of:
   - Grain intensity
   - Vignette intensity
   - Contrast boost amount
   - Tint intensity

   However, this is a design choice—a single parameter simplifies the UI and the current defaults
   produce a historically plausible daguerreotype look. Not flagged as a bug.

### Historical Accuracy Notes

The implementation captures key characteristics of daguerreotype photography:
- **High contrast**: The S-curve contrast boost (0.65 mix of smoothstep) approximates the harsh
  tonal compression of silver-salt emulsions
- **Silver-blue tint**: The metallic tint shifts neutral grey toward cool blue-grey (B > R),
  matching the reflective silver surface of actual daguerreotypes
- **Vignette**: Strong radial darkening (start 0.38, end 0.80) simulates early lens optics
- **Fine grain**: Luminance-weighted hash noise (0.015 amplitude) simulates silver particle
  distribution

**Not simulated** (would require advanced rendering):
- Angle-dependent positive/negative appearance (daguerreotypes appear positive or negative
  depending on viewing angle due to their mirror-like surface)
- Edge tarnishing (characteristic brown→iridescent blue→black oxidation pattern around edges)

References:
- [Daguerreotype - Wikipedia](https://en.wikipedia.org/wiki/Daguerreotype)
- [The Daguerreotype Medium - Library of Congress](https://www.loc.gov/collections/daguerreotypes/articles-and-essays/the-daguerreotype-medium/)
- [Daguerreotype | Britannica](https://www.britannica.com/technology/daguerreotype)
- [The Degradation of Daguerreotypes - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10181581/)

---

## Implementation Plan

### PR 1: Fix Vignette Aspect Ratio Correction

**Goal**: Correct the vignette formula so the falloff is circular in physical image space.

**Scope**:
- Edit `daguerreotype_pass1.wgsl` lines 71-73

**Implementation**:

Change from:
```wgsl
let d_sq       = centered.x * centered.x + (centered.y * aspect) * (centered.y * aspect);
```

To:
```wgsl
let d_sq       = (centered.x * aspect) * (centered.x * aspect) + centered.y * centered.y;
```

The comment on line 69-70 should also be updated since it currently says "elliptical distance that
accounts for non-square images" but with the fix, this becomes a proper circular distance in
physical space.

Update comment to:
```wgsl
// Circular radial distance in physical image space. The aspect ratio scales
// horizontal distance so that corners equidistant from center (in pixels)
// receive equal vignette darkening regardless of image dimensions.
```

**Tests to Add**:

1. `test_daguerreotype_vignette_symmetric_on_landscape`: On a landscape image (e.g., 64×32),
   verify that the top/bottom edge centers are brighter than the left/right edge centers,
   since left/right edges are physically further from center.

2. `test_daguerreotype_vignette_symmetric_on_portrait`: On a portrait image (e.g., 32×64),
   verify that the left/right edge centers are brighter than the top/bottom edge centers.

**Existing Tests to Update**:
- `test_daguerreotype_vignette_darkens_corners`: Already passes (tests corners vs center on a
  square 32×32 image). Keep as-is.

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_daguerreotype_vignette_symmetric_on_landscape`

```rust
/// On a landscape image, the horizontal edges (left/right) are physically further
/// from center than vertical edges (top/bottom). After the aspect ratio fix, the
/// horizontal edges should be darker due to the circular vignette.
#[test]
fn test_daguerreotype_vignette_symmetric_on_landscape() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // 64×32 landscape image (2:1 aspect ratio)
    let img = make_solid_image(64, 32, 40000, 40000, 40000);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "daguerreotype",
            values: vec![1.0],
        }],
    );

    // Sample edge center pixels:
    // - Top edge center: (32, 0) - physically 16 pixels from center vertically
    // - Right edge center: (63, 16) - physically 32 pixels from center horizontally
    let top_edge = out.get_pixel(32, 0)[0] as i32;
    let right_edge = out.get_pixel(63, 16)[0] as i32;

    // Right edge is 2x further from center physically, so should be darker
    assert!(
        right_edge < top_edge,
        "right edge (further from center) must be darker than top edge: \
         right={right_edge}, top={top_edge}"
    );
}
```

#### `test_daguerreotype_vignette_symmetric_on_portrait`

```rust
/// On a portrait image, the vertical edges (top/bottom) are physically further
/// from center than horizontal edges (left/right). The vertical edges should
/// be darker due to the circular vignette.
#[test]
fn test_daguerreotype_vignette_symmetric_on_portrait() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // 32×64 portrait image (1:2 aspect ratio)
    let img = make_solid_image(32, 64, 40000, 40000, 40000);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "daguerreotype",
            values: vec![1.0],
        }],
    );

    // Sample edge center pixels:
    // - Right edge center: (31, 32) - physically 16 pixels from center horizontally
    // - Bottom edge center: (16, 63) - physically 32 pixels from center vertically
    let right_edge = out.get_pixel(31, 32)[0] as i32;
    let bottom_edge = out.get_pixel(16, 63)[0] as i32;

    // Bottom edge is 2x further from center physically, so should be darker
    assert!(
        bottom_edge < right_edge,
        "bottom edge (further from center) must be darker than right edge: \
         bottom={bottom_edge}, right={right_edge}"
    );
}
```

---

## Validation Checklist

After PR 1 is merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (including new vignette symmetry tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] On a 2:1 landscape test image, the left/right edges appear darker than top/bottom edges
- [ ] On a 1:2 portrait test image, the top/bottom edges appear darker than left/right edges
- [ ] Existing test `test_daguerreotype_vignette_darkens_corners` still passes

---

## Out of Scope

- **Tintype vignette bug**: The tintype shader (`tintype_pass1.wgsl:41-43`) has the identical
  aspect ratio bug. This should be fixed in a separate audit/PR to keep changes focused.
