# Fix Coffee Stained Transform

## Problem Summary

The `coffee_stained` transform has a fundamental visual inaccuracy: it simulates stains that are
darkest at the center and fade outward, but real dried coffee stains exhibit the opposite pattern
due to the well-documented "coffee ring effect" (Wikipedia, ScienceABC).

### Critical Issues

1. **Inverted stain pattern**: The shader uses exponential falloff from blob centers
   (`exp(-d * BLOB_SCALE)` at line 79), making the mask strongest at centers. Real coffee stains
   are darker at the **edges** (ring effect) because evaporating liquid carries particles outward
   via capillary flow. The center dries relatively clear.

   Current behavior:
   - Blob center → mask ≈ 1.0 → maximum darkening
   - Blob edge → mask → 0.0 → no darkening

   Expected behavior for coffee ring effect:
   - Ring perimeter → maximum darkening (particle concentration)
   - Ring center → minimal darkening (liquid carried particles outward)

### Moderate Issues

2. **No ring radius/thickness control**: Real coffee stains vary in ring thickness and inner
   clarity. The shader has no parameters to adjust these characteristics.

3. **Limited parameter set**: Only `strength` is exposed (line 17-23). Missing useful controls:
   - Blob/ring size variation
   - Ring edge sharpness
   - Inner clarity (how light the center is vs. the edge)

### Minor Issues

4. **Fixed blob positions**: The 7 blob centers (lines 61-67) are hard-coded. While this ensures
   determinism, adding an optional position offset or seed could provide variety without
   sacrificing reproducibility.

5. **Stain color is reasonable**: The tint `(0.45, 0.25, 0.10)` in linear light corresponds
   approximately to `#B88559` in sRGB, which is within the range of realistic coffee brown tones
   (research shows coffee browns from `#6F4E37` to `#8A624A`). No change needed here.

### Current Parameters

| Parameter | Range   | Issue |
|-----------|---------|-------|
| Strength  | 0.0–1.0 | OK    |

### Missing Parameters

- **Ring Width**: Controls thickness of the dark ring edge
- **Inner Clarity**: How much lighter the center is compared to the ring edge

---

## Implementation Plan

### PR 1: Implement Coffee Ring Effect

**Goal**: Replace center-darkest falloff with edge-darkest ring pattern that matches real coffee
stain physics.

**Scope**:
- Rewrite `blob()` function in `coffee_stained.wgsl` to produce ring-shaped masks instead of
  center-heavy blobs
- Add `ring_width` parameter to control the thickness of the dark edge
- Add `inner_clarity` parameter to control how light the center is
- Update `CoffeeStainedParams` struct with new fields
- Update `mod.rs` with new slider definitions

**New Parameters**:

```rust
pub struct CoffeeStainedParams {
    pub strength: f32,      // 0.0–1.0, default 0.0 (identity)
    pub ring_width: f32,    // 0.0–1.0, default 0.3 (relative ring thickness)
    pub inner_clarity: f32, // 0.0–1.0, default 0.7 (how clear the center is)
    pub _padding: f32,
}
```

**New Shader Algorithm** (pseudocode):

```wgsl
// Ring-shaped stain mask:
// - Maximum at a specific radius from center (the ring edge)
// - Falls off toward both the center AND outward from the ring

fn ring_blob(uv: vec2<f32>, centre: vec2<f32>, ring_radius: f32, ring_width: f32) -> f32 {
    let d = distance(uv, centre);
    
    // Distance from the ring edge (not from center)
    let dist_from_ring = abs(d - ring_radius);
    
    // Smooth falloff from ring edge
    let ring_intensity = exp(-dist_from_ring * (10.0 / ring_width));
    
    return ring_intensity;
}

fn stain_mask(uv: vec2<f32>, ring_width: f32, inner_clarity: f32) -> f32 {
    // Each blob has a characteristic ring radius
    let raw = ring_blob(uv, CENTRE_0, 0.15, ring_width)
            + ring_blob(uv, CENTRE_1, 0.12, ring_width)
            // ... etc for all 7 centres with varying ring radii
    
    let clamped = min(raw, 1.0);
    
    // Inner clarity: reduce darkening inside the ring
    // (handled via a separate inner falloff term, or by the ring_blob structure itself)
    
    return pow(clamped, 0.6);
}
```

**Key Changes**:
1. `blob()` → `ring_blob()`: Returns maximum at `ring_radius` distance from center, not at center
2. Each blob center now has an associated ring radius (hardcoded or derived from position)
3. `ring_width` controls how thick/diffuse the ring edge is
4. `inner_clarity` can blend between ring-only (clear center) and filled (current behavior)

**Tests to Add**:

1. `test_coffee_stained_ring_effect_darker_at_edge`: On a white input with full strength, verify
   that pixels at ring-edge distance from blob centers are darker than pixels at blob centers.

2. `test_coffee_stained_ring_width_affects_edge_thickness`: Compare output at ring_width=0.1 vs
   ring_width=0.5 — wider ring should affect more pixels around the ring edge.

3. `test_coffee_stained_inner_clarity_affects_center`: At inner_clarity=1.0, center of stains
   should be nearly unchanged. At inner_clarity=0.0, center should be darker (approaching current
   filled-blob behavior).

**Existing Tests to Preserve**:
- `test_coffee_stained_zero_strength_is_identity` — must still pass
- `test_coffee_stained_alpha_preserved` — must still pass
- `test_coffee_stained_deterministic` — must still pass
- `test_coffee_stained_chaining_with_brightness` — must still pass

**Tests to Update**:
- `test_coffee_stained_full_strength_warms_image` — may need updated expectations
- `test_coffee_stained_full_strength_darkens_image` — may need updated expectations

---

### PR 2: Documentation and Parameter Help

**Goal**: Ensure user-facing documentation explains the coffee ring effect clearly.

**Scope**:
- Update `DESCRIPTION` in `CoffeeStainedParams` to explain the ring effect
- Ensure all slider descriptions explain the visual impact
- Update any related documentation

**Updated Description**:

```rust
const DESCRIPTION: &'static str = "Simulates realistic coffee or tea stains with the characteristic \
    ring effect where the stain edge is darker than the center, matching how real coffee dries \
    via capillary flow.";
```

**Slider Descriptions**:

```rust
SliderDef {
    name: "Strength",
    min: 0.0,
    max: 1.0,
    default: 0.0,
    description: "Blend between original image (0.0) and coffee-stained effect (1.0).",
},
SliderDef {
    name: "Ring Width",
    min: 0.0,
    max: 1.0,
    default: 0.3,
    description: "Thickness of the dark ring edge. Lower values produce thin, defined edges; \
        higher values spread the darkening wider.",
},
SliderDef {
    name: "Inner Clarity",
    min: 0.0,
    max: 1.0,
    default: 0.7,
    description: "How clear the center of each stain is. 1.0 = center nearly unchanged \
        (realistic ring); 0.0 = center also darkened (filled stain).",
},
```

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_coffee_stained_ring_effect_darker_at_edge`

```rust
/// Verify the coffee ring effect: edges of stain blobs are darker than centers.
/// This matches real coffee physics where particles concentrate at the perimeter.
#[test]
fn test_coffee_stained_ring_effect_darker_at_edge() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // White input to clearly see darkening pattern
    let img = make_solid_image(128, 128, 65535, 65535, 65535);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "coffee_stained",
            values: vec![1.0, 0.3, 0.7], // strength, ring_width, inner_clarity
        }],
    );

    // Sample near blob center (CENTRE_0 = 0.18, 0.22 → pixel ~23, 28 in 128x128)
    // and at ring edge distance (~0.15 radius → ~19 pixels away)
    let center_pixel = out.get_pixel(23, 28);
    let edge_pixel = out.get_pixel(23 + 19, 28); // approximately at ring edge

    // Ring edge should be darker (lower values) than center
    // Comparing luminance (or just R channel since it's warmed)
    assert!(
        edge_pixel[0] < center_pixel[0],
        "ring edge should be darker than center: edge R={}, center R={}",
        edge_pixel[0], center_pixel[0]
    );
}
```

#### `test_coffee_stained_ring_width_affects_edge_thickness`

```rust
/// Verify ring_width parameter controls the thickness of the dark edge.
#[test]
fn test_coffee_stained_ring_width_affects_edge_thickness() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(64, 64, 65535, 65535, 65535);

    let out_thin = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "coffee_stained",
            values: vec![1.0, 0.1, 0.8], // thin ring
        }],
    );

    let out_wide = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "coffee_stained",
            values: vec![1.0, 0.5, 0.8], // wide ring
        }],
    );

    // Count pixels that are noticeably darkened (< 90% of white)
    let count_darkened = |img: &Rgba16Image| {
        img.pixels().filter(|p| p[0] < 59000).count()
    };

    let thin_darkened = count_darkened(&out_thin);
    let wide_darkened = count_darkened(&out_wide);

    assert!(
        wide_darkened > thin_darkened,
        "wider ring should darken more pixels: thin={}, wide={}",
        thin_darkened, wide_darkened
    );
}
```

#### `test_coffee_stained_inner_clarity_affects_center`

```rust
/// Verify inner_clarity controls how light the stain center is.
#[test]
fn test_coffee_stained_inner_clarity_affects_center() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(64, 64, 65535, 65535, 65535);

    // High inner clarity = clear center
    let out_clear = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "coffee_stained",
            values: vec![1.0, 0.3, 1.0], // inner_clarity = 1.0
        }],
    );

    // Low inner clarity = darkened center
    let out_filled = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "coffee_stained",
            values: vec![1.0, 0.3, 0.0], // inner_clarity = 0.0
        }],
    );

    // Sample near a blob center
    let center_clear = out_clear.get_pixel(23, 28)[0];
    let center_filled = out_filled.get_pixel(23, 28)[0];

    assert!(
        center_clear > center_filled,
        "high inner_clarity should leave center lighter: clear={}, filled={}",
        center_clear, center_filled
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Coffee stain effect shows darker rings at stain edges (not centers)
- [ ] Ring width parameter visibly affects edge thickness
- [ ] Inner clarity at 1.0 leaves stain centers nearly unchanged
- [ ] Inner clarity at 0.0 produces filled/solid stains (similar to old behavior)
- [ ] Strength=0 returns source unchanged (identity)
- [ ] Alpha channel is preserved

---

## References

- [Coffee ring effect - Wikipedia](https://en.wikipedia.org/wiki/Coffee_ring_effect)
- [Why Are Coffee Stains Darker Along The Edges? - ScienceABC](https://www.scienceabc.com/pure-sciences/why-are-coffee-stains-darker-along-the-edges.html)
- [12.4. Coffee Stain - GIMP Documentation](https://docs.gimp.org/2.10/en/script-fu-coffee-stain.html)
- [Coffee Brown Color Codes](https://colorcodes.io/brown/coffee-brown-color-codes-2/)
- [DIY: How to Create a Coffee-Stained Texture - Digital Photography School](https://digital-photography-school.com/coffee-stained-texture-photoshop/)
