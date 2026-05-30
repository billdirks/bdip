# Fix Highlights Transform

## Problem Summary

The highlights transform has a description/implementation mismatch regarding specular protection, plus
confusing parameter semantics.

### Critical Issue

1. **Specular protection not implemented**: The "End" parameter description claims "Luminance above
   which the highlight weight tapers to zero, protecting specular whites." However, the actual
   implementation uses `smoothstep(range, end, L)`, which saturates at 1.0 for L >= end. Specular
   whites (L approaching 1.0) receive **full** adjustment, not protection.

   Current behavior (highlights.wgsl:22):
   ```wgsl
   let w_h = smoothstep(params.range, params.end, L);  // 1.0 at L >= 0.95
   ```
   
   Expected behavior per description: w_h should taper toward zero as L approaches 1.0.

### Moderate Issues

2. **Parameter name confusion**: "Range" and "End" are unintuitive. They define a smoothstep
   transition band, but the names don't clearly convey which boundary is which. Compare to the
   shadows shader which uses "Start" and "Range" - also not ideal, but different naming creates
   inconsistency between complementary tools.

3. **Shadows shader has the same bug**: For symmetry reference, the shadows shader description says
   "Luminance below which the shadow weight tapers to zero, protecting pure blacks" but
   `1.0 - smoothstep(start, range, L)` applies full effect at L=0. However, fixing shadows is
   out of scope for this plan.

### Minor Issues

4. **Harsh clipping when brightening**: The multiplicative formula `color * (1 + amt * w_h)` can
   push values well above 1.0 when amt=1.0, which then hard-clips. A soft rolloff near clipping
   would be more professional. (Nice-to-have, not critical.)

### Current Parameters

| Parameter | Range     | Default | Issue |
|-----------|-----------|---------|-------|
| Amount    | -1.0–1.0  | 0.0     | OK    |
| Range     | 0.0–1.0   | 0.6     | Confusing name; this is the lower bound of the transition |
| End       | 0.0–1.0   | 0.95    | Description promises specular protection; implementation doesn't deliver |

---

## Implementation Plan

### PR 1: Add Specular Protection

**Goal**: Implement the specular protection described in the parameter documentation.

**Scope**:
- Modify `highlights.wgsl` to taper w_h back toward zero as luminance approaches 1.0
- Keep parameter defaults and ranges unchanged (0.95 default end is reasonable)
- Add a "shoulder" transition above `end` that brings w_h back down

**Algorithm Change** (pseudocode):

```wgsl
// Current (broken):
let w_h = smoothstep(params.range, params.end, L);

// Fixed: rise through highlights, fall for specular protection
// Compute shoulder from end to 1.0
let w_h_rise = smoothstep(params.range, params.end, L);
let w_h_fall = 1.0 - smoothstep(params.end, 1.0, L);
let w_h = w_h_rise * w_h_fall;
```

With default end=0.95:
- L < 0.6: w_h = 0 (no effect on midtones/shadows)
- L = 0.775: w_h peaks near 1.0 (full effect on mid-highlights)
- L = 0.95: w_h starts falling (transition to specular protection)
- L = 1.0: w_h = 0 (specular whites protected)

**Tests to Add**:

1. `test_highlights_specular_protection`: Pure white input (65535) should be minimally affected
   even with max negative amount, since specular whites are protected.

2. `test_highlights_mid_highlights_affected`: Input at ~75% luminance should be strongly affected
   (verify w_h is high in the highlight band, not just at extremes).

**Existing Tests to Update** (may need assertion tolerance adjustments):
- `test_highlights_darkens_bright_pixels`: Still valid; bright (but not specular) pixels darken
- Other tests should pass unchanged

---

### PR 2: Rename Parameters for Clarity

**Goal**: Align parameter names with the shadows shader for consistency, and clarify semantics.

**Scope**:
- Rename `range` to `start` and `end` to `range` to match shadows naming convention
- OR adopt clearer names: `transition_start` and `transition_end`
- Update `mod.rs` parameter definitions
- Update `highlights.wgsl` uniform struct
- Update all tests using parameter value arrays

**Option A (match shadows convention)**:
```rust
pub struct HighlightsParams {
    pub amt: f32,
    pub start: f32,   // was: range — lower luminance boundary
    pub end: f32,     // was: end — upper luminance boundary (keep name)
    pub _padding: f32,
}
```

**Option B (clearer names)**:
```rust
SliderDef {
    name: "Transition Start",
    description: "Lower luminance boundary where highlight adjustment begins.",
},
SliderDef {
    name: "Transition End", 
    description: "Upper luminance boundary where adjustment peaks before specular rolloff.",
},
```

**Recommendation**: Option A is simpler and creates consistency with the shadows shader.

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_highlights_specular_protection`

```rust
/// Verify that specular whites (pure white) are protected from adjustment.
#[test]
fn test_highlights_specular_protection() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    
    // Pure white input (specular)
    let img = make_solid_image(2, 2, 65535, 65535, 65535);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "highlights",
            values: vec![-1.0, 0.6, 0.95], // max darkening
        }],
    );
    
    // With specular protection, pure white should remain nearly white
    // Allow small tolerance for floating-point rounding
    for pixel in out.pixels() {
        assert!(
            pixel[0] > 60000,
            "specular white should be protected, got {}",
            pixel[0]
        );
    }
}
```

#### `test_highlights_mid_highlights_affected`

```rust
/// Verify that mid-highlights (~75% luminance) receive strong adjustment.
#[test]
fn test_highlights_mid_highlights_affected() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    
    // 75% luminance (roughly in middle of highlight band)
    let val = (65535.0 * 0.75) as u16;
    let img = make_solid_image(2, 2, val, val, val);
    
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "highlights",
            values: vec![-1.0, 0.6, 0.95], // max darkening
        }],
    );
    
    // Mid-highlights should be significantly darkened
    for pixel in out.pixels() {
        assert!(
            pixel[0] < val - 5000,
            "mid-highlights should be darkened, input {} output {}",
            val, pixel[0]
        );
    }
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Amount=0 returns source unchanged (identity)
- [ ] Pure white input is minimally affected (specular protection working)
- [ ] Mid-highlight input (75% luminance) is strongly affected
- [ ] Bright but non-specular input (90% luminance) is moderately affected
- [ ] Alpha channel preserved through adjustment

---

## References

- [Adjust shadow and highlight detail in Photoshop](https://helpx.adobe.com/photoshop/using/adjust-shadow-highlight-detail.html)
- [Editing Highlights & Shadows in Lightroom](https://jenbilodeauphotography.com/2020/12/editing-highlights-shadows-in-lightroom-the-difference-between-the-tone-curve-the-basic-panel/)
- [Digital Image Processing: Point Operations](https://www.allaboutcircuits.com/technical-articles/digital-image-processing-point-operations/)
