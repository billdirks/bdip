# Fix Pastel Dreams Transform

## Problem Summary

The `pastel_dreams` transform has one moderate issue and two minor issues that prevent it from
producing authentic pastel output at full strength.

### Moderate Issues

1. **Full strength produces grayscale instead of pastel** (lines 43-44 in shader): At `strength=1.0`,
   `sat_scale = 1 - strength = 0`, which means the output is fully desaturated to lifted luminance.
   True pastel colors retain some hue — they're defined as having "low to medium saturation," not
   zero saturation. The effect produces washed-out gray rather than soft pastel tones.

   Current behavior at strength=1.0:
   ```wgsl
   let sat_scale = 1.0 - params.strength;  // = 0.0
   let desaturated_rgb = mix(vec3<f32>(luma_clamped), lifted_rgb, sat_scale);  // = luma_clamped
   ```

   This outputs pure grayscale (the lifted luminance value), losing all color information.

### Minor Issues

2. **Single parameter limits artistic control**: The effect has only a "strength" slider that
   controls both brightness lift and desaturation in lockstep. Users cannot separately adjust:
   - How much to lift brightness toward white
   - How much to reduce saturation

3. **Additive brightness lift clips highlights abruptly**: The current `color.rgb + brightness_lift`
   approach can push near-white colors above 1.0 immediately. A multiplicative mix toward white
   (`mix(color.rgb, vec3(1.0), lift)`) would be smoother, lifting darker colors more than lighter
   ones while preserving highlight headroom.

### What's Working Correctly

- Luminance calculation uses correct Rec. 709 coefficients
- Alpha channel is preserved
- Identity behavior at strength=0.0
- The mathematical relationship between lifted luminance and lifted RGB is correct

### References

- [Pastel (color) - Wikipedia](https://en.wikipedia.org/wiki/Pastel_(color)): "Pastel colors,
  when described in the HSV color space, have high value and low or medium saturation."
- [HSL and HSV - Wikipedia](https://en.wikipedia.org/wiki/HSL_and_HSV): Background on color space
  transformations
- [How to Create a Soft Pastel Look - BWillCreative](https://www.bwillcreative.com/how-to-create-a-soft-pastel-look-with-lightroom/):
  Practical guidance on pastel post-processing

---

## Implementation Plan

### PR 1: Fix Desaturation to Retain Color at Full Strength

**Goal**: Ensure the pastel effect retains some hue even at full strength, producing true pastel
tones instead of grayscale.

**Scope**:
- Modify `pastel_dreams.wgsl` to cap desaturation at ~70% instead of 100%
- Update shader comments to explain the design decision

**Implementation Details**:

Change the saturation scale calculation to retain minimum 30% saturation:

```wgsl
// Before:
let sat_scale = 1.0 - params.strength;

// After: Cap desaturation at 70% to retain pastel hues
let sat_scale = max(0.3, 1.0 - params.strength);
```

Alternatively, use a softer curve that retains more color at high strength:

```wgsl
// Quadratic falloff retains more color: at strength=1.0, sat_scale=0.3
let sat_scale = mix(1.0, 0.3, params.strength);
```

The second approach is simpler and more intuitive — strength=1.0 gives 30% saturation retention.

**Tests to Add**:

1. `test_pastel_dreams_retains_hue_at_full_strength`: Verify that at strength=1.0, a saturated
   color input still has channel variation (not grayscale).

**Tests to Update**:

- `test_pastel_dreams_reduces_saturation_at_full_strength`: Update assertion to verify saturation
  is reduced but not eliminated.

---

### PR 2: Use Multiplicative Brightness Lift for Smoother Highlights

**Goal**: Replace additive brightness lift with multiplicative mix toward white for smoother
highlight behavior.

**Scope**:
- Modify `pastel_dreams.wgsl` to use `mix(color.rgb, vec3(1.0), lift_amount)` instead of
  `color.rgb + lift_amount`
- Adjust lift amount to compensate for the different behavior

**Implementation Details**:

```wgsl
// Before:
let brightness_lift = params.strength * 0.5;
let lifted_rgb = color.rgb + vec3<f32>(brightness_lift);

// After: Multiplicative mix toward white (smoother for highlights)
let lift_amount = params.strength * 0.6;  // Adjusted for similar visual impact
let lifted_rgb = mix(color.rgb, vec3<f32>(1.0), lift_amount);
```

The multiplicative approach has the property:
- Dark colors (0.0) → lifted by full `lift_amount`
- Bright colors (1.0) → unchanged (already at white)
- Mid colors → proportionally lifted

This prevents harsh clipping on near-white inputs while still lifting shadows and midtones.

**Tests to Add**:

1. `test_pastel_dreams_highlights_not_clipped`: Verify that near-white input (e.g., 0.95 linear)
   doesn't clip to exactly 1.0 but stays near its original value.

---

### PR 3 (Optional): Add Separate Saturation Parameter

**Goal**: Allow independent control of brightness lift and saturation reduction for more artistic
flexibility.

**Scope**:
- Add `saturation` parameter (0.0-1.0, default 0.5) to control desaturation amount
- Update `PastelDreamsParams` struct
- Update slider definitions in `mod.rs`
- Update shader to use new parameter

**New Parameters**:

```rust
pub struct PastelDreamsParams {
    pub strength: f32,     // 0.0-1.0, default 0.0 - overall effect intensity
    pub saturation: f32,   // 0.0-1.0, default 0.5 - color retention (0=grayscale, 1=full color)
    pub _padding: [f32; 2],
}
```

**Slider Definitions**:

```rust
const PARAM: ParamKind = ParamKind::Sliders(&[
    SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Overall effect intensity. 0 leaves the image unchanged.",
    },
    SliderDef {
        name: "Color Retention",
        min: 0.0,
        max: 1.0,
        default: 0.5,
        description: "How much original color to retain. 0 produces grayscale pastels; \
                      1 keeps full saturation with only brightness lifted.",
    },
]);
```

**Shader Changes**:

```wgsl
// Replace fixed 0.3 minimum with parameter-controlled retention
let min_saturation = params.saturation;
let sat_scale = mix(1.0, min_saturation, params.strength);
```

**Tests to Add**:

1. `test_pastel_dreams_saturation_zero_produces_grayscale`: At saturation=0.0, strength=1.0,
   output should be grayscale.
2. `test_pastel_dreams_saturation_one_preserves_hue`: At saturation=1.0, strength=1.0, output
   should have same hue as input (only brightness changed).

---

## Test Specifications

### PR 1 Tests

#### `test_pastel_dreams_retains_hue_at_full_strength`

```rust
/// Verify that at full strength, the output retains some color (not grayscale).
#[test]
fn test_pastel_dreams_retains_hue_at_full_strength() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Pure red input
    let img = make_solid_image(2, 2, 65535, 0, 0);
    let out_img = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "pastel_dreams",
            values: vec![1.0],
        }],
    );

    let pixel = out_img.pixels().next().unwrap();
    let r = pixel[0] as i32;
    let g = pixel[1] as i32;
    let b = pixel[2] as i32;

    // R channel should still be higher than G and B (hue retained)
    assert!(
        r > g && r > b,
        "pastel effect should retain red hue: R={}, G={}, B={}",
        r, g, b
    );

    // But the spread should be less than the original (saturation reduced)
    let spread = r - g.min(b);
    assert!(
        spread < 30000,
        "saturation should be reduced but not eliminated: spread={}",
        spread
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (including new tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] At strength=0.0, image is unchanged (identity)
- [ ] At strength=1.0, saturated colors become soft pastels (not grayscale)
- [ ] At strength=1.0, midtones are lifted toward white
- [ ] White input stays near white
- [ ] Black input is lifted (not pure black)
- [ ] Alpha channel preserved at all strength values
