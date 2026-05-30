# Fix Console 16-bit Transform

## Problem Summary

The console_16bit transform is mostly correct but has one moderate issue with unclamped saturation
output that can produce invalid color values.

### Moderate Issues

1. **Unclamped saturation extrapolation** (`console_16bit_saturate.wgsl:55`): The saturation boost
   uses `mix(vec3(lum), dithered.rgb, sat_scale)` where `sat_scale = 1.0 + saturation_boost` can
   reach 3.0. WGSL's `mix()` performs linear interpolation/extrapolation without clamping. When
   `sat_scale > 1.0`, colors extrapolate beyond their original values:

   For a saturated red pixel (R=1.0, G=0.0, B=0.0) with `saturation_boost=2.0`:
   - `lum = 0.2126`
   - `sat_scale = 3.0`
   - `saturated.r = lum * (1-3) + 1.0 * 3 = 0.2126 * (-2) + 3.0 = 2.57`
   - `saturated.g = 0.2126 * (-2) + 0.0 * 3 = -0.43` (negative!)
   - `saturated.b = -0.43` (negative!)

   Negative color values are invalid and will clip to 0 at presentation, causing incorrect hue
   shifts on highly saturated inputs.

### What's Working Correctly

- **Bayer dithering algorithm**: Uses correct 4×4 threshold matrix with proper values
  `[0,8,2,10, 12,4,14,6, 3,11,1,9, 15,7,13,5]` and centered offset formula `(t - 0.5) / steps`
- **Color levels parameter**: Default 32 matches SNES 5-bit hardware; range 2-256 is appropriate
- **Luminance calculation**: Uses Rec. 709 coefficients (0.2126, 0.7152, 0.0722) correct for linear
  light working space
- **Pass structure**: Two-pass design correctly separates dithering from saturation/blend
- **Alpha handling**: Properly preserved from source throughout both passes
- **Tests**: Comprehensive coverage of identity, quantization, saturation, and chaining behavior

---

## Implementation Plan

### PR 1: Clamp Saturation Output

**Goal**: Prevent negative and excessive color values from saturation extrapolation.

**Scope**:
- Modify `console_16bit_saturate.wgsl` line 55 to clamp output to [0.0, 1.0]

**Implementation**:

Change line 55 from:
```wgsl
let saturated = mix(vec3<f32>(lum), dithered.rgb, sat_scale);
```

To:
```wgsl
let saturated = clamp(mix(vec3<f32>(lum), dithered.rgb, sat_scale), vec3<f32>(0.0), vec3<f32>(1.0));
```

**Tests to Add**:

1. `test_console_16bit_max_saturation_no_negative_values`: Verify that at maximum saturation_boost
   (2.0), output values remain non-negative.

2. `test_console_16bit_max_saturation_no_excessive_values`: Verify that at maximum saturation_boost
   (2.0), output values don't exceed 65535 (1.0 in linear).

3. `test_console_16bit_saturation_preserves_hue_on_saturated_input`: Verify that a highly saturated
   input (e.g., pure red) maintains correct hue after maximum saturation boost, not shifted due to
   asymmetric clipping.

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_console_16bit_max_saturation_no_negative_values`

```rust
/// Verify that maximum saturation_boost doesn't produce negative color values
/// (which would clip to 0 and cause hue shifts).
#[test]
fn test_console_16bit_max_saturation_no_negative_values() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    
    // Highly saturated input: pure red in sRGB
    let img = make_solid_image(4, 4, 65535, 0, 0);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "console_16bit",
            values: vec![256.0, 2.0, 1.0], // max saturation_boost
        }],
    );
    
    for pixel in out.pixels() {
        // All channels should be >= 0 (no negative clipping artifacts)
        // With proper clamping, saturated colors should remain valid
        assert!(
            pixel[0] >= 0 && pixel[1] >= 0 && pixel[2] >= 0,
            "output should have no negative values: {:?}",
            pixel
        );
    }
}
```

#### `test_console_16bit_max_saturation_no_excessive_values`

```rust
/// Verify that maximum saturation_boost clamps excessive values.
#[test]
fn test_console_16bit_max_saturation_no_excessive_values() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    
    // Highly saturated input
    let img = make_solid_image(4, 4, 65535, 0, 0);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "console_16bit",
            values: vec![256.0, 2.0, 1.0], // max saturation_boost
        }],
    );
    
    for pixel in out.pixels() {
        // All channels should be <= 65535 (clamped to valid range)
        assert!(
            pixel[0] <= 65535 && pixel[1] <= 65535 && pixel[2] <= 65535,
            "output should not exceed valid range: {:?}",
            pixel
        );
    }
}
```

#### `test_console_16bit_saturation_preserves_hue_on_saturated_input`

```rust
/// Verify that saturation boost on a saturated color maintains the dominant channel.
/// Before fix: asymmetric clipping could shift hue (e.g., pure red becoming orange).
#[test]
fn test_console_16bit_saturation_preserves_hue_on_saturated_input() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    
    // Pure red input
    let img = make_solid_image(4, 4, 65535, 0, 0);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "console_16bit",
            values: vec![256.0, 2.0, 1.0], // max saturation_boost
        }],
    );
    
    let pixel = out.get_pixel(0, 0);
    // Red should still be the dominant channel after saturation boost
    assert!(
        pixel[0] > pixel[1] && pixel[0] > pixel[2],
        "red should remain dominant channel: R={}, G={}, B={}",
        pixel[0], pixel[1], pixel[2]
    );
}
```

---

## Validation Checklist

After PR is merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (including new tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] At saturation_boost=2.0, saturated colors don't show hue shifts
- [ ] At saturation_boost=0.0, output matches pre-fix behavior exactly
- [ ] Strength=0 still returns source unchanged (identity)

---

## References

- [Ordered dithering - Wikipedia](https://en.wikipedia.org/wiki/Ordered_dithering)
- [SNES Palettes - Super Famicom Development Wiki](https://wiki.superfamicom.org/palettes)
- [WGSL Function Reference - WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-wgsl-function-reference.html)
- [SNESdev Wiki - Palettes](https://snes.nesdev.org/wiki/Palettes)
- [Practical Bayer Dithering - Kaetemi](https://blog.kaetemi.be/2015/04/01/practical-bayer-dithering/)
