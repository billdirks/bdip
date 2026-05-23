# Fix Parchment Transform

## Problem Summary

The parchment transform's core algorithm (paper grain overlay via multiply blend) is correct, but it
doesn't fully deliver on the "aged parchment" promise in its description. Real aged parchment has
two key visual characteristics: texture grain AND warm yellowing. The current implementation only
provides the former.

### Moderate Issues

1. **Missing warmth/sepia tint**: The description claims to "simulate aged parchment" but only
   applies grain texture without the characteristic yellowing of aged paper. Real parchment yellows
   over time due to lignin oxidation, and most parchment effects in image processing include a warm
   color shift.

   - Line 18-19 in `mod.rs`: Description says "simulate aged parchment" but implementation at
     lines 28-29 in `parchment.wgsl` only multiplies by paper texture without color tinting.

### Minor Observations

1. **Default intensity is 0.0**: The effect is invisible by default. While this is a safe identity
   state, most creative effects have a visible default (e.g., 0.5) so users see the effect
   immediately. This is a UX choice, not a bug.

### What's Correct

- Multiply blend is appropriate for paper grain simulation (industry standard per Photoshop
  workflows)
- UV scaling math is correct (higher scale = texture zooms in)
- Alpha channel preserved correctly
- Tests are comprehensive and accurate

---

## Implementation Plan

### PR 1: Add Warmth Parameter for Aged Parchment Effect

**Goal**: Add a "Warmth" parameter that applies sepia-like color tinting to complete the aged
parchment simulation.

**Scope**:
- Modify `bdip_core/src/gpu/shaders/parchment/mod.rs`:
  - Add `warmth: f32` field to `ParchmentParams` struct
  - Add slider definition for Warmth (0.0–1.0, default 0.3)
  - Adjust padding (remove one padding field)
- Modify `bdip_core/src/gpu/shaders/parchment/parchment.wgsl`:
  - Add sepia tinting logic before the multiply blend
  - Use standard sepia matrix coefficients (Microsoft formula)
  - Interpolate between original color and sepia-tinted color based on warmth

**Updated Params Struct**:

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParchmentParams {
    pub intensity: f32,
    pub scale: f32,
    pub warmth: f32,
    pub _padding: f32,
}
```

**New Slider Definition**:

```rust
SliderDef {
    name: "Warmth",
    min: 0.0,
    max: 1.0,
    default: 0.3,
    description: "Sepia tint strength simulating paper yellowing from age; \
                  0 is neutral, 1 is full sepia.",
},
```

**Shader Algorithm** (to insert before multiply blend):

```wgsl
// Sepia tint matrix (Microsoft standard coefficients)
fn apply_sepia(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.393, 0.769, 0.189)),
        dot(c, vec3<f32>(0.349, 0.686, 0.168)),
        dot(c, vec3<f32>(0.272, 0.534, 0.131))
    );
}

// In main():
let sepia = apply_sepia(color.rgb);
let warmed = mix(color.rgb, sepia, params.warmth);
let parchment = warmed * paper;
let out = mix(color.rgb, parchment, params.intensity);
```

**Tests to Add**:

1. `test_parchment_warmth_zero_preserves_hue`: Verify that warmth=0 doesn't shift color hue.
2. `test_parchment_warmth_shifts_blue_toward_yellow`: Blue input with warmth=1.0 should shift
   toward warm/brown tones.
3. `test_parchment_warmth_and_intensity_independent`: Verify both parameters work independently.

**Tests to Update**:

- `test_parchment_registry_metadata`: Update expected slider count and definitions
- All roundtrip tests: Update `values` vector to include warmth parameter (third value)

---

## Test Specifications

### `test_parchment_warmth_zero_preserves_hue`

```rust
/// Verify warmth=0 applies grain without color shift.
#[test]
fn test_parchment_warmth_zero_preserves_hue() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    
    // Pure blue input
    let img = make_solid_image(8, 8, 0, 0, 32767);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "parchment",
            values: vec![1.0, 1.0, 0.0], // intensity=1, scale=1, warmth=0
        }],
    );
    
    // Blue channel should still dominate (allowing for grain darkening)
    for pixel in out.pixels() {
        assert!(
            pixel[2] >= pixel[0] && pixel[2] >= pixel[1],
            "blue should remain dominant with warmth=0, got RGB {:?}",
            [pixel[0], pixel[1], pixel[2]]
        );
    }
}
```

### `test_parchment_warmth_shifts_blue_toward_yellow`

```rust
/// Verify warmth=1 applies sepia tint (blue shifts toward warm tones).
#[test]
fn test_parchment_warmth_shifts_blue_toward_yellow() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    
    // Pure blue input
    let img = make_solid_image(8, 8, 0, 0, 32767);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "parchment",
            values: vec![1.0, 1.0, 1.0], // intensity=1, scale=1, warmth=1
        }],
    );
    
    // With full sepia, red channel should increase relative to blue
    // Sepia of pure blue (0,0,1): R=0.189, G=0.168, B=0.131
    // So R > B after sepia transform
    let pixel = out.get_pixel(4, 4);
    assert!(
        pixel[0] > pixel[2],
        "sepia tint should shift blue toward warm: R={} should exceed B={}",
        pixel[0], pixel[2]
    );
}
```

### `test_parchment_warmth_and_intensity_independent`

```rust
/// Verify warmth and intensity operate independently.
#[test]
fn test_parchment_warmth_and_intensity_independent() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    
    let img = make_solid_image(8, 8, 32767, 32767, 32767);
    
    // Warmth only (intensity=0 should return original regardless of warmth)
    let out_warmth_only = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "parchment",
            values: vec![0.0, 1.0, 1.0], // intensity=0, warmth=1
        }],
    );
    
    // With intensity=0, output should match input (within tolerance)
    for pixel in out_warmth_only.pixels() {
        assert!(
            (pixel[0] as i32 - 32767).abs() <= 128,
            "intensity=0 should return original regardless of warmth"
        );
    }
}
```

---

## Validation Checklist

After PR is merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (including new warmth tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Warmth=0 produces pure grain effect (no color shift)
- [ ] Warmth=1 produces visible sepia/warm tint
- [ ] Intensity=0 returns source unchanged regardless of warmth
- [ ] Default values (intensity=0, scale=1, warmth=0.3) produce visible aged paper effect when
      intensity is raised

---

## References

- [Sepia Tone Formula - Microsoft Learn](https://learn.microsoft.com/en-us/archive/msdn-magazine/2005/january/net-matters-sepia-tone-stringlogicalcomparer-and-more)
- [Texture Blend Modes - Imagitool](https://imagitool.com/blog/texture-blend-modes-overlay-multiply-soft-light)
- [Sepia Shader - Shadertoy](https://www.shadertoy.com/view/3slfDl)
- [Paper Texture Blending - Adobe Community](https://community.adobe.com/t5/photoshop-ecosystem-discussions/best-practices-for-adding-paper-textures-to-images/td-p/13453232)
