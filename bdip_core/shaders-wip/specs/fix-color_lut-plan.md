# Fix Color LUT Transform

## Problem Summary

The `color_lut` transform has fundamental issues that prevent it from being useful for color grading.

### Critical Issues

1. **Only identity LUT available**: The shader uses `identity_lut_64` which is a passthrough — it
   maps every input color to itself. At full intensity, this produces no visible color grading
   effect. The transform description claims to apply "a 3D color look-up table (LUT) color grade"
   but cannot actually color grade anything. There is a `polaroid_lut_64` asset registered but not
   usable by this shader.

2. **No LUT selection mechanism**: Users cannot choose which LUT to apply. The aux texture system
   hardcodes the LUT at compile time (`aux_textures: &[AuxTextureDef { name: "identity_lut_64",
   ...}]`). The `ParamKind` enum only supports `Sliders` and `Toggle` — there is no
   dropdown/choice type for runtime asset selection.

3. **Simplified gamma approximation**: The shader uses `pow(color, 1/2.2)` and `pow(color, 2.2)`
   for sRGB conversion instead of the proper piecewise sRGB transfer function. The true sRGB
   curve has a linear segment near black (below 0.0031308 linear / 0.04045 encoded) that
   transitions to a power curve with exponent 2.4 (not 2.2). While visually subtle, the
   mismatch causes color shifts in deep shadows when roundtripping, and produces visible seams
   when differently-converted images are composited.

### What Is Correct

- **Half-texel offset**: The texel coordinate calculation (`scale = (lut_size - 1) / lut_size`,
  `offset = 0.5 / lut_size`) correctly samples from texel centers, avoiding boundary artifacts.
- **Trilinear filtering**: Uses `AuxSamplerFilter::Linear` which is correct for smooth LUT
  interpolation.
- **Intensity blending**: The `mix(original, graded, intensity)` approach is standard for LUT
  strength controls.
- **Alpha preservation**: Correctly passes through the alpha channel unchanged.

### Current Parameters

| Parameter | Range   | Issue |
|-----------|---------|-------|
| Intensity | 0.0–1.0 | Default 0.0 makes effect invisible by default |

### Missing Parameters

- **LUT selection** (dropdown to choose from available LUTs)

---

## Implementation Plan

### PR 1: Add Proper sRGB Transfer Functions

**Goal**: Replace simplified gamma 2.2 approximation with accurate piecewise sRGB conversion.

**Scope**:
- Add sRGB helper functions to `color_lut.wgsl`
- Replace `pow(color, 1/2.2)` with proper `linear_to_srgb()` function
- Replace `pow(color, 2.2)` with proper `srgb_to_linear()` function

**Implementation Details**:

The proper sRGB OETF (linear → sRGB) is:
```wgsl
fn linear_to_srgb_channel(c: f32) -> f32 {
    if c <= 0.0031308 {
        return c * 12.92;
    } else {
        return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    }
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    return vec3(
        linear_to_srgb_channel(color.r),
        linear_to_srgb_channel(color.g),
        linear_to_srgb_channel(color.b)
    );
}
```

The proper sRGB EOTF (sRGB → linear) is:
```wgsl
fn srgb_to_linear_channel(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    } else {
        return pow((c + 0.055) / 1.055, 2.4);
    }
}

fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    return vec3(
        srgb_to_linear_channel(color.r),
        srgb_to_linear_channel(color.g),
        srgb_to_linear_channel(color.b)
    );
}
```

**Tests to Add**:

1. `test_color_lut_srgb_roundtrip_accuracy`: Verify that dark values (near black) roundtrip
   through sRGB conversion with minimal error. The simplified gamma 2.2 approach has significant
   error for values below 0.01 linear; the piecewise function should have <0.1% error.

2. `test_color_lut_srgb_linear_portion`: Verify that very dark values (linear < 0.003) produce
   output proportional to input (the linear segment of sRGB), not the curved response of pure
   gamma 2.2.

**Existing Tests to Update** (tolerance may tighten):
- `test_color_lut_identity_lut_is_passthrough` — may need tolerance adjustment

---

### PR 2: Add LUT Selection via Separate Shaders

**Goal**: Enable users to apply actual color grades by creating LUT-specific shader variants.

**Rationale**: The current architecture does not support runtime aux texture selection. Adding a
dropdown parameter type would require changes to `ParamKind`, the UI, and the pipeline's aux
texture binding logic. A simpler approach that works within the existing architecture is to
create separate shader registrations for each LUT (e.g., `color_lut_polaroid`, `color_lut_cinema`,
etc.), each hardcoded to its specific LUT asset.

**Scope**:
- Create `bdip_core/src/gpu/shaders/color_lut_polaroid/` as a copy of `color_lut/`
- Update the aux texture reference from `identity_lut_64` to `polaroid_lut_64`
- Update ID, display name, and description
- Register as a separate shader

**New Shader Metadata**:

```rust
impl TransformShader for ColorLutPolaroidParams {
    const ID: &'static str = "color_lut_polaroid";
    const DISPLAY_NAME: &'static str = "Color LUT: Polaroid";
    const DESCRIPTION: &'static str =
        "Applies a Polaroid-style color grade using a 3D look-up table.";
    // ... same PARAM and PASSES structure, but aux_textures references "polaroid_lut_64"
}
```

**Tests to Add**:

1. `test_color_lut_polaroid_registry_entry_exists`: Verify registration.

2. `test_color_lut_polaroid_visibly_differs_from_identity`: Apply Polaroid LUT at intensity=1.0 to
   a mid-gray image and verify output differs significantly from input (unlike identity LUT).

3. `test_color_lut_polaroid_warm_tint`: Polaroid LUTs typically add warmth. Verify that neutral
   gray input produces output with higher red channel than blue channel.

---

### PR 3: Change Default Intensity to 1.0

**Goal**: Make the effect visible by default.

**Rationale**: An intensity default of 0.0 means the effect is invisible when first applied,
which is confusing. Most filters default to "on" (e.g., brightness=0 means no change, but that's
a centered value; for a LUT, 0 means "off"). A default of 1.0 applies the full LUT, which is
the expected starting point for a color grade.

**Scope**:
- Update `SliderDef` default from `0.0` to `1.0` in both `color_lut` and `color_lut_polaroid`

**Tests to Update**:
- `test_color_lut_registry_metadata` — update expected default value

---

### PR 4 (Optional): Add ParamKind::Choice for Runtime LUT Selection

**Goal**: Enable a single `color_lut` shader that dynamically selects from available LUTs.

**Rationale**: This is a larger architectural change that would benefit multiple future shaders
(e.g., film emulation presets, gradient maps). However, it touches the parameter system, UI, and
aux texture binding logic. It should be considered tech debt to address later rather than
blocking the immediate fix.

**Scope** (if pursued):
- Add `ParamKind::Choice` variant to `mod.rs`
- Extend `AuxTextureDef` to support multiple named options
- Update UI sidebar to render dropdown for Choice parameters
- Update pipeline to bind the selected aux texture at runtime

**Priority**: Low — defer unless user feedback indicates strong demand for consolidated LUT
selection.

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_color_lut_srgb_roundtrip_accuracy`

```rust
/// Verify that sRGB conversion roundtrips accurately, especially for dark values
/// where simplified gamma 2.2 diverges from the true sRGB curve.
#[test]
fn test_color_lut_srgb_roundtrip_accuracy() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    
    // Test with a very dark gray (linear ~0.01, which is in the sRGB linear segment)
    // In 16-bit: 0.01 * 65535 ≈ 655
    let img = make_solid_image(4, 4, 655, 655, 655);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "color_lut",
            values: vec![1.0], // Full intensity with identity LUT
        }],
    );
    
    for pixel in out.pixels() {
        // With proper sRGB, error should be minimal (< 1% of input value)
        // With gamma 2.2, error in this range is ~5-10%
        assert!(
            (pixel[0] as i32 - 655).abs() <= 20,
            "dark value roundtrip error too large: expected ~655, got {}",
            pixel[0]
        );
    }
}
```

#### `test_color_lut_srgb_linear_portion`

```rust
/// Verify the linear portion of sRGB is implemented (values below 0.0031308 linear
/// should map linearly, not via the power curve).
#[test]
fn test_color_lut_srgb_linear_portion() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    
    // Two very dark values in the linear region
    let val1 = 100u16;  // ~0.0015 linear
    let val2 = 200u16;  // ~0.003 linear
    
    let img1 = make_solid_image(4, 4, val1, val1, val1);
    let img2 = make_solid_image(4, 4, val2, val2, val2);
    
    let out1 = roundtrip(&mut renderer, &engine, &img1, &[Transform {
        shader_id: "color_lut",
        values: vec![1.0],
    }]);
    let out2 = roundtrip(&mut renderer, &engine, &img2, &[Transform {
        shader_id: "color_lut",
        values: vec![1.0],
    }]);
    
    let ratio_in = val2 as f32 / val1 as f32;  // Should be ~2.0
    let ratio_out = out2.get_pixel(0, 0)[0] as f32 / out1.get_pixel(0, 0)[0] as f32;
    
    // In the linear region, doubling input should approximately double output
    // (within tolerance for quantization and f16 precision)
    assert!(
        (ratio_out - ratio_in).abs() < 0.2,
        "linear region ratio mismatch: input ratio {:.2}, output ratio {:.2}",
        ratio_in, ratio_out
    );
}
```

### PR 2 Tests (Detailed)

#### `test_color_lut_polaroid_visibly_differs_from_identity`

```rust
/// Verify the Polaroid LUT produces visible color grading (unlike identity).
#[test]
fn test_color_lut_polaroid_visibly_differs_from_identity() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    
    let img = make_solid_image(4, 4, 32767, 32767, 32767); // Mid-gray
    
    let out_identity = roundtrip(&mut renderer, &engine, &img, &[Transform {
        shader_id: "color_lut",
        values: vec![1.0],
    }]);
    let out_polaroid = roundtrip(&mut renderer, &engine, &img, &[Transform {
        shader_id: "color_lut_polaroid",
        values: vec![1.0],
    }]);
    
    // Polaroid should produce a different result than identity
    let identity_pixel = out_identity.get_pixel(0, 0);
    let polaroid_pixel = out_polaroid.get_pixel(0, 0);
    
    let diff = (identity_pixel[0] as i32 - polaroid_pixel[0] as i32).abs()
             + (identity_pixel[1] as i32 - polaroid_pixel[1] as i32).abs()
             + (identity_pixel[2] as i32 - polaroid_pixel[2] as i32).abs();
    
    assert!(
        diff > 1000,
        "Polaroid LUT should produce visibly different output than identity, diff={}",
        diff
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] `color_lut` with identity LUT at intensity=1.0 produces near-passthrough
- [ ] `color_lut_polaroid` at intensity=1.0 produces visible warm color grade
- [ ] Dark values (near black) roundtrip accurately through the LUT shader
- [ ] Intensity=0 returns source unchanged for both shaders
- [ ] Alpha channel is preserved

---

## References

- [NVIDIA GPU Gems 2 - Using Lookup Tables to Accelerate Color Transformations](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-24-using-lookup-tables-accelerate-color)
- [WebGPU Fundamentals - 3D Lookup Table (LUT)](https://webgpufundamentals.org/webgpu/lessons/webgpu-3dlut.html)
- [sRGB - Wikipedia](https://en.wikipedia.org/wiki/SRGB) — Authoritative reference for sRGB
  transfer function formulas
- [Colour Science - sRGB EOTF: Pure Gamma 2.2 or Piece-Wise Function?](https://www.colour-science.org/posts/srgb-eotf-pure-gamma-22-or-piece-wise-function/)
- [Understanding Half-Pixel and Half-Texel Offsets - GameDev.net](https://gamedev.net/blogs/entry/1848486-understanding-half-pixel-and-half-texel-offsets)
- [3D Game Shaders For Beginners - Lookup Table](https://lettier.github.io/3d-game-shaders-for-beginners/lookup-table.html)
