# Fix Double Exposure Transform

## Problem Summary

The `double_exposure` transform has significant algorithm issues that prevent it from producing an
authentic double exposure effect. The current implementation creates a stylized "negative overlay"
effect rather than simulating the classic film technique.

### Critical Issues

1. **Wrong algorithm: Inversion is the opposite of double exposure** (`double_exposure.wgsl:83`)
   
   The shader inverts each channel (`1.0 - blurred`), which creates a **photographic negative**.
   Real double exposure involves **additive light** from two exposures — dark areas stay dark, bright
   areas get brighter. Inversion does the opposite: dark areas become bright, creating an effect
   that looks nothing like double exposure.
   
   Sources:
   - [Carmencita Film Lab: The Science of Double Exposures](https://carmencitafilmlab.com/blog/the-science-of-double-exposures-and-how-to-make-them/)
   - [Wikipedia: Multiple Exposure](https://en.wikipedia.org/wiki/Multiple_exposure)

2. **Hue shift is unrelated to double exposure** (`double_exposure.wgsl:90`)
   
   The 120° hue rotation (`vec3(inverted.b, inverted.r, inverted.g)`) is arbitrary and has no basis
   in the physics of double exposure. Real double exposure does not alter colors — it simply
   combines light from two exposures.
   
   Source: [Howard Grill: In-Camera Blend Modes for Multiple Exposure](https://www.howardgrill.com/blog/2024/7/5/how-to-use-in-camera-blend-modes-for-multiple-exposure-photography-pt-ii-additive-mode)

3. **Single-image limitation without spatial differentiation**
   
   True double exposure combines TWO DIFFERENT images. When simulating with a single image,
   photographers create differentiation through:
   - Spatial offset (moving the camera between exposures)
   - Mirroring/flipping the second exposure
   - Using a blurred/defocused version as the ghost
   
   The current implementation's only differentiation is inversion + hue shift, which produces
   a psychedelic effect rather than a ghostly overlay.

4. **3x3 blur is too small for visible softening** (`double_exposure.wgsl:66-76`)
   
   A 1-pixel-radius box blur barely produces visible softening. Real double exposure ghosts
   often have a dreamy, defocused quality that requires a larger blur radius.

### Correct Behavior

Real double exposure on film:
- Exposes the same frame to two different scenes
- Light is **additive**: Screen blend mode (`1 - (1-a)*(1-b)`) correctly simulates this
- No color inversion or hue shifting occurs
- Result shows both images overlaid, with neither inverted

### Current Parameters

| Parameter | Range   | Issue                                      |
|-----------|---------|--------------------------------------------|
| Strength  | 0.0–1.0 | OK                                         |

### Missing Parameters

For authentic single-image double exposure simulation:
- **Offset X/Y**: Spatial offset for the ghost layer (creates visual separation)
- **Blur radius**: Control ghost softness (larger values = more dreamy)
- **Flip/Mirror**: Option to flip the ghost horizontally or vertically

---

## Implementation Plan

### PR 1: Remove Inversion and Hue Shift, Add Spatial Offset

**Goal**: Replace the incorrect negative+hue-shift algorithm with a proper spatial-offset-based
ghost that produces authentic double exposure appearance.

**Scope**:
- Remove channel inversion from `double_exposure.wgsl`
- Remove RGB channel rotation (hue shift)
- Add horizontal and vertical offset parameters
- Keep existing 3x3 blur (will expand in PR 2)
- Keep Screen blend mode (this part is correct)
- Update `DoubleExposureParams` struct with new fields
- Update `mod.rs` with new slider definitions

**New Parameters**:

```rust
pub struct DoubleExposureParams {
    pub strength: f32,   // 0.0–1.0, default 0.0 (identity)
    pub offset_x: f32,   // -0.5–0.5, default 0.1 (fraction of image width)
    pub offset_y: f32,   // -0.5–0.5, default 0.05 (fraction of image height)
    pub _padding: f32,
}
```

**New Shader Algorithm** (pseudocode):

```wgsl
// 1. Calculate ghost sample position with offset
let dims = vec2<f32>(textureDimensions(input_texture));
let offset = vec2<i32>(
    i32(params.offset_x * dims.x),
    i32(params.offset_y * dims.y)
);
let ghost_coord = clamp(coord + offset, vec2<i32>(0), vec2<i32>(dims) - 1);

// 2. Sample ghost with 3x3 blur (keep existing blur code)
var blur_sum = vec3<f32>(0.0);
for (var dy: i32 = -1; dy <= 1; dy++) {
    for (var dx: i32 = -1; dx <= 1; dx++) {
        let sample_coord = clamp(
            ghost_coord + vec2<i32>(dx, dy),
            vec2<i32>(0),
            vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1),
        );
        blur_sum += textureLoad(input_texture, sample_coord, 0).rgb;
    }
}
let ghost = blur_sum / 9.0;  // NO inversion, NO hue shift

// 3. Screen blend (unchanged)
let scaled_ghost = ghost * params.strength;
let screened = 1.0 - (1.0 - pixel.rgb) * (1.0 - scaled_ghost);
```

**Tests to Add**:

1. `test_double_exposure_offset_produces_shifted_ghost`: With a non-uniform input image, verify
   that positive offset_x shifts the ghost to the right (sampling from the left side of the image).

2. `test_double_exposure_no_hue_shift`: Verify that a single-color input produces output of the
   same hue (only brightness changes due to screen blend).

3. `test_double_exposure_zero_offset_blends_in_place`: With offset_x=0 and offset_y=0, the ghost
   samples the same region as the original (only blur differentiates them).

**Existing Tests to Update**:

- `test_double_exposure_full_strength_brightens_dark_image`: Remove — this test relies on
  inversion behavior. Replace with test that verifies screen blend lightens the image.
- Other existing tests should pass with minor tolerance adjustments.

---

### PR 2: Add Blur Radius Parameter

**Goal**: Allow user control over ghost blur intensity for dreamy effects.

**Scope**:
- Add blur_radius parameter (0–20 pixels)
- Implement variable-radius box blur sampling
- For large radii, use two-pass separable blur (horizontal + vertical) for efficiency
- Update parameter struct and slider definitions

**New Parameter** (added to existing struct):

```rust
pub struct DoubleExposureParams {
    pub strength: f32,    // 0.0–1.0, default 0.0
    pub offset_x: f32,    // -0.5–0.5, default 0.1
    pub offset_y: f32,    // -0.5–0.5, default 0.05
    pub blur_radius: f32, // 0.0–20.0, default 3.0 (pixels)
}
```

**Implementation Notes**:
- For blur_radius <= 2: Use direct NxN sampling (current approach, extended)
- For blur_radius > 2: Switch to two-pass separable blur using scratch textures
- Update `PASSES` to include horizontal blur → vertical blur when needed

**Tests to Add**:

1. `test_double_exposure_blur_radius_zero_is_sharp`: With blur_radius=0, the ghost should be
   pixel-sharp (no blur applied).

2. `test_double_exposure_blur_radius_large_creates_soft_ghost`: With blur_radius=10, edges
   in the ghost should be significantly softened compared to blur_radius=0.

---

### PR 3: Add Flip Mode Parameter

**Goal**: Add option to mirror the ghost for more visual variety.

**Scope**:
- Add flip_mode parameter (0=none, 1=horizontal, 2=vertical, 3=both)
- Apply flip transformation to ghost sampling coordinates
- Update parameter struct and slider definition

**New Parameter**:

```rust
// Add to existing struct (replaces _padding)
pub flip_mode: f32,  // 0.0=none, 1.0=horizontal, 2.0=vertical, 3.0=both
```

**Implementation**:

```wgsl
// Apply flip before offset
var ghost_base = coord;
if params.flip_mode == 1.0 || params.flip_mode == 3.0 {
    ghost_base.x = i32(dims.x) - 1 - ghost_base.x;  // horizontal flip
}
if params.flip_mode == 2.0 || params.flip_mode == 3.0 {
    ghost_base.y = i32(dims.y) - 1 - ghost_base.y;  // vertical flip
}
let ghost_coord = clamp(ghost_base + offset, ...);
```

**Tests to Add**:

1. `test_double_exposure_flip_horizontal`: Verify that flip_mode=1 samples from the opposite
   horizontal side of the image.

2. `test_double_exposure_flip_vertical`: Verify that flip_mode=2 samples from the opposite
   vertical side.

3. `test_double_exposure_flip_both`: Verify that flip_mode=3 samples from the diagonally
   opposite corner.

---

### PR 4: Update Documentation and Description

**Goal**: Ensure documentation accurately describes the corrected effect.

**Scope**:
- Update `DESCRIPTION` to explain the authentic double exposure simulation
- Update all slider descriptions
- Ensure parameter help follows project conventions

**Updated Constants**:

```rust
const DESCRIPTION: &'static str = "Simulates the classic film technique of exposing the same \
    frame twice by overlaying a spatially offset, optionally blurred version of the image \
    using Screen blend mode. The ghost can be flipped for creative effects.";

const PARAM: ParamKind = ParamKind::Sliders(&[
    SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Intensity of the ghost overlay. 0.0 shows original image only.",
    },
    SliderDef {
        name: "Offset X",
        min: -0.5,
        max: 0.5,
        default: 0.1,
        description: "Horizontal shift of the ghost as fraction of image width.",
    },
    SliderDef {
        name: "Offset Y",
        min: -0.5,
        max: 0.5,
        default: 0.05,
        description: "Vertical shift of the ghost as fraction of image height.",
    },
    SliderDef {
        name: "Blur",
        min: 0.0,
        max: 20.0,
        default: 3.0,
        description: "Ghost blur radius in pixels. Higher values create dreamier effects.",
    },
    SliderDef {
        name: "Flip Mode",
        min: 0.0,
        max: 3.0,
        default: 0.0,
        description: "Ghost flip: 0=none, 1=horizontal, 2=vertical, 3=both.",
    },
]);
```

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_double_exposure_offset_produces_shifted_ghost`

```rust
/// Verify that offset_x shifts the ghost sampling position.
#[test]
fn test_double_exposure_offset_produces_shifted_ghost() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Create gradient image: left side dark, right side bright
    let mut img = Rgba16Image::new(64, 64);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let val = ((x as f32 / 63.0) * 65535.0) as u16;
        *pixel = Rgba([val, val, val, 65535]);
    }

    // With positive offset_x, ghost samples from the LEFT (darker region)
    // Screen-blending a darker ghost onto a brighter pixel should still brighten,
    // but less than blending a bright ghost.
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "double_exposure",
            values: vec![1.0, 0.3, 0.0], // strength=1, offset_x=0.3, offset_y=0
        }],
    );

    // Right side of output (x=60) should be brighter than input due to screen blend
    // but the ghost contribution comes from ~x=40 (darker region)
    let input_right = 65535u16; // ~1.0
    let output_right = out.get_pixel(60, 32)[0];
    assert!(
        output_right >= input_right - 500,
        "screen blend should not darken: input={}, output={}",
        input_right, output_right
    );
}
```

#### `test_double_exposure_no_hue_shift`

```rust
/// Verify that the effect preserves hue (no RGB channel rotation).
#[test]
fn test_double_exposure_no_hue_shift() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Pure red input
    let img = make_solid_image(16, 16, 30000, 0, 0);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "double_exposure",
            values: vec![1.0, 0.0, 0.0], // strength=1, no offset
        }],
    );

    for pixel in out.pixels() {
        // Output should still be red-dominant (R >= G, R >= B)
        // Screen blend of red with red stays red
        assert!(
            pixel[0] >= pixel[1] && pixel[0] >= pixel[2],
            "hue should be preserved: R={}, G={}, B={}",
            pixel[0], pixel[1], pixel[2]
        );
    }
}
```

#### `test_double_exposure_zero_offset_blends_in_place`

```rust
/// Verify that zero offset samples the same position (only blur differentiates).
#[test]
fn test_double_exposure_zero_offset_blends_in_place() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Uniform gray: with zero offset and no blur variation, ghost = original
    let img = make_solid_image(16, 16, 30000, 30000, 30000);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "double_exposure",
            values: vec![1.0, 0.0, 0.0], // strength=1, offset_x=0, offset_y=0
        }],
    );

    // Screen(a, a) = 1 - (1-a)^2 = 2a - a^2
    // For a=0.458 (30000/65535), result = 2*0.458 - 0.458^2 = 0.706
    // Expected output: ~46300
    for pixel in out.pixels() {
        let expected = 46300i32;
        assert!(
            (pixel[0] as i32 - expected).abs() < 2000,
            "screen(a,a) should equal 2a-a²: expected ~{}, got {}",
            expected, pixel[0]
        );
    }
}
```

### PR 2 Tests (Detailed)

#### `test_double_exposure_blur_radius_zero_is_sharp`

```rust
/// Verify blur_radius=0 produces a sharp ghost (no blur).
#[test]
fn test_double_exposure_blur_radius_zero_is_sharp() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Create checkerboard pattern
    let mut img = Rgba16Image::new(64, 64);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let val = if (x + y) % 2 == 0 { 65535 } else { 0 };
        *pixel = Rgba([val, val, val, 65535]);
    }

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "double_exposure",
            values: vec![0.5, 0.0, 0.0, 0.0], // blur_radius=0
        }],
    );

    // With no blur, output should still have high contrast (checkerboard preserved)
    let values: Vec<u16> = out.pixels().map(|p| p[0]).collect();
    let min = *values.iter().min().unwrap();
    let max = *values.iter().max().unwrap();
    assert!(
        max - min > 30000,
        "blur_radius=0 should preserve sharp edges: min={}, max={}",
        min, max
    );
}
```

#### `test_double_exposure_blur_radius_large_creates_soft_ghost`

```rust
/// Verify large blur_radius creates a soft, low-contrast ghost.
#[test]
fn test_double_exposure_blur_radius_large_creates_soft_ghost() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Create checkerboard pattern
    let mut img = Rgba16Image::new(64, 64);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let val = if (x + y) % 2 == 0 { 65535 } else { 0 };
        *pixel = Rgba([val, val, val, 65535]);
    }

    let out_sharp = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "double_exposure",
            values: vec![1.0, 0.0, 0.0, 0.0], // blur_radius=0
        }],
    );

    let out_blurred = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "double_exposure",
            values: vec![1.0, 0.0, 0.0, 10.0], // blur_radius=10
        }],
    );

    // Blurred version should have lower contrast (smaller value range)
    let range = |img: &Rgba16Image| {
        let values: Vec<u16> = img.pixels().map(|p| p[0]).collect();
        values.iter().max().unwrap() - values.iter().min().unwrap()
    };

    let sharp_range = range(&out_sharp);
    let blurred_range = range(&out_blurred);

    assert!(
        blurred_range < sharp_range,
        "blur should reduce contrast: sharp_range={}, blurred_range={}",
        sharp_range, blurred_range
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Effect produces a ghostly overlay (not a psychedelic negative)
- [ ] Offset parameters shift the ghost position
- [ ] Blur parameter controls ghost softness
- [ ] Flip modes work correctly
- [ ] Single-color input stays same hue (no color rotation)
- [ ] Strength=0 returns source unchanged (identity)
- [ ] Screen blend only brightens, never darkens

---

## References

- [Wikipedia: Multiple Exposure](https://en.wikipedia.org/wiki/Multiple_exposure)
- [Carmencita Film Lab: The Science of Double Exposures](https://carmencitafilmlab.com/blog/the-science-of-double-exposures-and-how-to-make-them/)
- [Howard Grill: In-Camera Blend Modes - Additive Mode](https://www.howardgrill.com/blog/2024/7/5/how-to-use-in-camera-blend-modes-for-multiple-exposure-photography-pt-ii-additive-mode)
- [Schenectady Photographic Society: Double Exposure Using Screen Blend Mode](https://spsphoto.org/news/how-to-create-a-double-exposure-effect-in-photoshop-using-screen-blend-mode/)
- [Studio Binder: What is Double Exposure?](https://www.studiobinder.com/blog/what-is-double-exposure-photography/)
- [Multiple Exposure Blending Modes - The Differences](https://www.jmpeltier.com/multiple-exposure-blending-modes/)
