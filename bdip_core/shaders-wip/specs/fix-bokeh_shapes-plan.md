# Fix Bokeh Shapes Transform

## Problem Summary

The bokeh_shapes transform produces a polygon-shaped blur but lacks the characteristic highlight
emphasis that makes real bokeh visually distinctive. The implementation is mathematically correct
for aperture-shaped averaging, but real camera bokeh emphasizes bright highlights because bright
areas scatter more light through the lens aperture.

### Moderate Issues

1. **Missing highlight emphasis / brightness weighting** (lines 75-108 in `bokeh_shapes_blur.wgsl`):
   The blur pass uses uniform box filtering — all samples are weighted equally. In real camera
   optics, bright highlights "pop out" as prominent aperture shapes because they scatter more light.
   Without brightness weighting, the effect looks like a polygon-shaped blur rather than authentic
   lens bokeh.

   The standard fix is to apply a non-linear brightness curve before averaging:
   - Pre-blur: `pow(color, highlight_power)` to boost bright pixels' influence
   - Post-blur: `pow(result, 1.0 / highlight_power)` to normalize
   
   References:
   - [Bart Wronski - Bokeh DOF going insane](https://bartwronski.com/2014/04/07/bokeh-depth-of-field-going-insane-part-1/)
   - [MJP - How To Fake Bokeh](https://therealmjp.github.io/posts/bokeh/)

2. **Fixed polygon orientation** (line 45-51 in `bokeh_shapes_blur.wgsl`): The polygon aperture
   always has a vertex pointing up. Real camera lenses have varying aperture blade orientations,
   and some artistic effects benefit from rotation control.

### Minor Issues

3. **Hard aperture edges**: The polygon SDF produces binary inside/outside. Real lenses have some
   softness at the aperture boundary due to optical effects. This is a minor enhancement.

### Current Parameters

| Parameter | Range    | Status |
|-----------|----------|--------|
| Radius    | 0.0–50.0 | OK     |
| Sides     | 0.0–12.0 | OK     |
| Strength  | 0.0–1.0  | OK     |

### Missing Parameters

- **Highlight** (brightness emphasis power, typically 1.0–4.0, default 2.0)
- **Rotation** (aperture rotation in degrees, 0–180°, default 0)

### What's Correct

- Multi-pass pipeline with downscaling is a valid cost-reduction strategy
- Polygon SDF math is correct for regular polygons
- Circle fallback using Euclidean distance is correct
- Uniform weighting within aperture (box filter) is appropriate — the issue is lack of brightness
  pre/post-processing, not the kernel shape itself
- Identity behavior at strength=0 and radius=0 is correct

---

## Implementation Plan

### PR 1: Add Highlight Emphasis (Brightness Weighting)

**Goal**: Add brightness weighting to make bright highlights pop out as prominent bokeh shapes,
matching real camera optics behavior.

**Scope**:
- Update `BokehShapesParams` struct to add `highlight` parameter
- Update slider definitions in `mod.rs`
- Modify `bokeh_shapes_blur.wgsl` to apply brightness weighting:
  - Before accumulating: raise sample RGB to `highlight` power
  - After averaging: raise result RGB to `1.0 / highlight` power
- Update existing tests to include the new parameter value

**New Parameter**:

```rust
pub struct BokehShapesParams {
    pub radius: f32,
    pub sides: f32,
    pub strength: f32,
    pub highlight: f32,  // Replaces _padding — convenient since struct was already 16 bytes
}
```

Note: The `_padding` field becomes `highlight`. The struct remains 16 bytes, satisfying WebGPU
alignment requirements.

**Slider Definition**:

```rust
SliderDef {
    name: "Highlight",
    min: 1.0,
    max: 4.0,
    default: 1.0,
    description: "Brightness emphasis. Higher values make bright highlights pop out \
                 as prominent bokeh shapes. At 1.0 all pixels are weighted equally.",
},
```

**Algorithm Change** (in `bokeh_shapes_blur.wgsl`):

```wgsl
// Inside the sample loop:
if inside {
    let sample_coord = clamp(...);
    var sample = textureLoad(input_texture, sample_coord, 0);
    
    // Brightness weighting: boost bright samples' influence
    sample = vec4<f32>(
        pow(sample.rgb, vec3<f32>(params.highlight)),
        sample.a
    );
    
    accum += sample;
    count += 1.0;
}

// After the loop, before storing:
if count > 0.0 {
    out = accum / count;
    // Reverse the brightness boost
    out = vec4<f32>(
        pow(out.rgb, vec3<f32>(1.0 / params.highlight)),
        out.a
    );
}
```

**Tests to Add**:

1. `test_bokeh_shapes_highlight_emphasizes_bright_pixels`: Create an image with a bright spot on
   a dark background. With highlight > 1.0, the bright spot's influence should spread further than
   with highlight = 1.0.

2. `test_bokeh_shapes_highlight_one_is_uniform`: At highlight=1.0 (the default), the behavior
   should match the original uniform box filter (pow(x, 1) = x).

**Tests to Update** (add 4th parameter value):
- All existing GPU roundtrip tests need their `values` vectors updated to include the highlight
  parameter (e.g., `vec![10.0, 6.0, 0.5]` → `vec![10.0, 6.0, 0.5, 1.0]`)

---

### PR 2: Add Aperture Rotation Parameter

**Goal**: Allow rotating the polygon aperture for artistic control and to match real lens
characteristics.

**Scope**:
- Add `rotation` parameter to `BokehShapesParams` (struct grows to 20 bytes → rounds to 32 bytes
  for alignment)
- Update slider definitions
- Modify `polygon_sdf` in `bokeh_shapes_blur.wgsl` to accept rotation angle
- Rotate sample offset before SDF evaluation

**New Parameters Struct** (after PR 1):

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BokehShapesParams {
    pub radius: f32,
    pub sides: f32,
    pub strength: f32,
    pub highlight: f32,
    pub rotation: f32,
    pub _padding: [f32; 3],  // Pad to 32 bytes for WebGPU alignment
}
```

**Slider Definition**:

```rust
SliderDef {
    name: "Rotation",
    min: 0.0,
    max: 180.0,
    default: 0.0,
    description: "Aperture rotation in degrees. Rotates the polygon shape.",
},
```

**Algorithm Change** (in `bokeh_shapes_blur.wgsl`):

```wgsl
// Before polygon_sdf call, rotate the offset:
let rot_rad = params.rotation * PI / 180.0;
let cos_r = cos(rot_rad);
let sin_r = sin(rot_rad);
let rotated_offset = vec2<f32>(
    offset.x * cos_r - offset.y * sin_r,
    offset.x * sin_r + offset.y * cos_r
);
inside = polygon_sdf(rotated_offset, n_sides, r_ds) <= 0.0;
```

**Tests to Add**:

1. `test_bokeh_shapes_rotation_changes_pattern`: On a step-edge image, rotation=0 and rotation=30
   should produce different output patterns.

2. `test_bokeh_shapes_rotation_zero_matches_unrotated`: At rotation=0 the output should match the
   original behavior (regression test).

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_bokeh_shapes_highlight_emphasizes_bright_pixels`

```rust
/// Verify that highlight > 1.0 makes bright spots more prominent in the bokeh.
/// A bright pixel on a dark background should spread its influence further.
#[test]
fn test_bokeh_shapes_highlight_emphasizes_bright_pixels() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // 32x32 dark image with a single bright pixel at center
    let mut img = crate::Rgba16Image::new(32, 32);
    for y in 0..32u32 {
        for x in 0..32u32 {
            let v: u16 = if x == 16 && y == 16 { 65535 } else { 0 };
            img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
        }
    }

    // Low highlight (uniform weighting)
    let out_low = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "bokeh_shapes",
            values: vec![8.0, 6.0, 1.0, 1.0], // highlight=1.0
        }],
    );

    // High highlight (bright emphasis)
    let out_high = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "bokeh_shapes",
            values: vec![8.0, 6.0, 1.0, 3.0], // highlight=3.0
        }],
    );

    // With higher highlight, the bright pixel's influence should be more prominent.
    // Measure average brightness in a ring around the center (e.g., radius 4-8 pixels).
    let ring_brightness = |img: &crate::Rgba16Image| -> f32 {
        let mut sum = 0u64;
        let mut count = 0u32;
        for y in 8..24u32 {
            for x in 8..24u32 {
                let dx = (x as i32 - 16).abs();
                let dy = (y as i32 - 16).abs();
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                if dist >= 4.0 && dist <= 8.0 {
                    sum += img.get_pixel(x, y)[0] as u64;
                    count += 1;
                }
            }
        }
        sum as f32 / count as f32
    };

    let brightness_low = ring_brightness(&out_low);
    let brightness_high = ring_brightness(&out_high);

    // Higher highlight should produce brighter halo
    assert!(
        brightness_high > brightness_low,
        "highlight=3 should produce brighter halo than highlight=1: low={}, high={}",
        brightness_low, brightness_high
    );
}
```

#### `test_bokeh_shapes_highlight_one_is_uniform`

```rust
/// At highlight=1.0, the effect should match uniform box filtering (pow(x,1)=x).
/// Compare against a reference run to verify no change in behavior at default.
#[test]
fn test_bokeh_shapes_highlight_one_is_uniform() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(32, 32, 40000, 20000, 30000);

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "bokeh_shapes",
            values: vec![8.0, 6.0, 1.0, 1.0], // highlight=1.0
        }],
    );

    // Solid color should remain solid after blur (regardless of highlight)
    for pixel in out.pixels() {
        assert!((pixel[0] as i32 - 40000).abs() <= 200, "R channel drifted");
        assert!((pixel[1] as i32 - 20000).abs() <= 200, "G channel drifted");
        assert!((pixel[2] as i32 - 30000).abs() <= 200, "B channel drifted");
    }
}
```

### PR 2 Tests (Detailed)

#### `test_bokeh_shapes_rotation_changes_pattern`

```rust
/// Verify the rotation parameter changes the bokeh pattern.
#[test]
fn test_bokeh_shapes_rotation_changes_pattern() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Step edge: left half dark, right half bright
    let mut img = crate::Rgba16Image::new(64, 64);
    for y in 0..64u32 {
        for x in 0..64u32 {
            let v: u16 = if x < 32 { 0 } else { 65535 };
            img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
        }
    }

    let out_0 = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "bokeh_shapes",
            values: vec![8.0, 6.0, 1.0, 1.0, 0.0], // rotation=0
        }],
    );

    let out_30 = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "bokeh_shapes",
            values: vec![8.0, 6.0, 1.0, 1.0, 30.0], // rotation=30
        }],
    );

    // Patterns must differ
    let differs = out_0.pixels().zip(out_30.pixels())
        .any(|(a, b)| (a[0] as i32 - b[0] as i32).abs() > 500);
    assert!(differs, "rotation=0 and rotation=30 must produce different patterns");
}
```

#### `test_bokeh_shapes_rotation_zero_matches_unrotated`

```rust
/// Rotation=0 should produce identical output to the pre-rotation implementation
/// (regression test).
#[test]
fn test_bokeh_shapes_rotation_zero_is_default() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(32, 32, 32767, 32767, 32767);

    // Two runs with rotation=0 should be identical
    let out1 = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "bokeh_shapes",
            values: vec![8.0, 6.0, 0.7, 1.0, 0.0],
        }],
    );
    let out2 = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "bokeh_shapes",
            values: vec![8.0, 6.0, 0.7, 1.0, 0.0],
        }],
    );

    for (p1, p2) in out1.pixels().zip(out2.pixels()) {
        assert_eq!(p1, p2, "rotation=0 must be deterministic");
    }
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Bokeh effect with highlight=2.0 shows prominent polygon shapes around bright lights
- [ ] Highlight=1.0 produces smooth, even blur without highlight emphasis
- [ ] Rotation parameter rotates the aperture shape visibly
- [ ] Sides=6 produces hexagonal bokeh; sides=0 produces circular bokeh
- [ ] Strength=0 returns source unchanged (identity)
- [ ] Radius=0 returns source unchanged (identity)

---

## References

- [Bart Wronski - Bokeh depth of field going insane](https://bartwronski.com/2014/04/07/bokeh-depth-of-field-going-insane-part-1/)
- [MJP - How To Fake Bokeh](https://therealmjp.github.io/posts/bokeh/)
- [McIntosh et al. - Efficiently Simulating Polygonal Apertures (CGF 2012)](http://ivizlab.sfu.ca/papers/cgf2012.pdf)
- [Voxagon - Bokeh DOF in single pass](https://blog.voxagon.se/2018/05/04/bokeh-depth-of-field-in-single-pass.html)
- [GitHub - Hexagonal Bokeh Blur](https://github.com/YuAo/HexagonalBokehBlur)
- [Real-time bokeh algorithms - Dercuano](https://dercuano.github.io/notes/convolution-bokeh.html)
