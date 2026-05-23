# Fix Frost Ice Transform

## Problem Summary

The current frost_ice transform produces reasonable frost coloring and masking but is missing a key
visual characteristic: **blur**. Real frost creates a diffuse, scattered-light appearance that
obscures detail rather than just displacing pixels. Without blur, the effect looks more like noisy
distortion than actual frost on glass.

### Critical Issues

1. **Missing blur** (lines 141, frost_ice.wgsl): The shader samples a single displaced pixel
   (`textureLoad(src_texture, clamped_coord, 0)`) rather than averaging multiple samples. This
   produces sharp displacement artifacts instead of the soft, diffuse appearance of real frosted
   glass. Research indicates Gaussian blur is a "standard ingredient" for frost effects, and
   "insufficient blur makes frost look like simple noise."

2. **Multi-pass architecture required**: Adding proper blur requires either:
   - A separate blur pass before the compositing pass (cleaner, reuses existing blur infrastructure)
   - Multi-tap sampling within the shader (less efficient but single-pass)

   The current single-pass design cannot efficiently implement quality blur.

### Moderate Issues

1. **Fixed noise scale** (lines 69-80, frost_ice.wgsl): The noise uses fixed scales (6.0, 12.0) that
   don't adapt to image dimensions. On a 4K image, the frost pattern will appear much finer than on
   a 256px thumbnail. Scale should be relative to image size for consistent visual appearance.

### Minor Issues

1. **No blur parameter**: Users cannot control how diffuse the frost appears. Adding a blur radius
   parameter would provide artistic control.

2. **Noise pattern style**: The current value noise produces organic blob-like variations. Real
   frost has dendritic (branching, tree-like) patterns. Cellular or Worley noise could better
   approximate crystal formations. However, the current approach is visually acceptable for a
   stylized effect.

### Current Parameters

| Parameter   | Range   | Assessment |
|-------------|---------|------------|
| Coverage    | 0.0–1.0 | Good - controls frost extent from edges |
| Distortion  | 0.0–1.0 | Good range, but effect is limited without blur |
| Strength    | 0.0–1.0 | Good - allows identity at 0.0 |

### Missing Parameters

- **Blur** (frost diffusion radius, 0.0–1.0) - critical for realistic appearance

---

## Implementation Plan

### PR 1: Add Blur Pass for Frost Diffusion

**Goal**: Convert frost_ice to a two-pass shader that blurs the source in frosted regions before
compositing, creating the characteristic diffuse appearance of real frost.

**Scope**:
- `bdip_core/src/gpu/shaders/frost_ice/mod.rs`:
  - Add `blur` parameter (0.0–1.0, default 0.3)
  - Update `FrostIceParams` struct with new field
  - Convert to two-pass pipeline: blur pass + composite pass
  - Update slider definitions
- Create `bdip_core/src/gpu/shaders/frost_ice/frost_ice_blur.wgsl`:
  - Gaussian blur weighted by frost mask
  - Variable radius based on blur parameter
- Update `bdip_core/src/gpu/shaders/frost_ice/frost_ice.wgsl`:
  - Read from blurred scratch texture instead of source
  - Remove or reduce UV distortion now that blur provides diffusion

**New Parameter**:

```rust
pub struct FrostIceParams {
    pub coverage: f32,    // 0.0–1.0, default 0.0 (frost extent from edges)
    pub distortion: f32,  // 0.0–1.0, default 0.0 (UV warp amplitude)
    pub blur: f32,        // 0.0–1.0, default 0.3 (frost diffusion radius)
    pub strength: f32,    // 0.0–1.0, default 0.0 (overall effect opacity)
}
```

**Pass Configuration**:

```rust
const PASSES: &'static [PassDef] = &[
    PassDef {
        label: "frost_ice_blur",
        wgsl_source: include_str!("frost_ice_blur.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Scratch("blurred"),
        output_scale: PassScale::Full,
        aux_textures: &[],
    },
    PassDef {
        label: "frost_ice",
        wgsl_source: include_str!("frost_ice.wgsl"),
        inputs: &[PassInput::Source, PassInput::Scratch("blurred")],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    },
];
```

**Blur Pass Algorithm** (frost_ice_blur.wgsl pseudocode):

```wgsl
// Compute frost mask (same as main shader)
let edge_dist = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
let frost_reach = params.coverage * 0.5;
let frost_mask = 1.0 - smoothstep(0.0, frost_reach + 0.001, edge_dist);

// Blur radius scales with both blur param and frost mask
// Max radius ~15 pixels at blur=1.0
let radius = params.blur * frost_mask * 15.0;

// 9-tap Gaussian blur (for efficiency, use separable blur in future optimization)
var total = vec3<f32>(0.0);
var weight_sum = 0.0;
for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
        let offset = vec2<f32>(f32(dx), f32(dy)) * radius / 2.0;
        let sample_coord = coord + vec2<i32>(offset);
        let w = gaussian_weight(dx, dy);
        total += textureLoad(src, clamp_coord(sample_coord), 0).rgb * w;
        weight_sum += w;
    }
}
let blurred = total / weight_sum;

// Mix blur based on frost mask - clear areas remain sharp
let result = mix(source.rgb, blurred, frost_mask);
```

**Main Shader Updates** (frost_ice.wgsl):

```wgsl
// Read pre-blurred texture for frosted regions
let blurred_pixel = textureLoad(blurred_texture, coord, 0);

// Use blurred source instead of distorted single-sample
// Distortion can still offset which blurred pixel we read for extra effect
let sample_coord = coord + distortion_offset;
let source_pixel = textureLoad(blurred_texture, sample_coord, 0);

// Rest of compositing unchanged
```

**Tests to Add**:

1. `test_frost_ice_blur_creates_diffuse_appearance`: With blur=1.0 on a checkerboard pattern, verify
   that pixel variance in frost region is reduced compared to blur=0.0.

2. `test_frost_ice_blur_zero_is_sharp`: With blur=0.0, output should match current behavior (no
   blur applied).

3. `test_frost_ice_blur_respects_frost_mask`: Areas outside frost region should remain sharp even
   with blur=1.0.

**Existing Tests to Update**:
- `test_frost_ice_make_uniform_known_value`: Add blur parameter to test data
- All existing tests: Add blur value (use 0.0 to preserve current behavior, or update expected
  results for new default)

---

### PR 2: Scale-Independent Noise

**Goal**: Make frost pattern visually consistent across different image sizes.

**Scope**:
- `bdip_core/src/gpu/shaders/frost_ice/frost_ice.wgsl`:
  - Scale noise frequencies relative to image dimensions
  - Target approximately 20-40 frost "cells" across the shortest dimension

**Algorithm Update**:

```wgsl
// Current: fixed scale creates size-dependent appearance
// let n0 = value_noise(uv * 6.0);

// New: scale relative to minimum dimension
let min_dim = min(f32(dims.x), f32(dims.y));
let base_scale = min_dim / 50.0;  // ~50 pixels per frost cell
let n0 = value_noise(uv * base_scale);
let n1 = value_noise(warped_uv * base_scale * 2.0);
```

**Tests to Add**:

1. `test_frost_ice_pattern_scales_with_image_size`: Render the same scene at 256x256 and 512x512.
   The frost pattern should have similar visual density (approximately same number of visible frost
   "cells" in each, not twice as many in the larger image).

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_frost_ice_blur_creates_diffuse_appearance`

```rust
/// Verify blur parameter creates a diffuse appearance by reducing variance
/// in the frost region on a high-contrast input.
#[test]
fn test_frost_ice_blur_creates_diffuse_appearance() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Create checkerboard pattern (high contrast, high variance)
    let mut img = Rgba16Image::new(64, 64);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let is_white = (x + y) % 2 == 0;
        let val = if is_white { 60000u16 } else { 5000u16 };
        *pixel = Rgba([val, val, val, 65535]);
    }

    let out_no_blur = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "frost_ice",
            values: vec![1.0, 0.0, 0.0, 1.0], // coverage, distortion, blur=0, strength
        }],
    );

    let out_with_blur = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "frost_ice",
            values: vec![1.0, 0.0, 1.0, 1.0], // coverage, distortion, blur=1, strength
        }],
    );

    // Compute variance of red channel in center region (fully frosted)
    fn variance(img: &Rgba16Image, x_range: Range<u32>, y_range: Range<u32>) -> f64 {
        let pixels: Vec<f64> = img.enumerate_pixels()
            .filter(|(x, y, _)| x_range.contains(x) && y_range.contains(y))
            .map(|(_, _, p)| p[0] as f64)
            .collect();
        let mean = pixels.iter().sum::<f64>() / pixels.len() as f64;
        pixels.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / pixels.len() as f64
    }

    let var_no_blur = variance(&out_no_blur, 20..44, 20..44);
    let var_with_blur = variance(&out_with_blur, 20..44, 20..44);

    assert!(
        var_with_blur < var_no_blur * 0.5,
        "blur should significantly reduce variance: no_blur={:.0}, with_blur={:.0}",
        var_no_blur, var_with_blur
    );
}
```

#### `test_frost_ice_blur_zero_is_sharp`

```rust
/// Verify blur=0.0 produces sharp output (no blur applied).
#[test]
fn test_frost_ice_blur_zero_is_sharp() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Single bright pixel on dark background - should remain a point, not spread
    let mut img = Rgba16Image::new(32, 32);
    for pixel in img.pixels_mut() {
        *pixel = Rgba([5000, 5000, 5000, 65535]);
    }
    *img.get_pixel_mut(16, 16) = Rgba([60000, 60000, 60000, 65535]);

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "frost_ice",
            values: vec![1.0, 0.0, 0.0, 1.0], // blur=0
        }],
    );

    // The bright pixel should still be the brightest point (possibly shifted by distortion=0)
    let max_val = out.pixels().map(|p| p[0]).max().unwrap();
    let bright_count = out.pixels().filter(|p| p[0] > max_val - 5000).count();

    assert!(
        bright_count <= 4,
        "with blur=0, bright spot should not spread significantly: {} bright pixels",
        bright_count
    );
}
```

#### `test_frost_ice_blur_respects_frost_mask`

```rust
/// Verify areas outside frost region remain sharp even with blur=1.0.
#[test]
fn test_frost_ice_blur_respects_frost_mask() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Checkerboard - center should remain sharp with coverage=0
    let mut img = Rgba16Image::new(64, 64);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let is_white = (x + y) % 2 == 0;
        let val = if is_white { 60000u16 } else { 5000u16 };
        *pixel = Rgba([val, val, val, 65535]);
    }

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "frost_ice",
            values: vec![0.0, 0.0, 1.0, 1.0], // coverage=0, blur=1
        }],
    );

    // With coverage=0, frost mask is 0 everywhere, so output should equal input
    for y in 0..64 {
        for x in 0..64 {
            let expected = if (x + y) % 2 == 0 { 60000 } else { 5000 };
            let actual = out.get_pixel(x, y)[0];
            assert!(
                (actual as i32 - expected as i32).abs() < 500,
                "at ({}, {}): expected ~{}, got {}",
                x, y, expected, actual
            );
        }
    }
}
```

### PR 2 Tests (Detailed)

#### `test_frost_ice_pattern_scales_with_image_size`

```rust
/// Verify frost pattern density is consistent across image sizes.
#[test]
fn test_frost_ice_pattern_scales_with_image_size() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Render at two different sizes
    let img_small = make_solid_image(128, 128, 32767, 32767, 32767);
    let img_large = make_solid_image(256, 256, 32767, 32767, 32767);

    let out_small = roundtrip(
        &mut renderer,
        &engine,
        &img_small,
        &[Transform {
            shader_id: "frost_ice",
            values: vec![1.0, 0.0, 0.3, 1.0],
        }],
    );

    let out_large = roundtrip(
        &mut renderer,
        &engine,
        &img_large,
        &[Transform {
            shader_id: "frost_ice",
            values: vec![1.0, 0.0, 0.3, 1.0],
        }],
    );

    // Count zero-crossings (transitions) along a horizontal line to measure pattern density
    fn count_transitions(img: &Rgba16Image, y: u32) -> usize {
        let mean: u16 = (img.enumerate_pixels()
            .filter(|(_, py, _)| *py == y)
            .map(|(_, _, p)| p[0] as u32)
            .sum::<u32>() / img.width()) as u16;

        img.enumerate_pixels()
            .filter(|(_, py, _)| *py == y)
            .collect::<Vec<_>>()
            .windows(2)
            .filter(|w| (w[0].2[0] > mean) != (w[1].2[0] > mean))
            .count()
    }

    // Sample a line through the middle
    let trans_small = count_transitions(&out_small, 64);
    let trans_large = count_transitions(&out_large, 128);

    // Pattern density should be similar (large image has ~2x width but should NOT have ~2x transitions)
    // Allow significant tolerance since this is approximate
    let ratio = trans_large as f32 / trans_small as f32;
    assert!(
        (0.8..=1.5).contains(&ratio),
        "pattern density should scale with image, not be 2x denser: \
         small={}, large={}, ratio={:.2}",
        trans_small, trans_large, ratio
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Frost effect produces visible blur/diffusion at default settings
- [ ] Blur=0 produces sharp displacement effect (backward compatible)
- [ ] Blur=1 produces soft, frosted-glass appearance
- [ ] Coverage=0 leaves entire image sharp (identity outside frost region)
- [ ] Strength=0 returns source unchanged (identity)
- [ ] Pattern looks similar on 512px and 2048px images (scale-independent)

---

## References

- [Optical properties of ice and snow - Royal Society Publishing](https://royalsocietypublishing.org/rsta/article/377/2146/20180161)
- [Shader Library: Frosted Glass Post Processing Shader - Geeks3D](https://www.geeks3d.com/20101228/shader-library-frosted-glass-post-processing-shader-glsl/)
- [Visual simulation of glazed frost - ResearchGate](https://www.researchgate.net/publication/262207837_Visual_simulation_of_glazed_frost)
- [An optimized real time algorithm for window frost formation - Springer](https://link.springer.com/article/10.1007/s11042-017-4819-2)
