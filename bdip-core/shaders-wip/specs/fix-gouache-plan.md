# Fix Gouache Transform

## Problem Summary

The current gouache transform uses fundamentally incorrect smoothing that produces a blurry photo
effect instead of the characteristic flat, opaque look of gouache paint. Gouache is defined by
flat color areas with preserved edges, not uniform blur.

### Critical Issues

1. **Wrong smoothing algorithm (CRITICAL)**: Uses Gaussian blur instead of edge-preserving
   smoothing. Gaussian blur uniformly blurs everything including edges, producing a soft-focus
   photograph look. Gouache paint has flat color regions WITH sharp, defined edges. The standard
   approach for painterly rendering is edge-preserving filters like Kuwahara or bilateral filters,
   which smooth textures while maintaining edge contrast.

   **Current behavior** (`gouache_blur_h.wgsl:54-63`, `gouache_blur_v.wgsl:46-55`):
   ```wgsl
   for (var t: i32 = -radius; t <= radius; t = t + 1) {
       // Standard Gaussian convolution - blurs edges uniformly
       let w = exp(-f32(t * t) / two_sigma_sq);
       accum += s * w;
   }
   ```

   **Expected behavior**: Edge-preserving smoothing that creates flat poster-paint regions while
   maintaining boundaries between distinct color areas.

2. **Result looks like blurred photo, not paint**: The combination of Gaussian blur + saturation
   boost produces an image that looks like a soft-focus photograph with vivid colors, not gouache
   paint. Authentic gouache simulation requires the "flat, matte, opaque" quality achieved through
   edge-aware smoothing.

### Moderate Issues

3. **Single parameter conflates distinct effects**: The `strength` parameter simultaneously controls
   both smoothing intensity and saturation boost amount. These are independent artistic choices
   that should be separately adjustable.

4. **No color quantization option**: Traditional gouache paintings often exhibit a limited color
   palette with flat areas of similar hues. Adding optional color quantization would enhance the
   "painted" appearance.

### What Works Correctly

- **Saturation boost logic** (`gouache_color.wgsl:48-52`): The saturation boost using Rec.709 luma
  and mix() is mathematically correct and appropriate for gouache's vibrant colors.
- **Multi-pass architecture**: The 3-pass structure (smooth → smooth → color blend) is reasonable.
- **Alpha preservation**: All passes correctly preserve alpha.
- **Identity at strength=0**: The shader correctly returns the source unchanged when strength=0.

### Current vs Expected Parameters

| Current Parameter | Issue |
|-------------------|-------|
| Strength (0.0–1.0) | Controls both smoothing and saturation; should be split |

| Missing Parameters | Purpose |
|--------------------|---------|
| Edge Sharpness | Control edge preservation threshold |
| Saturation | Independent control over color boost |

---

## Research Sources

- [Gouache - Wikipedia](https://en.wikipedia.org/wiki/Gouache): Defines gouache as opaque, matte,
  with flat even colors and larger pigment particles than watercolor.
- [Kuwahara filter - Wikipedia](https://en.wikipedia.org/wiki/Kuwahara_filter): Standard
  edge-preserving smoothing filter for artistic imaging that "smoothes noise while preserving
  major edges."
- [Bilateral filter - Wikipedia](https://en.wikipedia.org/wiki/Bilateral_filter): "Non-linear,
  edge-preserving, and noise-reducing smoothing filter."
- [On Crafting Painterly Shaders - Maxime Heckel](https://blog.maximeheckel.com/posts/on-crafting-painterly-shaders/):
  Describes Kuwahara filter as the standard approach for "transforming any image input into a
  painting-like work of art."
- [Jackson's Art - Guide to Gouache](https://www.jacksonsart.com/a-guide-to-gouache): Describes
  gouache as "opaque and matte, with a flat, shine-free surface."

---

## Implementation Plan

### PR 1: Replace Gaussian Blur with Kuwahara Filter

**Goal**: Replace the two-pass Gaussian blur with a single-pass Kuwahara filter to achieve
edge-preserving smoothing characteristic of gouache paint.

**Scope**:
- Delete `gouache_blur_h.wgsl` and `gouache_blur_v.wgsl`
- Create `gouache_smooth.wgsl` implementing Kuwahara filter
- Update `mod.rs` to use single smoothing pass instead of two blur passes
- Rename `strength` to `smoothing` for clarity

**New Pass Structure**:
```rust
const PASSES: &'static [PassDef] = &[
    // Pass 1: Kuwahara edge-preserving smooth
    PassDef {
        label: "smooth",
        wgsl_source: include_str!("gouache_smooth.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Scratch("smoothed"),
        output_scale: PassScale::Full,
        aux_textures: &[],
    },
    // Pass 2: blend with source and boost saturation
    PassDef {
        label: "color",
        wgsl_source: include_str!("gouache_color.wgsl"),
        inputs: &[PassInput::Source, PassInput::Scratch("smoothed")],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    },
];
```

**Kuwahara Filter Algorithm** (pseudocode for `gouache_smooth.wgsl`):
```wgsl
// Kuwahara filter divides pixel neighborhood into 4 overlapping quadrants.
// For each quadrant, compute mean color and variance.
// Output = mean of quadrant with lowest variance.
// This smooths uniform regions while preserving edges.

const KERNEL_SIZE: i32 = 4;  // Quadrant size, scaled by strength

fn compute_quadrant_stats(center: vec2<i32>, offset: vec2<i32>) -> QuadrantStats {
    var sum = vec3<f32>(0.0);
    var sum_sq = vec3<f32>(0.0);
    var count = 0.0;
    
    for (var y = 0; y <= KERNEL_SIZE; y++) {
        for (var x = 0; x <= KERNEL_SIZE; x++) {
            let sample_pos = center + offset + vec2<i32>(x, y);
            let color = textureLoad(input_texture, clamp(sample_pos, ...), 0).rgb;
            sum += color;
            sum_sq += color * color;
            count += 1.0;
        }
    }
    
    let mean = sum / count;
    let variance = dot(sum_sq / count - mean * mean, vec3<f32>(1.0));
    return QuadrantStats(mean, variance);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<i32>(gid.xy);
    let k = i32(f32(KERNEL_SIZE) * params.smoothing);
    
    // Four quadrants: top-left, top-right, bottom-left, bottom-right
    let q0 = compute_quadrant_stats(coord, vec2<i32>(-k, -k));
    let q1 = compute_quadrant_stats(coord, vec2<i32>(0, -k));
    let q2 = compute_quadrant_stats(coord, vec2<i32>(-k, 0));
    let q3 = compute_quadrant_stats(coord, vec2<i32>(0, 0));
    
    // Select mean of quadrant with lowest variance
    var best_mean = q0.mean;
    var best_var = q0.variance;
    if (q1.variance < best_var) { best_mean = q1.mean; best_var = q1.variance; }
    if (q2.variance < best_var) { best_mean = q2.mean; best_var = q2.variance; }
    if (q3.variance < best_var) { best_mean = q3.mean; }
    
    textureStore(output_texture, coord, vec4<f32>(best_mean, src_alpha));
}
```

**Tests to Add**:

1. `test_gouache_edge_preservation`: On a step-edge image, verify that the edge remains sharp
   after smoothing (unlike Gaussian blur which would smear it).

2. `test_gouache_texture_smoothing`: On a noisy but uniform-color region, verify smoothing reduces
   noise while keeping the average color.

**Tests to Update**:
- `test_gouache_smoothing_reduces_edge_contrast`: This test currently passes because Gaussian blur
  also reduces edge contrast, but the new implementation should show LESS edge contrast reduction
  while still smoothing texture.

---

### PR 2: Split Strength into Smoothing and Saturation Parameters

**Goal**: Provide independent control over smoothing intensity and saturation boost.

**Scope**:
- Update `GouacheParams` struct with two parameters
- Update slider definitions in `mod.rs`
- Update `gouache_color.wgsl` to use separate saturation parameter

**New Parameters**:
```rust
pub struct GouacheParams {
    pub smoothing: f32,   // 0.0–1.0, default 0.5
    pub saturation: f32,  // 0.0–1.0, default 0.5
    pub _padding: [f32; 2],
}

const PARAM: ParamKind = ParamKind::Sliders(&[
    SliderDef {
        name: "Smoothing",
        min: 0.0,
        max: 1.0,
        default: 0.5,
        description: "Controls how much texture detail is flattened into solid color regions. \
                      Higher values create more paint-like flat areas.",
    },
    SliderDef {
        name: "Saturation",
        min: 0.0,
        max: 1.0,
        default: 0.5,
        description: "Boosts color vibrancy to match gouache's opaque, pigment-rich appearance. \
                      0.0 leaves colors unchanged; 1.0 applies maximum saturation boost.",
    },
]);
```

**Shader Changes** (`gouache_color.wgsl`):
```wgsl
// Use params.saturation instead of params.strength for saturation boost
let sat_boost = params.saturation * MAX_SAT_BOOST;
```

**Tests to Add**:

1. `test_gouache_smoothing_independent_of_saturation`: Apply smoothing=1.0, saturation=0.0 and
   verify texture is flattened but colors are not boosted.

2. `test_gouache_saturation_independent_of_smoothing`: Apply smoothing=0.0, saturation=1.0 and
   verify colors are boosted but no smoothing occurs.

---

### PR 3 (Optional): Add Color Quantization

**Goal**: Add optional color quantization to enhance the poster-paint appearance.

**Scope**:
- Add `quantization` parameter (0.0 = off, 1.0 = strong)
- Implement color quantization in the color pass

This PR is optional and could be deferred if the basic Kuwahara + saturation approach produces
satisfactory results.

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_gouache_edge_preservation`

```rust
/// Verify that the Kuwahara filter preserves edges better than Gaussian blur would.
/// On a sharp step-edge image, the edge should remain relatively sharp after smoothing.
#[test]
fn test_gouache_edge_preservation() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Create step-edge image: left half dark, right half bright
    let mut img = crate::Rgba16Image::new(64, 64);
    for y in 0..64u32 {
        for x in 0..64u32 {
            let v: u16 = if x < 32 { 10000 } else { 55000 };
            img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
        }
    }

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "gouache",
            values: vec![0.8, 0.0], // high smoothing, no saturation
        }],
    );

    // Sample pixels near the edge (at x=31 and x=32)
    // Edge-preserving filter should maintain significant contrast
    let left_edge = out.get_pixel(30, 32)[0] as i32;
    let right_edge = out.get_pixel(33, 32)[0] as i32;
    let edge_contrast = (right_edge - left_edge).abs();

    // Original contrast is 45000. Edge-preserving filter should retain >50% of contrast.
    assert!(
        edge_contrast > 22000,
        "edge should be preserved: left={}, right={}, contrast={}",
        left_edge, right_edge, edge_contrast
    );
}
```

#### `test_gouache_texture_smoothing`

```rust
/// Verify that noisy uniform regions are smoothed while preserving the average color.
#[test]
fn test_gouache_texture_smoothing() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Create noisy image with values fluctuating around 32767
    let mut img = crate::Rgba16Image::new(32, 32);
    let mut rng_state = 12345u32;
    for y in 0..32u32 {
        for x in 0..32u32 {
            // Simple LCG for deterministic "noise"
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((rng_state >> 16) % 10000) as i32 - 5000;
            let v = (32767 + noise).clamp(0, 65535) as u16;
            img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
        }
    }

    // Compute input variance
    let input_variance = compute_variance(&img);

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "gouache",
            values: vec![1.0, 0.0], // full smoothing, no saturation
        }],
    );

    // Compute output variance
    let output_variance = compute_variance(&out);

    // Smoothing should reduce variance significantly
    assert!(
        output_variance < input_variance * 0.3,
        "smoothing should reduce variance: input={}, output={}",
        input_variance, output_variance
    );
}

fn compute_variance(img: &Rgba16Image) -> f64 {
    let pixels: Vec<f64> = img.pixels().map(|p| p[0] as f64).collect();
    let mean = pixels.iter().sum::<f64>() / pixels.len() as f64;
    pixels.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / pixels.len() as f64
}
```

### PR 2 Tests (Detailed)

#### `test_gouache_smoothing_independent_of_saturation`

```rust
/// Verify smoothing works independently of saturation parameter.
#[test]
fn test_gouache_smoothing_independent_of_saturation() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Chromatic input
    let img = make_solid_image(16, 16, 50000, 32767, 10000);

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "gouache",
            values: vec![1.0, 0.0], // full smoothing, zero saturation
        }],
    );

    // Colors should not be boosted (R should not increase, B should not decrease)
    let pixel = out.get_pixel(8, 8);
    let r_diff = (pixel[0] as i32 - 50000).abs();
    assert!(
        r_diff < 2000,
        "saturation=0 should not boost R channel: expected ~50000, got {}",
        pixel[0]
    );
}
```

#### `test_gouache_saturation_independent_of_smoothing`

```rust
/// Verify saturation boost works independently of smoothing parameter.
#[test]
fn test_gouache_saturation_independent_of_smoothing() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Chromatic input
    let img = make_solid_image(4, 4, 50000, 32767, 10000);

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "gouache",
            values: vec![0.0, 1.0], // zero smoothing, full saturation
        }],
    );

    // R (above luma) should be pushed higher, B (below luma) should be pushed lower
    let pixel = out.get_pixel(0, 0);
    assert!(
        pixel[0] as i32 > 50000,
        "saturation=1 should boost R: expected >50000, got {}",
        pixel[0]
    );
    assert!(
        (pixel[2] as i32) < 10000,
        "saturation=1 should reduce B: expected <10000, got {}",
        pixel[2]
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Visual inspection: edges remain sharp while textures are flattened
- [ ] Visual inspection: result looks like flat, opaque paint, not a blurry photo
- [ ] Smoothing parameter independently controls texture flattening
- [ ] Saturation parameter independently controls color vibrancy
- [ ] smoothing=0, saturation=0 returns source unchanged (identity)
- [ ] Alpha is preserved at all parameter combinations

---

## References

- [Gouache - Wikipedia](https://en.wikipedia.org/wiki/Gouache)
- [Kuwahara filter - Wikipedia](https://en.wikipedia.org/wiki/Kuwahara_filter)
- [Bilateral filter - Wikipedia](https://en.wikipedia.org/wiki/Bilateral_filter)
- [On Crafting Painterly Shaders - Maxime Heckel](https://blog.maximeheckel.com/posts/on-crafting-painterly-shaders/)
- [Jackson's Art - Guide to Gouache](https://www.jacksonsart.com/a-guide-to-gouache)
- [Image and Video Abstraction by Anisotropic Kuwahara Filtering](https://dl.acm.org/doi/10.1145/2024676.2024686)
