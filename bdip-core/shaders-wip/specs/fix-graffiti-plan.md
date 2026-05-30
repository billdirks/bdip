# Fix Graffiti Transform

## Problem Summary

The graffiti transform has a fundamental bug in its blur implementation that produces incorrect
visual output.

### Critical Issue

1. **Incomplete separable blur (horizontal only)**: The `graffiti_bleed.wgsl` shader claims to
   implement a "separable box blur" but only performs horizontal sampling (lines 54-68). The
   comment on lines 54-59 incorrectly states that "vertical is handled implicitly by symmetry" —
   this is mathematically wrong.

   A proper separable box blur requires **two passes**: one horizontal, one vertical. The current
   implementation produces a horizontal motion-blur effect rather than an isotropic 2D blur that
   would simulate spray-paint overspray.

   From [Box blur - Wikipedia](https://en.wikipedia.org/wiki/Box_blur):
   > "The box blur is a separable filter, so that only two 1D passes of averaging 2r+1 pixels will
   > be needed, one horizontal and one vertical, for each pixel."

   **Impact**: The spray-bleed effect only smears horizontally instead of softening uniformly in
   all directions. This is visually incorrect and does not match the intended "diffuse overspray"
   effect described in the parameter documentation.

### Other Observations (No Action Required)

- **Algorithm structure is sound**: Posterization → edge darkening → blend is the correct approach
  for a graffiti/stencil effect
- **Sobel implementation is correct**: Standard 3×3 kernels with proper BT.709 luma weights
- **Quantization formula is correct**: `floor(v * levels) / levels` is standard posterization
- **Parameters are well-chosen**: color_levels [2,16], edge_strength [0.5,5.0], bleed [0,1] are
  reasonable ranges
- **No auxiliary textures**: The shader computes the effect procedurally (correct)

---

## Implementation Plan

### PR 1: Fix Separable Blur with Two-Pass Implementation

**Goal**: Replace the single-pass horizontal blur with a proper two-pass separable blur (horizontal
then vertical).

**Scope**:
- Create new WGSL file `graffiti_bleed_v.wgsl` for the vertical blur pass
- Update `graffiti_bleed.wgsl` to be the horizontal pass (minimal changes, fix misleading comment)
- Update `mod.rs` to add a third pass and use two scratch textures
- Update tests to verify isotropic blur behavior

**Files to Modify**:
- `bdip_core/src/gpu/shaders/graffiti/mod.rs`
- `bdip_core/src/gpu/shaders/graffiti/graffiti_bleed.wgsl`

**Files to Create**:
- `bdip_core/src/gpu/shaders/graffiti/graffiti_bleed_v.wgsl`

**New Pass Configuration**:

```rust
const PASSES: &'static [PassDef] = &[
    PassDef {
        label: "bleed_h",
        wgsl_source: include_str!("graffiti_bleed.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Scratch("bleed_h"),
        output_scale: PassScale::Full,
        aux_textures: &[],
    },
    PassDef {
        label: "bleed_v",
        wgsl_source: include_str!("graffiti_bleed_v.wgsl"),
        inputs: &[PassInput::Scratch("bleed_h")],
        output: PassOutput::Scratch("bleed"),
        output_scale: PassScale::Full,
        aux_textures: &[],
    },
    PassDef {
        label: "quantize",
        wgsl_source: include_str!("graffiti_quantize.wgsl"),
        inputs: &[PassInput::Source, PassInput::Scratch("bleed")],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    },
];
```

**Horizontal Blur Shader** (`graffiti_bleed.wgsl` — update comment, keep algorithm):

```wgsl
// Graffiti — Pass 1a: horizontal blur.
//
// First pass of a separable box blur. Blurs horizontally only; the vertical
// pass (graffiti_bleed_v.wgsl) completes the 2D blur.

// ... (rest of shader unchanged, just remove the misleading comment about
// "vertical is handled implicitly by symmetry")
```

**Vertical Blur Shader** (`graffiti_bleed_v.wgsl`):

```wgsl
// Graffiti — Pass 1b: vertical blur.
//
// Second pass of the separable box blur. Reads the horizontally-blurred
// scratch texture and applies vertical blur to complete the 2D blur.

struct GraffitiParams {
    strength:      f32,
    color_levels:  f32,
    edge_strength: f32,
    bleed:         f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: GraffitiParams;

const BLEED_FRACTION: f32 = 0.015;
const RADIUS_CAP:     i32 = 90;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);

    // Compute blur radius (same formula as horizontal pass).
    let base_radius = BLEED_FRACTION * f32(max(dims.x, dims.y));
    let radius      = min(i32(ceil(base_radius * params.bleed)), RADIUS_CAP);

    if radius == 0 {
        let pixel = textureLoad(src_texture, coord, 0);
        textureStore(dst_texture, coord, pixel);
        return;
    }

    // Vertical blur: sample along a vertical line of height 2*radius+1.
    let diameter = 2 * radius + 1;
    let inv_diam = 1.0 / f32(diameter);
    var accum: vec4<f32> = vec4<f32>(0.0);

    for (var dy: i32 = -radius; dy <= radius; dy++) {
        let tap_y = clamp(coord.y + dy, 0, i32(dims.y) - 1);
        accum += textureLoad(src_texture, vec2<i32>(coord.x, tap_y), 0);
    }

    textureStore(dst_texture, coord, accum * inv_diam);
}
```

**Tests to Add**:

1. `test_graffiti_blur_is_isotropic`: On an image with a single bright pixel on a dark background,
   verify that the blurred result is roughly symmetric in both X and Y directions (not just
   horizontally smeared).

2. `test_graffiti_bleed_affects_vertical_edges`: On a horizontal step-edge image (top half dark,
   bottom half bright), verify that bleed > 0 softens the horizontal boundary. The current broken
   implementation would not affect horizontal edges because it only blurs horizontally.

**Tests to Update**:

- `test_graffiti_registry_metadata`: Update assertion to expect 3 passes instead of 2.

---

## Test Specifications

### `test_graffiti_blur_is_isotropic`

```rust
/// Verify that the blur is 2D (isotropic), not just horizontal.
/// A single bright pixel should spread in all directions, not just horizontally.
#[test]
fn test_graffiti_blur_is_isotropic() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // 64x64 black image with a single bright pixel at center.
    let mut img = crate::Rgba16Image::new(64, 64);
    for y in 0..64u32 {
        for x in 0..64u32 {
            img.put_pixel(x, y, image::Rgba([0, 0, 0, 65535]));
        }
    }
    img.put_pixel(32, 32, image::Rgba([65535, 65535, 65535, 65535]));

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "graffiti",
            values: vec![1.0, 16.0, 0.5, 1.0], // max bleed, minimal quantization effect
        }],
    );

    // Sample pixels at equal distances from center in horizontal and vertical directions.
    // With isotropic blur, they should have similar brightness.
    let horizontal_neighbor = out.get_pixel(32 + 5, 32)[0] as i32;
    let vertical_neighbor   = out.get_pixel(32, 32 + 5)[0] as i32;

    // Allow some tolerance for quantization effects, but they should be comparable.
    let diff = (horizontal_neighbor - vertical_neighbor).abs();
    assert!(
        diff < 5000,
        "blur should be isotropic: horizontal neighbor={}, vertical neighbor={}, diff={}",
        horizontal_neighbor, vertical_neighbor, diff
    );
}
```

### `test_graffiti_bleed_affects_vertical_edges`

```rust
/// Verify that bleed softens horizontal edges (top-to-bottom transitions).
/// The broken horizontal-only blur would not affect these edges.
#[test]
fn test_graffiti_bleed_affects_vertical_edges() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // 128x64 horizontal step: top half dark (21845), bottom half bright (43690).
    // This creates a horizontal edge that only a vertical blur can soften.
    let mut img = crate::Rgba16Image::new(128, 64);
    for y in 0..64u32 {
        for x in 0..128u32 {
            let v: u16 = if y < 32 { 21845 } else { 43690 };
            img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
        }
    }

    let out_no_bleed = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "graffiti",
            values: vec![1.0, 6.0, 0.5, 0.0], // bleed=0
        }],
    );
    let out_bleed = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "graffiti",
            values: vec![1.0, 6.0, 0.5, 1.0], // bleed=1
        }],
    );

    // At least one pixel near the horizontal boundary (y=31 or y=32) must differ.
    // With correct vertical blur, boundary pixels will be softened.
    let any_different = (0..128u32).any(|x| {
        let a = out_no_bleed.get_pixel(x, 31)[0] as i32;
        let b = out_bleed.get_pixel(x, 31)[0] as i32;
        (a - b).abs() > 64
    });

    assert!(
        any_different,
        "bleed should affect horizontal edges (vertical blur must work)"
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (including new isotropic blur tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Graffiti effect with high bleed produces uniform softening in all directions (not just
      horizontal smearing)
- [ ] Horizontal edges (top/bottom transitions) are softened when bleed > 0
- [ ] Vertical edges (left/right transitions) are softened when bleed > 0
- [ ] Visual inspection: spray-bleed effect looks like diffuse overspray, not motion blur
- [ ] Strength=0 returns source unchanged (identity)
- [ ] Alpha is preserved at all settings

---

## References

- [Box blur - Wikipedia](https://en.wikipedia.org/wiki/Box_blur)
- [Posterization - Wikipedia](https://en.wikipedia.org/wiki/Posterization)
- [Separable Gaussian Convolution](https://medium.com/@RaymondTayBL/blurring-at-the-speed-of-light-separable-gaussian-convolution-on-the-gpu-190f8c3d1f87)
- [Intel: Fast Real-Time GPU-Based Image Blur Algorithms](https://www.intel.com/content/www/us/en/developer/articles/technical/an-investigation-of-fast-real-time-gpu-based-image-blur-algorithms.html)
- [Paint.NET Discussion: Posterization Algorithm](https://forums.getpaint.net/topic/110081-posterization-filter-algorithm/)
