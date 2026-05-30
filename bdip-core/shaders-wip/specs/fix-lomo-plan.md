# Fix Lomo Transform

## Problem Summary

The current lomo transform implements vignetting and saturation boost but is **missing the defining
characteristic of the Lomography aesthetic: cross-processing color shifts**. Cross-processing (E-6
slide film developed in C-41 chemicals) produces distinctive cyan/blue shadows and yellow/warm
highlights via per-channel tone curve manipulation. Without this, the shader produces a generic
vignette+saturation effect rather than an authentic Lomo look.

### Critical Issues

1. **Missing cross-processing color shift** (lomo.wgsl:32-37): The shader only applies uniform
   saturation boost via Rec.709 luminance interpolation. It performs no per-channel curve
   manipulation. The authentic Lomo look requires:
   - Blue channel: inverted S-curve (boost shadows → cyan tint, reduce highlights → yellow tint)
   - Red/Green channels: standard S-curves (boost highlights, reduce shadows)

   This is the #1 missing feature. Without cross-processing, the effect is not recognizably "Lomo."

2. **No contrast enhancement** (lomo.wgsl): Lomo images are characterized by high contrast. The
   current implementation applies no contrast adjustment. Cross-processing inherently increases
   contrast, but even beyond that, the Lomo LC-A lens produces punchy, contrasty images.

### Moderate Issues

3. **Limited parameterization** (mod.rs:17-23): A single "strength" slider controls both vignette
   and saturation simultaneously. Users cannot independently adjust:
   - Vignette intensity
   - Cross-processing / color shift intensity
   - Saturation amount

   While some users prefer simplicity, the current approach prevents achieving specific Lomo looks
   (e.g., strong vignette with subtle color shift, or vice versa).

### What Works Correctly

- Vignette algorithm: Radial distance with smoothstep falloff is appropriate
- Saturation boost: Mix-from-luminance approach is correct
- Rec.709 luminance weights (0.2126, 0.7152, 0.0722) are correct for linear-light content
- Identity at strength=0 works properly
- Test coverage is comprehensive

---

## Implementation Plan

### PR 1: Add Cross-Processing Color Curves

**Goal**: Implement per-channel tone curves to create the characteristic cross-processed look.

**Scope**:
- Modify `lomo.wgsl` to apply per-channel S-curves
- Red/Green channels: standard S-curve (boost highlights, reduce shadows)
- Blue channel: inverted S-curve (boost shadows, reduce highlights)
- Blend curve effect by strength parameter

**Implementation Details**:

Cross-processing can be approximated with cubic curves applied to each channel. The curves are
designed to work in linear light space (the pipeline handles gamma).

```wgsl
// Cross-processing curves (approximate)
// Standard S-curve for R and G: x + k * x * (1 - x) * (2x - 1)
// Inverted S-curve for B: x - k * x * (1 - x) * (2x - 1)
// Where k controls curve intensity (0.3-0.5 is typical)

fn s_curve(x: f32, intensity: f32) -> f32 {
    // S-curve: boosts highlights, reduces shadows
    return x + intensity * x * (1.0 - x) * (2.0 * x - 1.0);
}

fn inv_s_curve(x: f32, intensity: f32) -> f32 {
    // Inverted S-curve: boosts shadows (adds blue), reduces highlights (adds yellow)
    return x - intensity * x * (1.0 - x) * (2.0 * x - 1.0);
}

// Apply to each channel with strength blending:
let curve_intensity = 0.4 * params.strength;
let r_curved = s_curve(color.r, curve_intensity);
let g_curved = s_curve(color.g, curve_intensity * 0.7);  // Less aggressive on green
let b_curved = inv_s_curve(color.b, curve_intensity);
```

The green channel gets a less aggressive curve (0.7x) because equal R and G curves can produce
overly orange highlights. The blue inverted S-curve is the signature of the cross-processed look.

**Algorithm Order** (updated shader flow):
1. Load pixel color
2. Apply cross-processing curves (per-channel)
3. Apply saturation boost to curved result
4. Apply vignette
5. Output

**Tests to Add**:

1. `test_lomo_cross_processing_shifts_shadow_blue`: Full-strength lomo on a dark gray pixel should
   have higher B relative to R than the identity pass (blue shadows).

2. `test_lomo_cross_processing_shifts_highlight_yellow`: Full-strength lomo on a bright pixel should
   have lower B relative to R than identity (yellow/warm highlights).

**Tests to Update**:
- `test_lomo_full_strength_increases_saturation`: Assertions may need adjustment due to curve
  effects interacting with saturation

---

### PR 2: Add Contrast Enhancement

**Goal**: Increase image contrast to match the punchy Lomo LC-A look.

**Scope**:
- Add contrast adjustment to `lomo.wgsl` after cross-processing curves
- Contrast is applied as linear interpolation toward/away from mid-gray

**Implementation Details**:

```wgsl
// Contrast enhancement: pull values away from 0.5 (linear space midpoint)
// contrast = 1.0 means no change, >1.0 increases contrast
let contrast_factor = 1.0 + 0.3 * params.strength;  // Max 1.3x at full strength
let contrasted = (curved_rgb - 0.5) * contrast_factor + 0.5;
```

This should be applied after curves but before saturation, as contrast affects the overall dynamic
range that saturation then operates on.

**Tests to Add**:

1. `test_lomo_increases_contrast`: Compare pixel spread (max-min across image) at strength=0 vs
   strength=1. Full strength should have wider spread (higher contrast).

---

### PR 3: Split Parameters for Fine Control (Optional Enhancement)

**Goal**: Allow independent control of vignette, color shift, and saturation.

**Scope**:
- Add `vignette` slider (0.0-1.0, default 0.7)
- Add `color_shift` slider (0.0-1.0, default 0.7)
- Keep `strength` as master blend
- Update `LomoParams` struct

**New Parameter Structure**:

```rust
pub struct LomoParams {
    pub strength: f32,     // 0.0-1.0, master blend
    pub vignette: f32,     // 0.0-1.0, vignette intensity
    pub color_shift: f32,  // 0.0-1.0, cross-processing intensity
    pub saturation: f32,   // 0.0-1.0, saturation boost (0.5 = current 1.5x max)
}
```

**Rationale**: This PR is optional but would give users flexibility to achieve different Lomo
sub-styles. Some photographers prefer heavy vignette with subtle color; others want strong color
shift with minimal vignette.

**Tests to Add**:

1. `test_lomo_vignette_slider_controls_corner_darkening`: Zero vignette should not darken corners.
2. `test_lomo_color_shift_slider_controls_blue_shadows`: Zero color_shift should not shift shadows
   toward blue.

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_lomo_cross_processing_shifts_shadow_blue`

```rust
/// Dark pixels should shift toward blue (higher B relative to R) with cross-processing.
#[test]
fn test_lomo_cross_processing_shifts_shadow_blue() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Dark gray pixel (shadows)
    let img = make_solid_image(2, 2, 8192, 8192, 8192);

    let identity = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "lomo",
            values: vec![0.0],
        }],
    );
    let lomo = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "lomo",
            values: vec![1.0],
        }],
    );

    // Cross-processing should boost blue in shadows relative to red
    // Compare B/R ratio (using first pixel, near center to minimize vignette)
    let id_px = identity.get_pixel(0, 0);
    let lomo_px = lomo.get_pixel(0, 0);

    let id_ratio = id_px[2] as f32 / id_px[0].max(1) as f32;
    let lomo_ratio = lomo_px[2] as f32 / lomo_px[0].max(1) as f32;

    assert!(
        lomo_ratio > id_ratio,
        "cross-processing should shift shadows toward blue: identity B/R={:.3}, lomo B/R={:.3}",
        id_ratio,
        lomo_ratio
    );
}
```

#### `test_lomo_cross_processing_shifts_highlight_yellow`

```rust
/// Bright pixels should shift toward yellow (lower B relative to R) with cross-processing.
#[test]
fn test_lomo_cross_processing_shifts_highlight_yellow() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Bright gray pixel (highlights)
    let img = make_solid_image(2, 2, 55000, 55000, 55000);

    let identity = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "lomo",
            values: vec![0.0],
        }],
    );
    let lomo = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "lomo",
            values: vec![1.0],
        }],
    );

    // Cross-processing should reduce blue in highlights relative to red (yellow shift)
    let id_px = identity.get_pixel(0, 0);
    let lomo_px = lomo.get_pixel(0, 0);

    let id_ratio = id_px[2] as f32 / id_px[0].max(1) as f32;
    let lomo_ratio = lomo_px[2] as f32 / lomo_px[0].max(1) as f32;

    assert!(
        lomo_ratio < id_ratio,
        "cross-processing should shift highlights toward yellow: identity B/R={:.3}, lomo B/R={:.3}",
        id_ratio,
        lomo_ratio
    );
}
```

### PR 2 Tests (Detailed)

#### `test_lomo_increases_contrast`

```rust
/// Full-strength lomo should increase contrast (wider value spread).
#[test]
fn test_lomo_increases_contrast() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Gradient image with varied luminance values
    let mut img = ImageBuffer::<Rgba<u16>, _>::new(4, 4);
    for (i, pixel) in img.pixels_mut().enumerate() {
        let val = ((i as f32 / 15.0) * 65535.0) as u16;
        *pixel = Rgba([val, val, val, 65535]);
    }

    let identity = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "lomo",
            values: vec![0.0],
        }],
    );
    let lomo = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "lomo",
            values: vec![1.0],
        }],
    );

    // Measure spread (max - min) of R channel
    let spread = |img: &Rgba16Image| {
        let vals: Vec<u16> = img.pixels().map(|p| p[0]).collect();
        vals.iter().max().unwrap() - vals.iter().min().unwrap()
    };

    let id_spread = spread(&identity);
    let lomo_spread = spread(&lomo);

    // Lomo should have higher contrast (wider spread), accounting for vignette
    // The center pixels should show contrast increase even if corners are darkened
    assert!(
        lomo_spread >= id_spread * 9 / 10, // Allow some tolerance due to vignette
        "lomo should maintain or increase contrast: identity spread={}, lomo spread={}",
        id_spread,
        lomo_spread
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Visual test: dark regions have visible blue/cyan tint
- [ ] Visual test: bright regions have visible yellow/warm tint
- [ ] Visual test: vignette darkens corners appropriately
- [ ] Visual test: overall image has punchy, contrasty look
- [ ] Strength=0 returns source unchanged (identity)
- [ ] Alpha channel is preserved

---

## References

- [Lomography - Wikipedia](https://en.wikipedia.org/wiki/Lomography)
- [Cross Processing - Wikipedia](https://en.wikipedia.org/wiki/Cross_processing)
- [Cross Processing Explained - The Darkroom](https://thedarkroom.com/cross-processing-film/)
- [Digital Cross Processing in Photoshop - Photography Mad](https://www.photographymad.com/pages/view/digital-cross-processing-in-photoshop)
- [The Ultimate Lomo Photography Effect Tutorial - SLR Lounge](https://www.slrlounge.com/the-ultimate-lomo-photography-effect-tutorial-lomography-photoshop-video-tutorial/)
- [What is Lomography - Expert Photography](https://expertphotography.com/what-is-lomography/)
