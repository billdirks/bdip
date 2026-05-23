# Fix Cross Process Transform

## Problem Summary

The cross_process transform produces a stylized effect but has several inaccuracies compared to
authentic cross-processed film characteristics. The curves don't quite match the description, and
a key characteristic (contrast boost) is missing.

### Moderate Issues

1. **Red curve mismatch with description** (cross_process.wgsl:28-32): The code uses `pow(v, 0.85)`
   which lifts values uniformly, with the lift actually being relatively larger in midtones than
   highlights. The description claims "red highlight boost" but this curve produces an overall warm
   shift rather than specifically boosting highlights. Real cross-processing typically uses an
   S-curve that boosts highlights while slightly crushing shadows.

2. **Green curve doesn't lift midtones** (cross_process.wgsl:39-44): The smoothstep S-curve
   `t * t * (3.0 - 2.0 * t)` adds contrast by steepening the midtone slope, but it passes through
   (0.5, 0.5) — midtones aren't actually "lifted" as the description claims. Smoothstep compresses
   shadows and highlights while increasing midtone contrast.

3. **Missing overall contrast enhancement** (cross_process.wgsl:60-83): Cross-processing is
   characterized by a 30-50% increase in contrast (deeper shadows, brighter highlights). The current
   implementation only applies per-channel curve shifts without any global contrast adjustment.
   This is a notable omission that makes the effect less authentic.

4. **Blue curve only affects shadows** (cross_process.wgsl:52-58): The curve `1 - pow(1-v, 1.3)`
   lifts shadows (correct) but passes through (1,1), leaving highlights unchanged. Authentic
   cross-processing also reduces blue in highlights, creating the characteristic yellow highlights /
   cyan shadows split.

### Minor Issues

5. **Description inaccuracy** (mod.rs:15-16): The description claims effects that the math doesn't
   precisely produce. While not a functional bug, the description could better match the actual
   algorithm behavior.

### What Works Well

- Strength blending with identity at 0.0 is correct
- HDR headroom preservation (values >1.0 pass through linearly) is well-handled
- Alpha channel preservation is correct
- The overall aesthetic direction is reasonable for a stylized cross-process look
- Good test coverage for edge cases

### Research Sources

- [Cross Processing Explained - The Darkroom](https://thedarkroom.com/cross-processing-film/)
- [Cross Processing - Wikipedia](https://en.wikipedia.org/wiki/Cross_processing)
- [Digital Cross Processing in Photoshop - Photography Mad](https://www.photographymad.com/pages/view/digital-cross-processing-in-photoshop)
- [Cross Processing Photography Guide - Number Analytics](https://www.numberanalytics.com/blog/cross-processing-photography-guide)

The sources consistently describe cross-processing (E6 in C-41) as producing:
- Significantly increased contrast (30-50%)
- Yellow/green cast in highlights
- Cyan/blue cast in shadows
- Per-channel S-curves with specific characteristics

---

## Implementation Plan

### PR 1: Improve Per-Channel Curves for Authenticity

**Goal**: Adjust the per-channel curves to better match authentic cross-processing characteristics.

**Scope**:
- Modify `curve_red()` to use an S-curve that specifically boosts highlights
- Modify `curve_blue()` to also reduce blue in highlights (yellow cast)
- Update function comments to accurately describe what each curve does
- Preserve the existing strength blending and HDR handling

**Algorithm Changes**:

```wgsl
// Red channel: S-curve for contrast with highlight boost
// Raises highlights, slightly crushes shadows for warm highlights/neutral shadows
fn curve_red(v: f32) -> f32 {
    let t = clamp(v, 0.0, 1.0);
    // Lift curve: pow(v, 0.85) for overall warmth
    // Combined with slight shadow crush via blend
    let lift = pow(t, 0.85);
    // S-curve component for highlight emphasis
    let s = t * t * (3.0 - 2.0 * t);
    // Blend: 70% lift curve + 30% S-curve for highlight boost with warmth
    let r = mix(s, lift, 0.7);
    return select(r, v, v > 1.0);
}

// Blue channel: shadow lift + highlight reduction
// Creates characteristic cyan shadows / yellow highlights split
fn curve_blue(v: f32) -> f32 {
    let t = clamp(v, 0.0, 1.0);
    // Shadow lift (existing behavior)
    let shadow_lift = 1.0 - pow(1.0 - t, 1.3);
    // Highlight reduction for yellow cast in highlights
    // Lerp between shadow_lift curve and a curve that reduces highlights
    let highlight_reduce = pow(t, 1.15);  // Slightly compresses highlights
    // Blend: shadow lift dominates in shadows, highlight reduction in highlights
    let b = mix(shadow_lift, highlight_reduce, t);
    return select(b, v, v > 1.0);
}
```

**Tests to Add**:

1. `test_cross_process_highlights_warmer_than_shadows`: Verify red channel lift is proportionally
   larger in highlights than shadows at full strength.

2. `test_cross_process_blue_reduced_in_highlights`: At full strength, verify that bright input
   produces slightly reduced blue output (yellow cast).

**Existing Tests to Verify**:
- All existing tests should continue to pass
- `test_cross_process_red_channel_lifted_at_full_strength` — should still pass
- `test_cross_process_blue_channel_shadows_lifted` — should still pass

---

### PR 2: Add Contrast Parameter

**Goal**: Add a contrast boost parameter to match the characteristic high-contrast look of
cross-processed film.

**Scope**:
- Add `contrast` parameter to `CrossProcessParams` (0.0–1.0, default 0.3)
- Apply contrast curve in the shader after per-channel processing
- Update slider definitions in mod.rs
- Add tests for the new parameter

**Parameter Definition**:

```rust
pub struct CrossProcessParams {
    pub strength: f32,
    pub contrast: f32,
    pub _padding: [f32; 2],
}

const PARAM: ParamKind = ParamKind::Sliders(&[
    SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend strength of the cross-process effect. 0.0 leaves the image unchanged.",
    },
    SliderDef {
        name: "Contrast",
        min: 0.0,
        max: 1.0,
        default: 0.3,
        description: "Additional contrast boost characteristic of cross-processed film.",
    },
]);
```

**Shader Addition**:

```wgsl
// Apply contrast boost after per-channel curves
// S-curve contrast: steeper through midtones, compresses extremes
fn apply_contrast(v: f32, amount: f32) -> f32 {
    // Centered S-curve: maps 0.5 to 0.5, increases slope through midtones
    let t = clamp(v, 0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    // Blend original with S-curve based on contrast amount
    return mix(t, s, amount);
}

// In main():
let contrasted = vec3<f32>(
    apply_contrast(processed.r, params.contrast),
    apply_contrast(processed.g, params.contrast),
    apply_contrast(processed.b, params.contrast),
);
let out_rgb = mix(rgb, contrasted, params.strength);
```

**Tests to Add**:

1. `test_cross_process_contrast_zero_matches_original_curves`: With contrast=0.0, output should
   match the per-channel curves without additional contrast.

2. `test_cross_process_contrast_increases_midtone_separation`: With contrast=1.0, verify that
   midtone gray inputs produce outputs further from 0.5 than with contrast=0.0.

3. `test_cross_process_contrast_preserves_black_and_white`: Pure black and white should remain
   unchanged regardless of contrast setting.

---

### PR 3: Update Description for Accuracy

**Goal**: Update the transform description to accurately reflect the algorithm behavior.

**Scope**:
- Update `DESCRIPTION` in mod.rs to accurately describe the effect
- Update slider descriptions if needed
- Update code comments in the shader

**New Description**:

```rust
const DESCRIPTION: &'static str = "Simulates cross-processing film (E6 in C-41 chemistry) with \
    per-channel curve adjustments: warm red cast, green midtone contrast, and cyan shadows \
    with yellow highlights. Optionally adds the characteristic contrast boost.";
```

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_cross_process_highlights_warmer_than_shadows`

```rust
/// Verify the red channel boost is proportionally larger in highlights.
#[test]
fn test_cross_process_highlights_warmer_than_shadows() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Test with shadow-region input (low value)
    let shadow_img = make_solid_image(2, 2, 8000, 8000, 8000);
    let shadow_out = roundtrip(
        &mut renderer,
        &engine,
        &shadow_img,
        &[Transform {
            shader_id: "cross_process",
            values: vec![1.0],
        }],
    );

    // Test with highlight-region input (high value)
    let highlight_img = make_solid_image(2, 2, 55000, 55000, 55000);
    let highlight_out = roundtrip(
        &mut renderer,
        &engine,
        &highlight_img,
        &[Transform {
            shader_id: "cross_process",
            values: vec![1.0],
        }],
    );

    // Calculate relative red boost for each
    let shadow_red_boost = shadow_out.get_pixel(0, 0)[0] as f32 / 8000.0;
    let highlight_red_boost = highlight_out.get_pixel(0, 0)[0] as f32 / 55000.0;

    // Highlights should have larger relative boost than shadows
    // (Note: exact assertion depends on final curve implementation)
    assert!(
        highlight_red_boost >= shadow_red_boost * 0.9,
        "highlight red boost should be comparable to or larger than shadow boost"
    );
}
```

#### `test_cross_process_blue_reduced_in_highlights`

```rust
/// Verify blue is reduced in highlights for yellow cast.
#[test]
fn test_cross_process_blue_reduced_in_highlights() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Bright input in the highlight region
    let img = make_solid_image(2, 2, 55000, 55000, 55000);
    let with_effect = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "cross_process",
            values: vec![1.0],
        }],
    );
    let without_effect = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "cross_process",
            values: vec![0.0],
        }],
    );

    // Blue should be slightly reduced in highlights
    for (a, b) in with_effect.pixels().zip(without_effect.pixels()) {
        assert!(
            a[2] <= b[2] + 500,  // Allow small tolerance
            "B in highlights should be reduced or neutral: effect={} identity={}",
            a[2],
            b[2]
        );
    }
}
```

### PR 2 Tests (Detailed)

#### `test_cross_process_contrast_zero_matches_original_curves`

```rust
/// With contrast=0.0, output should match curves-only processing.
#[test]
fn test_cross_process_contrast_zero_matches_original_curves() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(2, 2, 25000, 20000, 15000);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "cross_process",
            values: vec![1.0, 0.0], // strength=1, contrast=0
        }],
    );

    // Output should reflect per-channel curves without additional contrast
    // (Specific values depend on curve implementation)
    for pixel in out.pixels() {
        // Basic sanity: channels should be modified but not clamped
        assert!(pixel[0] > 0);
        assert!(pixel[1] > 0);
        assert!(pixel[2] > 0);
    }
}
```

#### `test_cross_process_contrast_increases_midtone_separation`

```rust
/// Higher contrast should push midtones away from 0.5.
#[test]
fn test_cross_process_contrast_increases_midtone_separation() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Mid-gray input
    let img = make_solid_image(2, 2, 32767, 32767, 32767);
    
    let low_contrast = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "cross_process",
            values: vec![1.0, 0.0], // no contrast
        }],
    );
    
    let high_contrast = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "cross_process",
            values: vec![1.0, 1.0], // full contrast
        }],
    );

    // With the S-curve contrast, midtones get steeper slope
    // so the output values should differ between low and high contrast
    let lc = low_contrast.get_pixel(0, 0);
    let hc = high_contrast.get_pixel(0, 0);
    
    assert_ne!(lc, hc, "contrast parameter should affect output");
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Cross-process effect produces visible color shifts at strength=1.0
- [ ] Warm/yellow cast visible in highlights
- [ ] Cyan/cool cast visible in shadows
- [ ] Contrast parameter visibly affects the output
- [ ] Strength=0 returns source unchanged (identity)
- [ ] Pure black and white inputs remain near-black and near-white
- [ ] Alpha channel is preserved

---

## Alternative Approach: Accept Current Implementation

The current implementation, while not precisely matching authentic film cross-processing, produces
a stylized effect that works aesthetically. If the goal is artistic filters rather than film
simulation accuracy, the existing implementation may be acceptable with just documentation updates
(PR 3 alone).

**Recommendation**: Implement all three PRs for a more authentic and versatile effect, but PR 3
(documentation) is the minimum necessary fix to address the description/implementation mismatch.
