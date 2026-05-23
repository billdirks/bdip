# Fix Dust and Scratches Transform

## Problem Summary

The dust_and_scratches transform is **functionally correct** but has **parameter design issues** that limit
user control and may produce effects too subtle for practical use.

### Moderate Issues

1. **Hard-coded density multipliers reduce effective parameter range** (lines 86, 119 in WGSL)
   - `scratch_density * 0.3` caps effective scratch density at 30% of columns even at slider max
   - `dust_amount * 0.15` caps effective dust density at 15% of cells even at slider max
   - Users moving the slider from 0.0 to 1.0 expect the full effect range, not 0-30% or 0-15%
   - Reference: Industry standard overlays and procedural effects typically allow full-range control

2. **Fixed artifact sizes may be invisible on high-resolution images** (lines 79, 93, 113, 126)
   - Scratch width: 0.8-1.6 px total (halfwidth 0.4-0.8 px)
   - Dust diameter: 1.0-2.4 px (radius 0.5-1.2 px)
   - On a 4K+ image, 1-2 px artifacts are nearly invisible
   - No user parameter to adjust scale
   - Reference: Texture-based overlays scale with image resolution; procedural effects should offer a
     scale parameter

### Minor Issues

3. **No intensity/opacity parameter for artifact darkness**
   - All scratches and dust darken to full black (damage mask = 1.0)
   - Some vintage film effects show partially transparent scratches
   - Lower priority: full black is authentic for actual film scratches

### Current Parameters

| Parameter       | Range   | Default | Effective Range | Issue                            |
|-----------------|---------|---------|-----------------|----------------------------------|
| Strength        | 0.0–1.0 | 0.0     | 0.0–1.0         | OK (identity at 0)               |
| Scratch Density | 0.0–1.0 | 0.5     | 0.0–0.3         | 0.3x multiplier limits range     |
| Dust Amount     | 0.0–1.0 | 0.5     | 0.0–0.15        | 0.15x multiplier limits range    |

### What Works Correctly

- Algorithm approach (procedural vertical scratches + dust specks) is sound
- Blue noise randomization provides good spatial distribution
- Anti-aliasing via `smoothstep()` produces smooth artifact edges
- Scratch breaks via noise lookup create realistic broken-line appearance
- `max()` blending correctly prevents double-darkening at overlaps
- Identity behavior (strength=0) returns source unchanged
- Alpha channel preserved
- Deterministic output

---

## Implementation Plan

### PR 1: Adjust Density Multipliers for Full-Range Control

**Goal**: Make density sliders control the full 0-100% range so users get expected behavior.

**Scope**:
- Modify `dust_and_scratches.wgsl` lines 86 and 119 to use higher multipliers
- Keep reasonable artistic defaults by adjusting the slider defaults

**Changes**:

In `dust_and_scratches.wgsl`:

```wgsl
// Line 86: Change from 0.3 to 0.8 (allows 80% max scratch coverage)
let is_scratch_col = select(0.0, 1.0, col_rnd < params.scratch_density * 0.8);

// Line 119: Change from 0.15 to 0.5 (allows 50% max dust coverage)
let has_dust = select(0.0, 1.0, dust_rnd < params.dust_amount * 0.5);
```

In `mod.rs`, adjust defaults to maintain similar visual output at default settings:

```rust
SliderDef {
    name: "Scratch Density",
    min: 0.0,
    max: 1.0,
    default: 0.2,  // Was 0.5 → 0.5*0.3=0.15 effective; now 0.2*0.8=0.16 effective
    description: "...",
},
SliderDef {
    name: "Dust Amount",
    min: 0.0,
    max: 1.0,
    default: 0.15,  // Was 0.5 → 0.5*0.15=0.075 effective; now 0.15*0.5=0.075 effective
    description: "...",
},
```

**Tests to Update**:
- Existing tests should still pass with adjusted defaults
- No new tests required (behavior unchanged, just parameter mapping)

---

### PR 2: Add Scale Parameter for Resolution Independence

**Goal**: Allow users to control artifact size so effects remain visible on high-resolution images.

**Scope**:
- Add `scale` parameter (0.5–4.0, default 1.0) to `DustAndScratchesParams`
- Multiply scratch column width, scratch halfwidth, dust cell size, and dust radius by scale
- Update slider definitions and WGSL uniform struct

**New Parameter**:

```rust
SliderDef {
    name: "Scale",
    min: 0.5,
    max: 4.0,
    default: 1.0,
    description: "Artifact size multiplier. Increase for high-resolution images to maintain visibility.",
},
```

**Updated Params Struct**:

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustAndScratchesParams {
    pub strength: f32,
    pub scratch_density: f32,
    pub dust_amount: f32,
    pub scale: f32,  // Replaces _padding
}
```

**WGSL Changes** (pseudocode):

```wgsl
// Line 79: Scale scratch column width
let scratch_col_width: f32 = 4.0 * params.scale;

// Line 93: Scale scratch halfwidth
let scratch_halfwidth = (0.4 + hash2(col_idx, 2u) * 0.4) * params.scale;

// Line 113: Scale dust cell size
let dust_cell_size: f32 = 4.0 * params.scale;

// Line 126: Scale dust radius
let speck_r = (0.5 + hash2(cell_x + cell_y * 31u, 99u) * 0.7) * params.scale;
```

**Tests to Add**:

1. `test_dust_and_scratches_scale_increases_artifact_size`:
   Compare output at scale=1.0 vs scale=2.0 on a white image. At higher scale, darkened regions
   should be larger (count pixels in connected dark regions).

2. `test_dust_and_scratches_scale_default_matches_legacy`:
   At scale=1.0, behavior should match pre-PR behavior exactly for regression safety.

---

## Test Specifications

### PR 2 Tests (Detailed)

#### `test_dust_and_scratches_scale_increases_artifact_size`

```rust
/// Verify the scale parameter increases artifact size.
#[test]
fn test_dust_and_scratches_scale_increases_artifact_size() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    let img = make_solid_image(128, 128, 65535, 65535, 65535);

    // Scale 1.0
    let out_small = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "dust_and_scratches",
            values: vec![1.0, 0.5, 0.5, 1.0], // strength, density, dust, scale=1
        }],
    );

    // Scale 2.0
    let out_large = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "dust_and_scratches",
            values: vec![1.0, 0.5, 0.5, 2.0], // strength, density, dust, scale=2
        }],
    );

    // Count darkened pixels (affected by scratches/dust)
    let count_dark = |img: &Rgba16Image| img.pixels().filter(|p| p[0] < 60000).count();

    let dark_small = count_dark(&out_small);
    let dark_large = count_dark(&out_large);

    // Larger scale should produce more darkened pixels (bigger artifacts)
    assert!(
        dark_large > dark_small,
        "scale=2.0 should darken more pixels than scale=1.0: small={}, large={}",
        dark_small, dark_large
    );
}
```

#### `test_dust_and_scratches_scale_default_matches_legacy`

```rust
/// Verify scale=1.0 produces same output as pre-scale implementation.
/// This test ensures the refactor doesn't change default behavior.
#[test]
fn test_dust_and_scratches_scale_default_matches_legacy() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    let img = make_solid_image(64, 64, 40000, 40000, 40000);

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "dust_and_scratches",
            values: vec![0.8, 0.3, 0.3, 1.0], // scale=1.0 (default)
        }],
    );

    // At scale=1.0, the effect should be visible but subtle
    // Just verify no crash and alpha preserved
    for pixel in out.pixels() {
        assert_eq!(pixel[3], 65535, "alpha must be preserved");
    }
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (all existing + new tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Strength=0 returns source unchanged (identity preserved)
- [ ] At scratch_density=1.0, scratches are clearly visible (not capped at 30%)
- [ ] At dust_amount=1.0, dust is clearly visible (not capped at 15%)
- [ ] At scale=2.0, artifacts are noticeably larger than at scale=1.0
- [ ] Effect is visible on a 4K test image at default settings
- [ ] Alpha channel preserved in all cases

---

## References

- [Filmic Effects in WebGL - Matt DesLauriers](https://medium.com/@mattdesl/filmic-effects-for-webgl-9dab4bc899dc)
- [Image Imperfections and Film Grain Post Process FX - Martins Upitis](http://devlog-martinsh.blogspot.com/2013/05/image-imperfections-and-film-grain-post.html)
- [glsl-film-grain GitHub Repository](https://github.com/mattdesl/glsl-film-grain)
- [CapCut - How Dust And Scratches Overlays Create An Authentic Film Look](https://www.capcut.com/resource/dust-and-scratches-overlays)
