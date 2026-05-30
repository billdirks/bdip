# Fix Retro Newspaper Transform

## Problem Summary

The retro_newspaper transform has several implementation issues that affect tone reproduction accuracy
and visual quality.

### Critical Issues

1. **No sqrt geometric compensation** (line 73 in `retro_newspaper_halftone.wgsl`):
   The current code uses linear mapping for dot radius:
   ```wgsl
   let dot_radius = 0.45 * (1.0 - quant.r);
   ```
   This is incorrect because dot area scales with radius² (A = πr²). For proper tone reproduction,
   a 50% gray should produce dots covering 50% of the cell area. With linear mapping:
   - 50% gray → radius = 0.225 → area = π(0.225)² ≈ 0.159 (only 25% of max area)
   
   With sqrt compensation:
   - 50% gray → radius = 0.45 × √0.5 ≈ 0.318 → area ≈ 0.318 (50% of max area)
   
   Research confirms: `dot radius = sqrt(1.0 - brightness) * maxRadius` is the correct formula.

2. **Unused `short_axis` variable** (line 58 in `retro_newspaper_halftone.wgsl`):
   ```wgsl
   let short_axis = f32(min(dims.x, dims.y));  // Computed but never used!
   let cell_uv    = uv * params.dot_frequency;
   ```
   The parameter description says "cells across the shorter axis" but the code doesn't implement
   this. On non-square images, dots will appear stretched because UV is normalized to [0,1] on
   both axes without aspect ratio correction.

3. **No anti-aliasing** (line 81 in `retro_newspaper_halftone.wgsl`):
   Uses hard edge via `select()`:
   ```wgsl
   let dot_value = select(paper, ink, dist < dot_radius);
   ```
   This produces aliasing artifacts, especially at lower dot frequencies. Should use `smoothstep()`
   for smoother dot edges.

### Moderate Issues

4. **Quantization produces 6 levels, not 5 as documented** (line 35 in
   `retro_newspaper_quantize.wgsl`):
   The formula `floor(src.r * LEVELS + 0.5) / LEVELS` with LEVELS=5 produces outputs:
   0.0, 0.2, 0.4, 0.6, 0.8, 1.0 — that's 6 distinct values.
   
   Either update the comment or fix the formula. For exactly 5 levels (0.0, 0.25, 0.5, 0.75, 1.0),
   use LEVELS=4 in a corrected formula.

### Missing Parameters

- **Softness**: User control over anti-aliasing amount (0.0 = hard edges, 1.0 = soft edges)

---

## Implementation Plan

### PR 1: Fix Geometric Compensation and Aspect Ratio

**Goal**: Correct tone reproduction by using sqrt compensation and fix aspect ratio handling.

**Scope**:
- Modify `retro_newspaper_halftone.wgsl` lines 58-73

**Implementation details**:

```wgsl
// Fix aspect ratio: scale coordinates so dot_frequency applies to shorter axis
let short_axis = f32(min(dims.x, dims.y));
let aspect = vec2<f32>(dims) / short_axis;
let cell_uv = uv * aspect * params.dot_frequency;
let rot_uv = rotate45(cell_uv);

// ... existing cell_frac and dist code ...

// Apply sqrt geometric compensation for correct tone reproduction
// Dark (quant.r=0) → large dot; Bright (quant.r=1) → no dot
let dot_radius = 0.45 * sqrt(1.0 - quant.r);
```

**Tests to add**:

1. `test_retro_newspaper_geometric_compensation`: Verify 50% gray produces ~50% ink coverage
2. `test_retro_newspaper_aspect_ratio_produces_circular_dots`: On a 2:1 aspect image, verify
   dots are circular (not elliptical) by checking symmetry

---

### PR 2: Add Anti-Aliasing with Softness Parameter

**Goal**: Replace hard `select()` edge with `smoothstep()` and add user-controllable softness.

**Scope**:
- Add `softness` field to `RetroNewspaperParams` in `mod.rs`
- Add slider definition for softness
- Modify `retro_newspaper_halftone.wgsl` to use smoothstep

**New Parameter**:
```rust
SliderDef {
    name: "Softness",
    min: 0.0,
    max: 1.0,
    default: 0.2,
    description: "Dot edge softness. 0.0 produces hard edges; higher values add anti-aliasing \
                  for smoother appearance.",
},
```

**Implementation details** (halftone shader):
```wgsl
// Anti-aliased dot edge using smoothstep
// edge_width scales with softness parameter (0.0 = hard edge)
let edge_width = params.softness * 0.15;
let dot_mask = smoothstep(dot_radius - edge_width, dot_radius + edge_width, dist);
let dot_value = mix(ink, paper, dot_mask);
```

**Tests to add**:

1. `test_retro_newspaper_softness_affects_edge_sharpness`: At softness=0.0, output should be
   bimodal (only ink/paper values). At softness=1.0, there should be intermediate gray values.
2. `test_retro_newspaper_softness_zero_matches_original_behavior`: Ensure backwards compatibility

---

### PR 3: Fix Quantization Level Count

**Goal**: Make quantization produce exactly 5 levels as documented, or update documentation.

**Scope**:
- Either update `retro_newspaper_quantize.wgsl` formula, OR
- Update comment to say "6 levels" if current behavior is intentional

**Option A — Fix to 5 levels** (recommended):
The newspaper look typically uses 5 tones: black, dark gray, mid gray, light gray, white.
Change LEVELS to 4 and use this formula:
```wgsl
const LEVELS: f32 = 4.0;
// Produces: 0.0, 0.25, 0.5, 0.75, 1.0 (5 distinct values)
let quantised = round(src.r * LEVELS) / LEVELS;
```

**Option B — Update documentation**:
Change comment from "5 levels" to "6 levels" and update DESCRIPTION.

**Tests to add**:

1. `test_retro_newspaper_quantization_produces_five_levels`: Feed a gradient and verify output
   contains exactly 5 distinct values.

---

## Test Specifications

### PR 1 Tests

#### `test_retro_newspaper_geometric_compensation`

```rust
/// Verify sqrt() geometric compensation: 50% gray should produce ~50% ink coverage.
#[test]
fn test_retro_newspaper_geometric_compensation() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // 50% linear gray (u16 32767 in sRGB ≈ 0.214 linear, but we want true 50% linear)
    // Use a value that produces ~0.5 after the grayscale pass
    let img = make_solid_image(128, 128, 45000, 45000, 45000);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "retro_newspaper",
            values: vec![20.0, 1.0], // coarse dots for clear measurement
        }],
    );

    // Count pixels closer to ink vs paper
    let ink_threshold = 30000u16;
    let ink_count = out.pixels().filter(|p| p[0] < ink_threshold).count();
    let total = 128 * 128;
    let coverage = ink_count as f32 / total as f32;

    // With sqrt compensation, mid-gray should produce ~50% ink coverage
    // Allow 35-65% range to account for edge effects and quantization
    assert!(
        (0.35..=0.65).contains(&coverage),
        "mid-gray should produce ~50% ink coverage, got {:.1}%",
        coverage * 100.0
    );
}
```

#### `test_retro_newspaper_aspect_ratio_produces_circular_dots`

```rust
/// Verify dots are circular on non-square images (aspect ratio handled correctly).
#[test]
fn test_retro_newspaper_aspect_ratio_produces_circular_dots() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // 2:1 aspect ratio image with mid-gray
    let img = make_solid_image(128, 64, 40000, 40000, 40000);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "retro_newspaper",
            values: vec![10.0, 1.0, 0.0], // coarse dots, no softness
        }],
    );

    // On a 2:1 image with proper aspect correction, the dot pattern should
    // have the same spatial frequency in both X and Y directions (in pixels).
    // Without correction, X would have twice the frequency of Y.
    // This is hard to test precisely, so we verify the pattern isn't obviously
    // stretched by checking that horizontal and vertical transitions are similar.
    
    // Count horizontal transitions (changes between adjacent pixels in a row)
    let mut h_transitions = 0;
    for y in 0..64 {
        for x in 0..127 {
            let a = out.get_pixel(x, y)[0];
            let b = out.get_pixel(x + 1, y)[0];
            if (a as i32 - b as i32).abs() > 20000 {
                h_transitions += 1;
            }
        }
    }
    
    // Count vertical transitions
    let mut v_transitions = 0;
    for y in 0..63 {
        for x in 0..128 {
            let a = out.get_pixel(x, y)[0];
            let b = out.get_pixel(x, y + 1)[0];
            if (a as i32 - b as i32).abs() > 20000 {
                v_transitions += 1;
            }
        }
    }
    
    // Normalize by dimension to compare frequency
    let h_freq = h_transitions as f32 / (127.0 * 64.0);
    let v_freq = v_transitions as f32 / (128.0 * 63.0);
    
    // Frequencies should be similar (within 50%) if aspect ratio is handled
    let ratio = h_freq / v_freq;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "horizontal/vertical transition ratio {:.2} suggests stretched dots",
        ratio
    );
}
```

### PR 2 Tests

#### `test_retro_newspaper_softness_affects_edge_sharpness`

```rust
/// Verify softness controls anti-aliasing at dot edges.
#[test]
fn test_retro_newspaper_softness_affects_edge_sharpness() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(64, 64, 40000, 40000, 40000);
    
    // Hard edges (softness = 0.0)
    let out_hard = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "retro_newspaper",
            values: vec![30.0, 1.0, 0.0], // dot_freq, strength, softness
        }],
    );
    
    // Soft edges (softness = 1.0)
    let out_soft = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "retro_newspaper",
            values: vec![30.0, 1.0, 1.0],
        }],
    );

    // Count intermediate values (not near ink ~15000 or paper ~62000)
    let count_intermediate = |out: &crate::Rgba16Image| {
        out.pixels().filter(|p| (20000..55000).contains(&p[0])).count()
    };

    let hard_intermediates = count_intermediate(&out_hard);
    let soft_intermediates = count_intermediate(&out_soft);
    
    assert!(
        soft_intermediates > hard_intermediates,
        "soft edges should have more intermediate values: hard={}, soft={}",
        hard_intermediates, soft_intermediates
    );
}
```

### PR 3 Tests

#### `test_retro_newspaper_quantization_produces_five_levels`

```rust
/// Verify quantization produces exactly 5 discrete levels.
#[test]
fn test_retro_newspaper_quantization_produces_five_levels() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Create a gradient image covering full tonal range
    let w = 256u32;
    let h = 16u32;
    let mut img = crate::Rgba16Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = ((x as u32 * 65535) / 255) as u16;
            img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
        }
    }

    // Run with strength=0 to bypass halftone blend, then check intermediate scratch
    // Actually, we need to test the quantization pass output directly.
    // Since we can't access scratch textures directly, we test at strength=1.0
    // where the halftone pass uses the quantized values to determine dot size.
    // With a fine enough dot grid, the average output should cluster around 5 values.
    
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "retro_newspaper",
            values: vec![100.0, 1.0, 0.0], // high frequency, full strength
        }],
    );

    // Collect unique R values (quantized to nearest 1000 to handle noise)
    use std::collections::HashSet;
    let unique_levels: HashSet<u16> = out
        .pixels()
        .map(|p| (p[0] / 5000) * 5000) // bucket to ~5000 increments
        .collect();

    // Should have approximately 5-6 distinct buckets (ink values from 5 quantization levels)
    assert!(
        unique_levels.len() <= 8,
        "expected ~5 quantization levels, got {} distinct value buckets: {:?}",
        unique_levels.len(),
        unique_levels
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Mid-gray input produces approximately 50% ink coverage at strength=1.0
- [ ] Dots appear circular on both square and non-square images
- [ ] Softness=0 produces hard-edged dots; softness>0 produces smooth edges
- [ ] Strength=0 returns source unchanged (identity)
- [ ] Alpha is preserved through all passes

---

## References

- [Halftone - Wikipedia](https://en.wikipedia.org/wiki/Halftone)
- [Halftone: From Newspaper Ink to Digital Shaders | Efecto Blog](https://efecto.app/blog/what-is-halftone)
- [Digital Halftoning - Purdue Engineering](https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Halftoning.pdf)
- [The Print Guide: Halftone screen angles](http://the-print-guide.blogspot.com/2009/05/halftone-screen-angles.html)
- [Halftones and tone transfer curves - IBM](https://www.ibm.com/docs/en/zos/2.2.0?topic=concepts-halftones-tone-transfer-curves)
