# Fix Halftone Dots Transform

## Problem Summary

The current halftone_dots transform has fundamental algorithm issues that prevent it from producing
authentic halftone output.

### Critical Issues

1. **Wrong algorithm**: Uses sine-wave product (`sin(x) * sin(y)`) instead of distance-to-cell-center.
   This produces diamond/rectangular shapes, not the characteristic circular dots of halftone
   printing.

2. **No angle parameter**: Traditional halftone grids are rotated 45° to minimize visibility. The
   human visual system is more sensitive to horizontal/vertical patterns.

3. **Unused asset by this shader**: `halftone_dots.png` is registered but not used by this shader.
   The shader computes halftone via sine waves instead. (Note: the asset may be used elsewhere.)

4. **No geometric radius compensation** (affects proposed fix): When switching to distance-based
   circular dots, the radius must use sqrt() because dot area scales with radius². Note: the
   current sine-wave approach's linear threshold is actually reasonable for tone response — this
   issue only applies to the distance-based replacement algorithm.

5. **No anti-aliasing**: Uses hard `select()` instead of `smoothstep()`.

### Current Parameters

| Parameter | Range     | Issue |
|-----------|-----------|-------|
| Strength  | 0.0–1.0   | OK    |
| Frequency | 0.01–0.5  | OK but unintuitive units |

### Missing Parameters

- **Angle** (rotation of dot grid, typically 45°)
- **Softness** (anti-aliasing control)

---

## Implementation Plan

### PR 1: Rewrite Core Algorithm with Circular Dots

**Goal**: Replace sine-wave threshold with proper distance-to-cell-center algorithm producing
circular dots.

**Scope**:
- Rewrite `halftone_dots.wgsl` to use Euclidean distance from cell center
- Add angle parameter for grid rotation (default 45°)
- Add softness parameter for anti-aliasing via smoothstep
- Apply sqrt() geometric compensation to luminance-to-radius mapping (area = πr²)
- Update `HalftoneDotParams` struct with new fields
- Update `mod.rs` with new slider definitions

**New Parameters**:

```rust
pub struct HalftoneDotParams {
    pub strength: f32,   // 0.0–1.0, default 0.0 (identity)
    pub frequency: f32,  // 0.01–0.5, default 0.1 (cycles per pixel)
    pub angle: f32,      // 0.0–180.0, default 45.0 (degrees)
    pub softness: f32,   // 0.0–1.0, default 0.1 (anti-aliasing width)
}
```

**New Shader Algorithm** (pseudocode):

```wgsl
// 1. Rotate coordinates by angle
let angle_rad = params.angle * TAU / 360.0;
let cos_a = cos(angle_rad);
let sin_a = sin(angle_rad);
let rot_x = fx * cos_a - fy * sin_a;
let rot_y = fx * sin_a + fy * cos_a;

// 2. Get position within cell [0, 1]
let cell_size = 1.0 / params.frequency;
let cell_x = fract(rot_x / cell_size);
let cell_y = fract(rot_y / cell_size);

// 3. Distance from cell center (0.5, 0.5), normalized to [0, ~0.707]
let dist = length(vec2<f32>(cell_x, cell_y) - vec2<f32>(0.5, 0.5));

// 4. Radius from luminance with sqrt geometric compensation (area = πr²)
// Dark pixels (lum=0) -> large radius (0.5) -> mostly black output
// Bright pixels (lum=1) -> small radius (0) -> mostly white output
let radius = 0.5 * sqrt(1.0 - lum);

// 5. Anti-aliased dot: 1.0 inside dot (black), 0.0 outside (white)
// Invert because halftone dots are printed ink (black) on white paper
let edge_width = params.softness * 0.1;
let dot_mask = 1.0 - smoothstep(radius - edge_width, radius + edge_width, dist);

// 6. Output: dot_mask=1 means inside dot (black), dot_mask=0 means outside (white)
let halftone_value = 1.0 - dot_mask;
```

**Tests to Add**:

1. `test_halftone_dots_produces_circular_pattern`: On a 50% gray input, verify the output has
   roughly circular dot shapes by checking that pixels equidistant from cell centers have similar
   values.

2. `test_halftone_dots_angle_rotates_grid`: Compare output at angle=0° vs angle=45° on a uniform
   gray image — the patterns should differ in a predictable way.

3. `test_halftone_dots_softness_affects_edge_sharpness`: At softness=0.0, edges should be hard
   (bimodal distribution). At softness=1.0, there should be intermediate gray values at dot edges.

4. `test_halftone_dots_geometric_compensation`: Verify that mid-gray input (0.5 luminance) produces
   dots covering approximately 50% of area (sqrt compensates for area = πr²).

**Existing Tests to Preserve** (update expected behavior if needed):
- `test_halftone_dots_zero_strength_is_identity`
- `test_halftone_dots_alpha_preserved`
- `test_halftone_dots_full_strength_produces_binary_output` (may need softness=0 for binary)
- `test_halftone_dots_white_input_produces_white_output`
- `test_halftone_dots_black_input_produces_black_output`
- `test_halftone_dots_different_frequencies_produce_different_output`
- `test_halftone_dots_chains_with_brightness`

---

### PR 2: Documentation and Parameter Help

**Goal**: Ensure user-facing documentation explains the halftone effect clearly.

**Scope**:
- Update `DESCRIPTION` in `HalftoneDotParams` to explain the effect
- Ensure all slider descriptions explain the visual impact
- Add parameter help following project conventions (see `specs/parameter-help.md` if exists)

**Slider Descriptions**:

```rust
SliderDef {
    name: "Strength",
    min: 0.0,
    max: 1.0,
    default: 0.0,
    description: "Blend between original image (0.0) and halftone effect (1.0).",
},
SliderDef {
    name: "Frequency",
    min: 0.01,
    max: 0.5,
    default: 0.1,
    description: "Dot density in cycles per pixel. Higher values produce smaller, denser dots.",
},
SliderDef {
    name: "Angle",
    min: 0.0,
    max: 180.0,
    default: 45.0,
    description: "Grid rotation in degrees. 45° minimizes visible grid lines.",
},
SliderDef {
    name: "Softness",
    min: 0.0,
    max: 1.0,
    default: 0.1,
    description: "Dot edge softness. 0.0 produces hard edges; higher values add anti-aliasing.",
},
```

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_halftone_dots_produces_circular_pattern`

```rust
/// Verify dots are circular by checking that pixels at the same distance from
/// cell centers have similar output values on a uniform gray input.
#[test]
fn test_halftone_dots_produces_circular_pattern() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Uniform 50% gray, large enough to contain multiple complete cells
    let img = make_solid_image(128, 128, 32767, 32767, 32767);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "halftone_dots",
            values: vec![1.0, 0.1, 0.0, 0.0], // strength, freq, angle=0, softness=0
        }],
    );

    // With angle=0 and frequency=0.1, cell size is 10 pixels.
    // Check that (5,5) and (5,15) are both at cell centers and have same value.
    // Check that (0,5) and (5,0) are both at cell edges and have same value.
    let center1 = out.get_pixel(5, 5)[0];
    let center2 = out.get_pixel(15, 5)[0];
    assert!(
        (center1 as i32 - center2 as i32).abs() < 1000,
        "cell centers should have similar values: {} vs {}",
        center1, center2
    );
}
```

#### `test_halftone_dots_angle_rotates_grid`

```rust
/// Verify the angle parameter rotates the dot grid.
#[test]
fn test_halftone_dots_angle_rotates_grid() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(64, 64, 32767, 32767, 32767);
    
    let out_0deg = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "halftone_dots",
            values: vec![1.0, 0.1, 0.0, 0.0], // angle=0
        }],
    );
    
    let out_45deg = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "halftone_dots",
            values: vec![1.0, 0.1, 45.0, 0.0], // angle=45
        }],
    );

    // The patterns must differ
    let differs = out_0deg.pixels().zip(out_45deg.pixels())
        .any(|(a, b)| (a[0] as i32 - b[0] as i32).abs() > 1000);
    assert!(differs, "angle=0 and angle=45 must produce different patterns");
}
```

#### `test_halftone_dots_softness_affects_edge_sharpness`

```rust
/// Verify softness controls anti-aliasing at dot edges.
#[test]
fn test_halftone_dots_softness_affects_edge_sharpness() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(64, 64, 32767, 32767, 32767);
    
    let out_hard = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "halftone_dots",
            values: vec![1.0, 0.1, 45.0, 0.0], // softness=0
        }],
    );
    
    let out_soft = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "halftone_dots",
            values: vec![1.0, 0.1, 45.0, 1.0], // softness=1
        }],
    );

    // Count intermediate values (not near black or white)
    let count_intermediate = |img: &Rgba16Image| {
        img.pixels().filter(|p| (5000..60000).contains(&p[0])).count()
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

#### `test_halftone_dots_geometric_compensation`

```rust
/// Verify sqrt() geometric compensation: 50% gray should produce ~50% dot coverage
/// because radius = sqrt(1 - lum) compensates for area = πr².
#[test]
fn test_halftone_dots_geometric_compensation() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // 50% linear gray
    let img = make_solid_image(128, 128, 32767, 32767, 32767);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "halftone_dots",
            values: vec![1.0, 0.05, 45.0, 0.0], // coarse dots for clear measurement
        }],
    );

    // Count black vs white pixels
    let total = 128 * 128;
    let black_count = out.pixels().filter(|p| p[0] < 16000).count();
    let coverage = black_count as f32 / total as f32;

    // With sqrt geometric compensation (area = πr²), 50% gray should produce ~50% coverage
    // Allow 35-65% range to account for edge effects
    assert!(
        (0.35..=0.65).contains(&coverage),
        "50% gray should produce ~50% dot coverage, got {:.1}%",
        coverage * 100.0
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Halftone effect produces visible circular dots at default settings
- [ ] Angle=45° produces a diagonal dot pattern
- [ ] Softness=0 produces hard-edged dots; softness=1 produces soft edges
- [ ] White input → white output; black input → black output
- [ ] Strength=0 returns source unchanged (identity)

---

## References

- [Halftone - Wikipedia](https://en.wikipedia.org/wiki/Halftone)
- [Digital Halftoning - Purdue Engineering](https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Halftoning.pdf)
- [Halftone Shader - Maxime Heckel](https://blog.maximeheckel.com/posts/shades-of-halftone/)
