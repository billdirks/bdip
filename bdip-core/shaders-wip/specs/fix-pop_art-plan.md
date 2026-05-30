# Fix Pop Art Transform

## Problem Summary

The pop art transform produces a functional effect but has algorithmic issues in the halftone
dot overlay that reduce visual fidelity.

### Moderate Issues

1. **Incorrect paper simulation formula** (`combine.wgsl:42`)
   - Current: `let paper = colorize.rgb * 0.5 + 0.5;`
   - This shifts colors toward 0.75 (midpoint between color and white), not toward white
   - For dark colors (0.2): paper = 0.6 (lighter)
   - For bright colors (0.8): paper = 0.9 (lighter, but asymmetric)
   - Authentic silkscreen printing shows white paper through ink gaps
   - **Fix**: Use `let paper = mix(colorize.rgb, vec3<f32>(1.0), 0.5);` or simply
     `vec3<f32>(1.0)` for pure white paper

2. **Hard-edged dots cause aliasing** (`combine.wgsl:38`)
   - Current: `let dot = step(dist, 0.35);`
   - Produces binary 0/1 transitions with visible stair-stepping on diagonal edges
   - **Fix**: Use `smoothstep()` with a softness parameter for anti-aliasing

### Minor Issues

3. **HSL conversion in linear light space** (`colorize.wgsl:35-39`)
   - HSL is conventionally defined for gamma-corrected sRGB, not linear RGB
   - Lightness ramps will appear differently than expected from standard HSL tools
   - This matches the pattern noted in `specs/tech_debt.md` for the cartoon shader
   - **Assessment**: Acceptable for a stylized effect; not a correctness bug

4. **No dot grid rotation parameter** (`combine.wgsl:36`)
   - Traditional halftone uses 45° rotation to minimize visible grid lines
   - Axis-aligned grids are more perceptually prominent
   - **Assessment**: Enhancement opportunity, not a defect

### Design Choices (Not Issues)

- **Fixed dot size**: The uniform Ben-Day dot style (vs. variable-size halftone) is
  appropriate for pop art. Roy Lichtenstein's work used uniform Ben-Day dots, making
  this a valid artistic choice.

- **Default strength = 0.0**: Consistent with other transforms; provides identity behavior.

---

## Implementation Plan

### PR 1: Fix Paper Simulation and Add Anti-Aliasing

**Goal**: Correct the paper color formula and add smooth dot edges.

**Scope**:
- `bdip_core/src/gpu/shaders/pop_art/combine.wgsl` — fix paper formula, add smoothstep
- `bdip_core/src/gpu/shaders/pop_art/mod.rs` — add `softness` parameter

**Changes to `combine.wgsl`**:

```wgsl
// Current (line 42):
let paper  = colorize.rgb * 0.5 + 0.5;

// Fixed — mix toward white to simulate paper showing through:
let paper = mix(colorize.rgb, vec3<f32>(1.0), 0.6);

// Current (line 38):
let dot  = step(dist, 0.35);

// Fixed — anti-aliased edge based on softness:
let edge_width = params.softness * 0.1;
let dot = 1.0 - smoothstep(0.35 - edge_width, 0.35 + edge_width, dist);
```

**Changes to `mod.rs`**:

Add softness parameter to `PopArtParams`:

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PopArtParams {
    pub strength: f32,
    pub levels: f32,
    pub dot_scale: f32,
    pub softness: f32,  // replaces _padding
}
```

Add slider definition:

```rust
SliderDef {
    name: "Softness",
    min: 0.0,
    max: 1.0,
    default: 0.3,
    description: "Dot edge softness. 0.0 produces hard edges; higher values add anti-aliasing.",
},
```

Update `from_values`:

```rust
fn from_values(values: &[f32]) -> Self {
    Self {
        strength: values[0],
        levels: values[1],
        dot_scale: values[2],
        softness: values[3],
    }
}
```

Update all three WGSL files to include `softness` in the `PopArtParams` struct (required for
uniform buffer validation even if not used by that pass).

**Tests to Update**:
- `test_pop_art_make_uniform_known_value` — add softness parameter to test values
- `test_pop_art_zero_strength_is_identity` — add softness to transform values
- `test_pop_art_full_strength_reduces_unique_color_values` — add softness
- `test_pop_art_more_levels_produces_more_unique_values` — add softness
- `test_pop_art_larger_dot_scale_changes_pattern` — add softness
- `test_pop_art_alpha_preserved` — add softness
- `test_pop_art_deterministic` — add softness
- `test_pop_art_registry_metadata` — update expected slider count and definitions

**Tests to Add**:

1. `test_pop_art_softness_affects_edge_sharpness`
2. `test_pop_art_paper_is_lighter_than_ink`

---

### PR 2 (Optional Enhancement): Add Dot Grid Angle Parameter

**Goal**: Allow rotation of the halftone dot grid.

**Scope**:
- `bdip_core/src/gpu/shaders/pop_art/combine.wgsl` — add rotation math
- `bdip_core/src/gpu/shaders/pop_art/mod.rs` — add angle parameter

This PR is optional and should only be implemented if users request the feature. The
axis-aligned grid is acceptable for basic pop art.

**Changes to `combine.wgsl`**:

```wgsl
// Add rotation before computing cell position:
let angle_rad = params.angle * 0.01745329251;  // degrees to radians
let cos_a = cos(angle_rad);
let sin_a = sin(angle_rad);
let pos = vec2<f32>(gid.xy);
let rot_x = pos.x * cos_a - pos.y * sin_a;
let rot_y = pos.x * sin_a + pos.y * cos_a;
let cell_frac = fract(vec2<f32>(rot_x, rot_y) / params.dot_scale) - 0.5;
```

**New Slider**:

```rust
SliderDef {
    name: "Angle",
    min: 0.0,
    max: 90.0,
    default: 0.0,
    description: "Dot grid rotation in degrees. 45° reduces visible grid lines.",
},
```

**Note**: Adding this parameter requires updating the struct to 6 floats (24 bytes), which
changes alignment. May need additional padding.

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_pop_art_softness_affects_edge_sharpness`

```rust
/// Verify softness controls anti-aliasing at dot edges.
#[test]
fn test_pop_art_softness_affects_edge_sharpness() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Gradient image to ensure dots at various coverage levels
    let mut img = crate::Rgba16Image::new(64, 64);
    for (i, pixel) in img.pixels_mut().enumerate() {
        let v = ((i as u32 * 65535) / (64 * 64)) as u16;
        *pixel = image::Rgba([v, v, v, 65535]);
    }

    let out_hard = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "pop_art",
            values: vec![1.0, 4.0, 12.0, 0.0], // softness=0
        }],
    );

    let out_soft = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "pop_art",
            values: vec![1.0, 4.0, 12.0, 1.0], // softness=1
        }],
    );

    // Count pixels with intermediate luminance (not near extremes)
    let count_intermediate = |img: &crate::Rgba16Image| {
        img.pixels()
            .filter(|p| {
                let lum = (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3;
                (10000..55000).contains(&lum)
            })
            .count()
    };

    let hard_intermediates = count_intermediate(&out_hard);
    let soft_intermediates = count_intermediate(&out_soft);

    assert!(
        soft_intermediates > hard_intermediates,
        "soft edges should have more intermediate values: hard={}, soft={}",
        hard_intermediates,
        soft_intermediates
    );
}
```

#### `test_pop_art_paper_is_lighter_than_ink`

```rust
/// Verify paper regions (between dots) are lighter than ink regions (dots).
#[test]
fn test_pop_art_paper_is_lighter_than_ink() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Mid-gray input produces visible dots with gaps
    let img = make_solid_image(64, 64, 32767, 32767, 32767);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "pop_art",
            values: vec![1.0, 4.0, 16.0, 0.0], // large dots for clear measurement
        }],
    );

    // Sample center of a cell (should be ink/dot) vs edge (should be paper)
    // With dot_scale=16, cells are 16x16. Center at (8,8), edge at (0,8).
    let center_pixel = out.get_pixel(8, 8);
    let edge_pixel = out.get_pixel(0, 8);

    let center_lum = (center_pixel[0] as u32 + center_pixel[1] as u32 + center_pixel[2] as u32) / 3;
    let edge_lum = (edge_pixel[0] as u32 + edge_pixel[1] as u32 + edge_pixel[2] as u32) / 3;

    assert!(
        edge_lum > center_lum,
        "paper (edge) should be lighter than ink (center): edge_lum={}, center_lum={}",
        edge_lum,
        center_lum
    );
}
```

---

## Validation Checklist

After PR 1 is merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (all existing + new tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Visual inspection: dots have smooth edges at softness > 0
- [ ] Visual inspection: paper regions between dots appear white/light, not tinted
- [ ] Strength=0 returns source unchanged (identity)
- [ ] Alpha channel preserved

---

## References

- [Halftone - Wikipedia](https://en.wikipedia.org/wiki/Halftone)
- [Ben-Day Dots - Arts Award Initiative](https://www.artsawardinitiative.co.uk/resources/ai_educationals/leger/Ben%20Day%20dots.pdf)
- [Halftone Shader - Ben Simonds](https://bensimonds.com/2013/02/14/halftone-shader/)
- [Posterization - Wikipedia](https://en.wikipedia.org/wiki/Posterization)
- [HSL and HSV - Wikipedia](https://en.wikipedia.org/wiki/HSL_and_HSV)
