# Fix Pointillism Transform

## Problem Summary

The pointillism transform has a fundamentally correct algorithm (sampling colors at grid centers,
rendering circular dots on white background) but lacks polish and authenticity features that
distinguish the effect from a simple "dot matrix" pattern.

### Critical Issues

1. **No anti-aliasing** (`pointillism_dots.wgsl:59-63`): Uses hard `if dist <= dot_radius` boundary
   instead of `smoothstep()`. This produces jagged/aliased edges on dots, particularly visible at
   larger dot sizes or when zoomed in.

### Moderate Issues

2. **No position jitter**: Dots are placed on a perfectly uniform grid, which looks mechanical
   rather than painterly. Authentic pointillism involves hand-placed dots with natural positional
   variation. Georges Seurat's technique relied on seemingly random dot placement that creates an
   organic, hand-painted look. The current rigid grid produces a clinical appearance more akin to
   pop-art Ben-Day dots than neo-impressionist pointillism.

3. **No size variation**: All dots are exactly the same size. Real pointillism has subtle size
   differences based on pressure, paint viscosity, and artistic intent. Digital implementations
   typically add 5-15% size jitter for authenticity.

### Minor Issues

4. **No softness parameter**: Users have no control over edge anti-aliasing width. Some use cases
   benefit from hard edges (stylized look), while others need soft edges (smooth appearance).

### Current Parameters

| Parameter | Range       | Issue |
|-----------|-------------|-------|
| Strength  | 0.0–1.0     | OK    |
| Grid Size | 4.0–64.0    | OK    |
| Dot Size  | 0.1–1.0     | OK    |

### Missing Parameters

- **Jitter** (position randomness, typically 0.0–0.5 as fraction of grid cell)
- **Size Variation** (random size multiplier, typically 0.0–0.3)
- **Softness** (anti-aliasing edge width, 0.0–1.0)

---

## Implementation Plan

### PR 1: Add Anti-Aliasing with Softness Parameter

**Goal**: Replace hard if/else boundary with smoothstep() for anti-aliased dot edges.

**Scope**:
- Modify `pointillism_dots.wgsl` to use smoothstep() for dot boundary
- Add `softness` parameter to `PointillismParams` struct
- Add Softness slider to `mod.rs`
- Update existing tests if needed

**New Parameter**:

```rust
pub struct PointillismParams {
    pub strength: f32,
    pub grid_size: f32,
    pub dot_size: f32,
    pub softness: f32,  // NEW: 0.0–1.0, default 0.2
}
```

**Shader Changes** (`pointillism_dots.wgsl`):

Replace lines 58-64:
```wgsl
// Current hard boundary:
var pointillist_rgb: vec3<f32>;
if dist <= dot_radius {
    pointillist_rgb = cell.rgb;
} else {
    pointillist_rgb = vec3<f32>(1.0, 1.0, 1.0);
}
```

With smoothstep:
```wgsl
// Anti-aliased boundary using smoothstep
// softness controls the transition width as a fraction of the dot radius
let edge_width = params.softness * dot_radius * 0.5;
let t = smoothstep(dot_radius - edge_width, dot_radius + edge_width, dist);
let pointillist_rgb = mix(cell.rgb, vec3<f32>(1.0, 1.0, 1.0), t);
```

**Tests to Add**:

1. `test_pointillism_softness_affects_edge_sharpness`: At softness=0.0, output should have mostly
   bimodal values (dot color or white). At softness=1.0, edges should have intermediate values.

2. `test_pointillism_softness_zero_matches_hard_edge`: Verify softness=0.0 produces equivalent
   output to the current hard-edge implementation (within tolerance).

---

### PR 2: Add Position Jitter for Organic Look

**Goal**: Add per-dot positional randomness to simulate hand-placed dots.

**Scope**:
- Add `jitter` parameter to `PointillismParams` struct
- Add Jitter slider to `mod.rs`
- Modify both shader passes to apply consistent jitter to cell centers
- Use deterministic hash-based noise (same jitter for same cell across passes)

**New Parameter**:

```rust
pub struct PointillismParams {
    pub strength: f32,
    pub grid_size: f32,
    pub dot_size: f32,
    pub softness: f32,
    pub jitter: f32,  // NEW: 0.0–0.5, default 0.15
}
```

**Algorithm**:

Both passes need consistent jitter so quantize samples the jittered center and dots renders at
the same jittered center:

```wgsl
// Hash function for deterministic per-cell randomness
fn hash2(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(
        dot(p, vec2<f32>(127.1, 311.7)),
        dot(p, vec2<f32>(269.5, 183.3))
    );
    return fract(sin(q) * 43758.5453);
}

// In main():
let cell_id = vec2<f32>(cell_col, cell_row);
let noise = hash2(cell_id) * 2.0 - 1.0;  // Range [-1, 1]
let jitter_offset = noise * params.jitter * gs * 0.5;
let centre = vec2<f32>((cell_col + 0.5) * gs, (cell_row + 0.5) * gs) + jitter_offset;
```

**Tests to Add**:

1. `test_pointillism_jitter_zero_matches_uniform_grid`: With jitter=0.0, output should match the
   non-jittered implementation.

2. `test_pointillism_jitter_produces_different_output`: With jitter=0.3, output should differ from
   jitter=0.0 on a gradient image.

3. `test_pointillism_jitter_is_deterministic`: Two runs with identical inputs and jitter must
   produce bit-identical output (hash-based randomness is deterministic).

---

### PR 3: Add Size Variation

**Goal**: Add per-dot size variation for additional organic appearance.

**Scope**:
- Add `size_variation` parameter to `PointillismParams`
- Add Size Variation slider to `mod.rs`
- Modify `pointillism_dots.wgsl` to vary dot radius per-cell

**New Parameter**:

```rust
pub struct PointillismParams {
    pub strength: f32,
    pub grid_size: f32,
    pub dot_size: f32,
    pub softness: f32,
    pub jitter: f32,
    pub size_variation: f32,  // NEW: 0.0–0.3, default 0.1
}
```

**Algorithm** (dots pass only):

```wgsl
// Per-cell size variation using the same hash
let size_noise = hash2(cell_id + vec2<f32>(100.0, 100.0)).x;  // Offset to get different noise
let size_mult = 1.0 + (size_noise * 2.0 - 1.0) * params.size_variation;
let dot_radius = params.dot_size * (gs * 0.5) * size_mult;
```

**Tests to Add**:

1. `test_pointillism_size_variation_zero_uniform`: With size_variation=0.0, all dots should have
   identical radius.

2. `test_pointillism_size_variation_produces_different_coverage`: With size_variation=0.2, the
   total colored pixel count should differ from size_variation=0.0.

---

### PR 4: Documentation and Parameter Descriptions

**Goal**: Update all slider descriptions to explain the new parameters clearly.

**Scope**:
- Update slider descriptions in `mod.rs`
- Update `DESCRIPTION` to reflect the enhanced effect

**Updated Sliders**:

```rust
const PARAM: ParamKind = ParamKind::Sliders(&[
    SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend between original image (0.0) and pointillism effect (1.0).",
    },
    SliderDef {
        name: "Grid Size",
        min: 4.0,
        max: 64.0,
        default: 16.0,
        description: "Spacing between dot centres in pixels. Larger values produce bigger, \
                      more widely spaced dots.",
    },
    SliderDef {
        name: "Dot Size",
        min: 0.1,
        max: 1.0,
        default: 0.8,
        description: "Dot radius as a fraction of the grid cell. 1.0 fills the cell; \
                      smaller values leave white gaps between dots.",
    },
    SliderDef {
        name: "Softness",
        min: 0.0,
        max: 1.0,
        default: 0.2,
        description: "Dot edge softness. 0.0 produces hard edges; higher values add \
                      anti-aliasing for a smoother appearance.",
    },
    SliderDef {
        name: "Jitter",
        min: 0.0,
        max: 0.5,
        default: 0.15,
        description: "Random offset of dot positions. 0.0 places dots on a uniform grid; \
                      higher values create an organic, hand-painted look.",
    },
    SliderDef {
        name: "Size Variation",
        min: 0.0,
        max: 0.3,
        default: 0.1,
        description: "Random variation in dot sizes. 0.0 produces uniform dots; higher \
                      values simulate natural brush pressure variation.",
    },
]);
```

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_pointillism_softness_affects_edge_sharpness`

```rust
/// Verify softness controls anti-aliasing at dot edges.
#[test]
fn test_pointillism_softness_affects_edge_sharpness() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(64, 64, 30000, 30000, 30000);
    
    // Hard edges: strength=1.0, grid=16, dot=0.6, softness=0.0
    let out_hard = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "pointillism",
            values: vec![1.0, 16.0, 0.6, 0.0],
        }],
    );
    
    // Soft edges: softness=1.0
    let out_soft = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "pointillism",
            values: vec![1.0, 16.0, 0.6, 1.0],
        }],
    );

    // Count intermediate values (neither near-black/source nor near-white)
    let count_intermediate = |img: &crate::Rgba16Image| {
        img.pixels()
            .filter(|p| (20000..55000).contains(&p[0]))
            .count()
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

#### `test_pointillism_softness_zero_produces_hard_edges`

```rust
/// Verify softness=0.0 produces hard-edged dots (bimodal distribution).
#[test]
fn test_pointillism_softness_zero_produces_hard_edges() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(64, 64, 30000, 30000, 30000);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "pointillism",
            values: vec![1.0, 16.0, 0.6, 0.0], // softness=0
        }],
    );

    // With hard edges, pixels should be either dot color (~30000) or white (~65535)
    // Very few should be in between
    let intermediate_count = out.pixels()
        .filter(|p| (35000..60000).contains(&p[0]))
        .count();
    let total = 64 * 64;
    let intermediate_ratio = intermediate_count as f32 / total as f32;
    
    assert!(
        intermediate_ratio < 0.05,
        "hard edges should have <5% intermediate values, got {:.1}%",
        intermediate_ratio * 100.0
    );
}
```

### PR 2 Tests (Detailed)

#### `test_pointillism_jitter_produces_different_output`

```rust
/// Verify jitter parameter changes dot positions.
#[test]
fn test_pointillism_jitter_produces_different_output() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Gradient image so different positions sample different colors
    let mut img = crate::Rgba16Image::new(64, 64);
    for y in 0..64u32 {
        for x in 0..64u32 {
            let v = (x * 1000) as u16;
            img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
        }
    }

    let out_no_jitter = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "pointillism",
            values: vec![1.0, 16.0, 0.8, 0.2, 0.0], // jitter=0
        }],
    );

    let out_with_jitter = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "pointillism",
            values: vec![1.0, 16.0, 0.8, 0.2, 0.3], // jitter=0.3
        }],
    );

    let any_different = out_no_jitter
        .pixels()
        .zip(out_with_jitter.pixels())
        .any(|(a, b)| (a[0] as i32 - b[0] as i32).abs() > 100);

    assert!(any_different, "jitter=0.3 must produce different output than jitter=0.0");
}
```

#### `test_pointillism_jitter_is_deterministic`

```rust
/// Verify jitter uses deterministic hash-based randomness.
#[test]
fn test_pointillism_jitter_is_deterministic() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(64, 64, 32767, 32767, 32767);
    let transform = Transform {
        shader_id: "pointillism",
        values: vec![1.0, 16.0, 0.8, 0.2, 0.25], // jitter=0.25
    };

    let out1 = roundtrip(&mut renderer, &engine, &img, std::slice::from_ref(&transform));
    let out2 = roundtrip(&mut renderer, &engine, &img, std::slice::from_ref(&transform));

    for (p1, p2) in out1.pixels().zip(out2.pixels()) {
        assert_eq!(p1, p2, "jittered outputs must be identical across runs");
    }
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Softness=0 produces hard-edged dots; softness=1 produces soft anti-aliased edges
- [ ] Jitter=0 produces uniform grid; jitter=0.3 produces organic, scattered dots
- [ ] Size variation=0 produces uniform dots; size variation=0.2 produces varied dot sizes
- [ ] Strength=0 returns source unchanged (identity)
- [ ] White gaps appear between dots when dot_size < 1.0
- [ ] Effect looks more "painterly" and less "mechanical" with default jitter/variation settings

---

## References

- [Pointillism - Wikipedia](https://en.wikipedia.org/wiki/Pointillism)
- [Divisionism - Wikipedia](https://en.wikipedia.org/wiki/Divisionism)
- [How Pointillism Exploits Optical Mixing](https://www.beyondeveryart.com/pointillism-optical-mixing-vs-pigment-mixing/)
- [Pointillist Techniques for Painting and Drawing](https://www.jacksonsart.com/blog/2025/09/09/pointillist-techniques-for-painting-and-drawing/)
- [Anti-Aliasing Basics for Procedural Shapes](https://shadergif.com/guides/anti-aliasing-basics/)
- [Samuel Karabetian - Pointillism Shader](https://samuelkarabetian.com/pointillism-shader/)
- [Stanford - Create Pointillism Art from Digital Images](https://web.stanford.edu/class/ee368/Project_Autumn_1516/Reports/Hong_Liu.pdf)
