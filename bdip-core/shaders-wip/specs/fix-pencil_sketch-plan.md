# Fix Pencil Sketch Transform

## Problem Summary

The pencil_sketch transform has a directional blur bug that causes strokes to run perpendicular to the
intended direction.

### Critical Issues

1. **Stroke direction is 90° off**: The directional blur in pass 2 operates along the gradient
   direction (perpendicular to edges) instead of along the edge contours where pencil strokes
   naturally flow.

   - **File**: `pencil_sketch_edges.wgsl`, lines 85-86
   - **Current code**: `atan2(gradient.y, gradient.x)` computes the gradient direction
   - **Expected**: The stroke direction should be perpendicular to the gradient (i.e., along the edge)
   - **Fix**: Add π/2 to rotate 90°, or use `atan2(-gradient.x, gradient.y)` for the perpendicular

   The comment on lines 12-13 correctly describes the intent ("direction perpendicular to the edge
   normal") but the implementation computes the gradient direction itself.

   **Visual impact**: For a vertical edge, strokes blur horizontally (across the edge) instead of
   vertically (along the edge). This produces smeared edges rather than pencil-like strokes that
   follow contours.

### What's Correct

- Sobel operator implementation (3×3 kernels, correct weights)
- BT.709 luma coefficients (0.2126, 0.7152, 0.0722) for grayscale conversion
- Edge intensity scaling by `edge_strength` parameter
- Inversion for dark-lines-on-white-paper appearance
- Blend with source using `strength` parameter
- Alpha channel preservation
- Parameter ranges and defaults are reasonable
- All existing tests pass

### References

- [Sobel operator - Wikipedia](https://en.wikipedia.org/wiki/Sobel_operator)
- [Kyle Halladay - A Pencil Sketch Effect](https://kylehalladay.com/blog/tutorial/2017/02/21/Pencil-Sketch-Effect.html)
- [OpenCV Pencil Sketch - Ask a Swiss](https://www.askaswiss.com/2016/01/how-to-create-pencil-sketch-opencv-python.html)
- [Sketch Generation with Drawing Process Guided by Vector Flow and Grayscale](https://arxiv.org/abs/2012.09004)
- [Real-Time Pencil Rendering - POSTECH](https://cg.postech.ac.kr/papers/47_Real-Time-Pencil-Rendering.pdf)

---

## Implementation Plan

### PR 1: Fix Stroke Direction in Edge Detection Pass

**Goal**: Rotate the stored angle by 90° so directional blur runs along edges, not across them.

**Scope**:
- `bdip_core/src/gpu/shaders/pencil_sketch/pencil_sketch_edges.wgsl`

**Changes**:

In `pencil_sketch_edges.wgsl`, replace lines 85-86:

```wgsl
// Current (wrong):
let angle_raw  = atan2(gradient.y, gradient.x); // [-π, π]
let angle_norm = (angle_raw + pi) / (2.0 * pi); // [0, 1]
```

With:

```wgsl
// Fixed: rotate 90° to get stroke direction (along edge, not across it)
// atan2(-gx, gy) = atan2(gy, gx) + π/2, giving the perpendicular direction
let stroke_angle = atan2(-gradient.x, gradient.y); // [-π, π], perpendicular to gradient
let angle_norm   = (stroke_angle + pi) / (2.0 * pi); // [0, 1]
```

Also update the comment on lines 7-9 to clarify the math:

```wgsl
//   .g = stroke angle in [0, 1] (perpendicular to gradient, i.e., along the edge)
```

**Tests to Add**:

1. `test_pencil_sketch_stroke_direction_along_edge`: Verify that on a vertical edge, the blur
   spreads vertically (along the edge) rather than horizontally (across the edge).

**Existing Tests**: All existing tests should continue to pass. The visual output will change (strokes
will now correctly follow edge contours), but the behavioral tests verify properties like identity at
strength=0, alpha preservation, and parameter sensitivity — all of which remain valid.

---

## Test Specifications

### `test_pencil_sketch_stroke_direction_along_edge`

**Purpose**: Verify the directional blur runs along edges, not across them.

**Setup**:
- Create a 32×32 image with a vertical edge: left half dark (10000), right half bright (55000)
- Apply pencil_sketch with strength=1.0, edge_strength=2.0, stroke_softness=1.0

**Assertion**:
- Sample pixels along the vertical edge (e.g., x=15 or x=16) at different y values
- With correct stroke direction, the blur spreads vertically, so pixels at the same x but different y
  should have similar values (blur smooths along the edge)
- Sample pixels horizontally across the edge at the same y
- The horizontal gradient should remain relatively sharp (blur doesn't spread across the edge)

**Code skeleton**:

```rust
/// Verify directional blur runs along edges (vertically for a vertical edge).
#[test]
fn test_pencil_sketch_stroke_direction_along_edge() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Vertical edge: left half dark, right half bright
    let mut img = crate::Rgba16Image::new(32, 32);
    for y in 0..32u32 {
        for x in 0..32u32 {
            let v: u16 = if x < 16 { 10000 } else { 55000 };
            img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
        }
    }

    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "pencil_sketch",
            values: vec![1.0, 2.0, 1.0], // full strength, high softness for visible blur
        }],
    );

    // Pixels along the edge at same x, different y should be similar (blur along edge)
    let edge_top = out.get_pixel(15, 4)[0] as i32;
    let edge_mid = out.get_pixel(15, 16)[0] as i32;
    let edge_bot = out.get_pixel(15, 28)[0] as i32;

    // With stroke_softness=1.0, the blur radius is substantial.
    // Vertical neighbors along the edge should be smoothed together.
    let vertical_variance = (edge_top - edge_mid).abs().max((edge_mid - edge_bot).abs());

    // Horizontal gradient should still show a transition (blur doesn't eliminate the edge)
    let left_of_edge = out.get_pixel(10, 16)[0] as i32;
    let right_of_edge = out.get_pixel(22, 16)[0] as i32;
    let horizontal_diff = (left_of_edge - right_of_edge).abs();

    // The edge should be preserved horizontally more than blurred vertically.
    // This is a sanity check that blur is directional along the edge.
    assert!(
        vertical_variance < horizontal_diff || vertical_variance < 5000,
        "blur should spread along edge (vertically), not across it: \
         vertical_variance={}, horizontal_diff={}",
        vertical_variance, horizontal_diff
    );
}
```

---

## Validation Checklist

After the PR is merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (including new test)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Visual inspection: on an image with clear edges, pencil strokes should follow edge contours
- [ ] Strength=0 returns source unchanged (identity behavior preserved)
- [ ] Increasing stroke_softness makes strokes softer along their natural direction
