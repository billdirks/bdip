# Fix Cyberpunk Transform

## Problem Summary

The cyberpunk transform produces a "teal and orange" cinematic look instead of the signature cyberpunk
"cyan and magenta" neon aesthetic. The implementation borrows directly from the teal_and_orange shader
but uses the wrong highlight color for cyberpunk imagery.

### Critical Issue

1. **Wrong highlight color**: The shader uses orange `(0.32, 0.14, 0.0)` for highlights, but cyberpunk
   aesthetics call for **magenta/pink** highlights. Research consistently shows cyberpunk color grading
   pairs cyan/teal shadows with magenta/pink highlights, not orange:

   > "The overall color palette tends to lean towards blue and magenta, with darker areas having a
   > colder blue tone, while the highlights from the lights have a magenta hue."
   > — [TourboxTech: Creating Cyberpunk Color Grading](https://www.tourboxtech.com/en/news/cyberpunk-colors.html)

   > "Dominant color palette of cyan and magenta."
   > — [Filmora: Cyberpunk Color Palette](https://filmora.wondershare.com/video-creative-tips/cyberpunk-color-palette.html)

   The code at `cyberpunk.wgsl:84` explicitly uses `orange_target = vec3<f32>(0.32, 0.14, 0.0)`, which
   is nearly identical to the teal_and_orange shader's orange target `(0.37, 0.18, 0.0)`.

2. **Misleading description**: The description claims to boost "cyans and magentas" but the split-tone
   actually pushes highlights toward orange, which conflicts with magenta. The cm_b boost (+14% blue)
   partially compensates but doesn't produce true magenta highlights.

### Current Algorithm (Steps)

1. Shadow deepening via power curve (exponent 1.0-1.6) — **Correct**
2. Cyan/magenta channel boost (R -8%, G +4%, B +14%) — **Partially correct** (boosts cyan, not magenta)
3. Teal-to-orange split tone — **Wrong** (should be teal-to-magenta)
4. Neon saturation boost (+35% chroma) — **Correct**
5. Blend with original — **Correct**

### Missing Parameters

The shader has only a single "Strength" parameter. Additional controls would improve usability:
- **Neon Intensity**: Control saturation boost separately from overall strength
- **Shadow Depth**: Control how much to crush shadows

However, these are enhancements, not critical fixes.

---

## Implementation Plan

### PR 1: Replace Orange Highlights with Magenta

**Goal**: Change the highlight split-tone from orange to magenta to produce authentic cyberpunk
color grading.

**Scope**:
- Modify `cyberpunk.wgsl` to use magenta highlight target instead of orange
- Update comments to reflect the cyan/magenta split (not teal/orange)
- Update `mod.rs` description to remove "teal-to-orange" reference

**Implementation Details**:

The highlight target should be changed from orange to magenta. A proper magenta in linear light
that maintains similar luminance (~0.25) to avoid brightness shifts:

```wgsl
// OLD (wrong - produces teal and orange look):
let orange_target = vec3<f32>(0.32, 0.14, 0.0);

// NEW (correct - produces cyan and magenta look):
// Magenta: high R and B, low G. Luminance ≈ 0.2126*0.7 + 0.7152*0.1 + 0.0722*0.7 ≈ 0.27
let magenta_target = vec3<f32>(0.70, 0.10, 0.70);
```

The magenta target `(0.70, 0.10, 0.70)` has:
- High red (0.70) for warmth
- Low green (0.10) to create magenta (R+B without G)
- High blue (0.70) for the neon glow
- Luminance ≈ 0.27, similar to the original orange's ~0.22

Also rename the variable from `orange_target` to `magenta_target` and update `orange_contrib` to
`magenta_contrib` for clarity.

**Updated Shader Section** (pseudocode):

```wgsl
// ── Step 3: Cyan-to-magenta split tone ─────────────────────────────────────
// Shadows tilt toward cyan (blue-green); highlights tilt toward magenta (pink).
// This produces the signature cyberpunk neon aesthetic.
let lum = luminance(neon_rgb);

let shadow_w    = smoothstep(0.5, 0.0, lum);
let highlight_w = smoothstep(0.5, 1.0, lum);

// Cyan: blue-green tint for shadows (darker areas of the scene).
// Magenta: neon pink tint for highlights (light sources, reflections).
let cyan_target    = vec3<f32>(0.0,  0.22, 0.30);
let magenta_target = vec3<f32>(0.70, 0.10, 0.70);

let split_strength  = params.strength * 0.5;
let cyan_contrib    = split_strength * shadow_w    * (cyan_target    - neon_rgb);
let magenta_contrib = split_strength * highlight_w * (magenta_target - neon_rgb);
let split_rgb = neon_rgb + cyan_contrib + magenta_contrib;
```

**Updated Description**:

```rust
const DESCRIPTION: &'static str = "Neon-lit color grade: boosts cyans and magentas, deepens shadows, \
     adds a cyan-to-magenta split tone, and pushes neon saturation.";
```

**Tests to Add**:

1. `test_cyberpunk_highlights_shift_toward_magenta`: Verify bright pixels gain magenta character
   (R and B both increase relative to G) at full strength.

2. `test_cyberpunk_does_not_add_orange`: Verify that highlights do NOT gain orange character
   (B should not decrease relative to R on bright inputs).

**Tests to Update**:

The existing test `test_cyberpunk_full_strength_boosts_blue_on_bright_red` should still pass because
magenta (R+B) also has high blue. However, update the comment to reflect the magenta split rather
than referencing the cm_b term as the primary blue source.

---

## Test Specifications

### `test_cyberpunk_highlights_shift_toward_magenta`

```rust
/// At full strength, bright (highlight) pixels should gain magenta character.
/// Magenta = high R + high B with suppressed G. On a near-white input, after grading
/// both R and B should remain high while G decreases relative to them.
#[test]
fn test_cyberpunk_highlights_shift_toward_magenta() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Bright neutral input (highlight range, lum > 0.5)
    let img = make_solid_image(2, 2, 55000, 55000, 55000);
    let graded = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "cyberpunk",
            values: vec![1.0],
        }],
    );

    for pixel in graded.pixels() {
        // Magenta means R and B both higher than G
        assert!(
            pixel[0] > pixel[1] && pixel[2] > pixel[1],
            "highlight should be magenta (R > G and B > G): R={} G={} B={}",
            pixel[0], pixel[1], pixel[2]
        );
    }
}
```

### `test_cyberpunk_does_not_add_orange`

```rust
/// Verify that the cyberpunk grade does NOT push highlights toward orange.
/// Orange would mean R increases while B decreases. With magenta highlights,
/// both R and B should be boosted.
#[test]
fn test_cyberpunk_does_not_add_orange() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Start with equal R and B on a bright pixel
    let img = make_solid_image(2, 2, 50000, 30000, 50000);
    let identity = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "cyberpunk",
            values: vec![0.0],
        }],
    );
    let graded = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "cyberpunk",
            values: vec![1.0],
        }],
    );

    for (g, id) in graded.pixels().zip(identity.pixels()) {
        // Orange would be: R increases, B decreases (R-B gap widens)
        // Magenta should keep R and B relatively balanced
        let id_gap = (id[0] as i32 - id[2] as i32).abs();
        let g_gap = (g[0] as i32 - g[2] as i32).abs();
        
        // The R-B gap should not significantly increase (which would indicate orange)
        assert!(
            g_gap <= id_gap + 5000,
            "R-B gap should not widen significantly (orange behavior): identity_gap={} graded_gap={}",
            id_gap, g_gap
        );
    }
}
```

---

## Validation Checklist

After PR 1 is merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (including new tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Visual check: bright areas show pink/magenta tint, not orange
- [ ] Visual check: dark areas show cyan/teal tint (unchanged)
- [ ] Strength=0 returns source unchanged (identity)
- [ ] Alpha channel preserved at all strength levels

---

## References

- [TourboxTech: Creating Cyberpunk Color Grading in Photoshop](https://www.tourboxtech.com/en/news/cyberpunk-colors.html)
- [Filmora: The Best 15 Cyberpunk Color Palette Ideas](https://filmora.wondershare.com/video-creative-tips/cyberpunk-color-palette.html)
- [SpoonGraphics: How to Apply Cyberpunk Style Color Grading](https://blog.spoongraphics.co.uk/tutorials/how-to-apply-cyberpunk-style-color-grading-neon-effects-to-your-photos)
- [Media.io: 20+ Cyberpunk Color Palette Combinations](https://www.media.io/color-palette/cyberpunk-color-palette.html)
- [Catlike Coding: Color Grading (Unity Tutorial)](https://catlikecoding.com/unity/tutorials/custom-srp/color-grading/)
