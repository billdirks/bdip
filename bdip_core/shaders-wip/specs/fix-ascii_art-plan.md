# Fix ASCII Art Transform

## Problem Summary

The ASCII art transform has a critical algorithm bug that inverts image brightness, plus potential atlas
sampling issues that need verification.

### Critical Issues

1. **Character density mapping is inverted** (line 80 of `ascii_art_ascii.wgsl`): The formula
   `density = 1.0 - luma` maps bright pixels to sparse characters and dark pixels to dense characters.
   Combined with the ink/background color scheme (ink = bright, bg = dark), this **inverts the image**:
   - Bright input → sparse char (space) → mostly dark background → dark output
   - Dark input → dense char (@) → mostly ink → relatively brighter output

   The comment at lines 77-78 correctly describes the intended behavior ("brighter areas map to denser
   characters"), but the code does the opposite.

   **Evidence**: For a white input (luma=1.0):
   - density = 1.0 - 1.0 = 0.0
   - char_idx = 0 (space character)
   - Space has ~0% ink coverage
   - ascii_col = bg_col = white * 0.05 = very dark gray
   - Output is very dark instead of bright

2. **Atlas UV coordinate calculation may be incorrect** (lines 94-96): The shader computes:
   ```wgsl
   let atlas_v = (sub_px.y + 0.5) / ATLAS_H;  // ATLAS_H = 8.0
   ```
   
   For a 128×128 texture, this produces `atlas_v` values from 0.0625 to 0.9375, which samples rows
   8-120 instead of rows 0-7 where the characters appear to be located. This needs verification by
   examining whether the atlas has characters replicated vertically or if the UV math is simply wrong.

### Minor Issues

3. **Documentation dimension mismatch**: The comment in `ascii_char_map.rs` says "128×128 greyscale
   character density atlas" but the effective character content appears to occupy only 128×8 pixels
   (16 characters × 8×8 each in a single row). The file is 128×128 but most rows appear unused.

### Current Parameters

| Parameter | Range     | Default | Status |
|-----------|-----------|---------|--------|
| Cell Size | 4.0–32.0  | 8.0     | OK     |
| Strength  | 0.0–1.0   | 0.0     | OK     |

### Missing Parameters (Enhancement Opportunities)

- **Contrast/Gamma**: Control tonal distribution across character ramp
- **Ink/Background Colors**: Allow customization of the color scheme
- **Edge Detection Mode**: Optional edge enhancement for improved detail

---

## Implementation Plan

### PR 1: Fix Character Density Mapping Inversion

**Goal**: Correct the density calculation so bright areas produce dense characters and preserve image
brightness.

**Scope**:
- `bdip_core/src/gpu/shaders/ascii_art/ascii_art_ascii.wgsl`

**Implementation Details**:

Change line 80 from:
```wgsl
let density = 1.0 - clamp(luma, 0.0, 1.0);
```
to:
```wgsl
let density = clamp(luma, 0.0, 1.0);
```

Update the comment at lines 73-78 to accurately describe the mapping:
```wgsl
// 3. Map luma to character index [0, 15].
//    luma is linear light; characters are ordered by visual ink density.
//    Bright areas (high luma) need dense characters (lots of ink) to stay
//    bright; dark areas need sparse characters (mostly background) to stay
//    dark.
```

**Tests to Add**:

1. `test_ascii_art_preserves_brightness_direction`: Verify that a bright input produces brighter
   output than a dark input (not inverted).

2. `test_ascii_art_white_input_stays_bright`: White input at strength=1.0 should produce output
   significantly brighter than mid-gray (average luminance > 0.5 relative to white).

3. `test_ascii_art_black_input_stays_dark`: Black input at strength=1.0 should produce output
   close to black (average luminance near 0).

---

### PR 2: Verify and Fix Atlas UV Sampling

**Goal**: Ensure the atlas UV calculation correctly samples character glyphs.

**Scope**:
- `bdip_core/src/gpu/shaders/ascii_art/ascii_art_ascii.wgsl`
- Potentially `bdip_core/src/gpu/assets/ascii_char_map_16x16.png` (may need regeneration)

**Investigation Steps**:

1. Examine the full 128×128 atlas texture to determine if characters are replicated vertically or
   only present in the top 8 rows.

2. If characters are only in rows 0-7, fix the UV calculation:
   ```wgsl
   // Current (potentially wrong):
   let atlas_v = (sub_px.y + 0.5) / ATLAS_H;
   
   // Fixed (if texture is 128 tall with chars in top 8 rows):
   let atlas_v = (sub_px.y + 0.5) / 128.0;
   ```

3. Alternatively, regenerate the atlas to be 128×8 and update the UV math accordingly.

**Tests to Add**:

1. `test_ascii_art_character_variation_within_cell`: Verify that different sub-pixel positions
   within a character cell produce different output values (proving the atlas lookup works).

2. `test_ascii_art_dense_character_has_more_ink`: For a uniform bright input, verify that pixels
   within dense character regions have more ink than sparse character regions.

---

### PR 3: Documentation Cleanup

**Goal**: Fix documentation inconsistencies.

**Scope**:
- `bdip_core/src/gpu/assets/ascii_char_map.rs`

**Changes**:

Update the comment to accurately describe the atlas dimensions:
```rust
// 128×8 greyscale character density atlas (stored in a 128×128 PNG with padding).
//
// The texture contains 16 ASCII characters (8×8 px each) arranged in a
// single row in the top 8 pixels, ordered from least dense (space, index 0)
// to most dense (@, index 15).
```

Or if the atlas is regenerated to 128×8:
```rust
// 128×8 greyscale character density atlas.
//
// The texture contains 16 ASCII characters (8×8 px each) arranged in a
// single row, ordered from least dense (space, index 0) to most dense
// (@, index 15).
```

---

## Test Specifications

### PR 1 Tests (Detailed)

#### `test_ascii_art_preserves_brightness_direction`

```rust
/// Verify that bright input produces brighter output than dark input
/// (i.e., the effect does not invert the image).
#[test]
fn test_ascii_art_preserves_brightness_direction() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // White input
    let white_img = make_solid_image(32, 32, 60000, 60000, 60000);
    let white_out = roundtrip(
        &mut renderer,
        &engine,
        &white_img,
        &[Transform {
            shader_id: "ascii_art",
            values: vec![8.0, 1.0],
        }],
    );

    // Black input
    let black_img = make_solid_image(32, 32, 5000, 5000, 5000);
    let black_out = roundtrip(
        &mut renderer,
        &engine,
        &black_img,
        &[Transform {
            shader_id: "ascii_art",
            values: vec![8.0, 1.0],
        }],
    );

    // Compute average luminance
    let avg_lum = |img: &Rgba16Image| {
        let sum: u64 = img.pixels().map(|p| p[0] as u64).sum();
        sum as f64 / (img.width() * img.height()) as f64
    };

    let white_avg = avg_lum(&white_out);
    let black_avg = avg_lum(&black_out);

    assert!(
        white_avg > black_avg,
        "white input must produce brighter output than black: white_avg={:.0}, black_avg={:.0}",
        white_avg, black_avg
    );
}
```

#### `test_ascii_art_white_input_stays_bright`

```rust
/// Verify that white input at full strength produces relatively bright output.
#[test]
fn test_ascii_art_white_input_stays_bright() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(32, 32, 65535, 65535, 65535);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "ascii_art",
            values: vec![8.0, 1.0],
        }],
    );

    let avg: u64 = out.pixels().map(|p| p[0] as u64).sum();
    let avg = avg / (out.width() * out.height()) as u64;

    // White input should produce output with average > 50% brightness
    // (dense characters with bright ink)
    assert!(
        avg > 32767,
        "white input should stay bright, got average {}",
        avg
    );
}
```

#### `test_ascii_art_black_input_stays_dark`

```rust
/// Verify that black input at full strength produces dark output.
#[test]
fn test_ascii_art_black_input_stays_dark() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(32, 32, 0, 0, 0);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "ascii_art",
            values: vec![8.0, 1.0],
        }],
    );

    let avg: u64 = out.pixels().map(|p| p[0] as u64).sum();
    let avg = avg / (out.width() * out.height()) as u64;

    // Black input should produce output with average < 10% brightness
    // (sparse characters with dark background)
    assert!(
        avg < 6553,
        "black input should stay dark, got average {}",
        avg
    );
}
```

### PR 2 Tests (Detailed)

#### `test_ascii_art_character_variation_within_cell`

```rust
/// Verify that the atlas lookup produces variation within character cells.
#[test]
fn test_ascii_art_character_variation_within_cell() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Mid-gray input to select a character with both ink and background pixels
    let img = make_solid_image(32, 32, 32767, 32767, 32767);
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "ascii_art",
            values: vec![8.0, 1.0], // 8px cells
        }],
    );

    // Within a single 8x8 cell, there should be variation between
    // ink pixels and background pixels
    let cell_pixels: Vec<u16> = (0..8)
        .flat_map(|y| (0..8).map(move |x| out.get_pixel(x, y)[0]))
        .collect();
    
    let min = *cell_pixels.iter().min().unwrap();
    let max = *cell_pixels.iter().max().unwrap();

    assert!(
        max - min > 1000,
        "character cell should have variation between ink and background: min={}, max={}",
        min, max
    );
}
```

---

## Validation Checklist

After all PRs are merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Visual test: white image → ASCII art produces bright output with visible characters
- [ ] Visual test: black image → ASCII art produces dark output (mostly background)
- [ ] Visual test: gradient image → ASCII art preserves gradient direction (bright to dark)
- [ ] Visual test: different cell sizes produce visibly different character densities
- [ ] Strength=0 returns source unchanged (identity)

---

## References

- [Character representation of grey scale images](https://paulbourke.net/dataformats/asciiart/) - Paul
  Bourke's reference on ASCII art character ramps ordered by density
- [ASCII characters are not pixels: a deep dive](https://alexharri.com/blog/ascii-rendering) - Detailed
  analysis of ASCII rendering techniques
- [Creating an ASCII Shader Using OGL](https://tympanus.net/codrops/2024/11/13/creating-an-ascii-shader-using-ogl/) -
  Modern WebGL implementation reference
- [ASCII Art in Pixel Shader](https://asawicki.info/news_1277_ascii_art_in_pixel_shader) - GPU shader
  implementation techniques
