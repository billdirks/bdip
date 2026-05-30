// ASCII Art — Pass 2: Character-cell rendering with atlas lookup.
//
// Bindings (position-indexed, 2 inputs):
//   @group(0) @binding(0)  source texture   (original colour)
//   @group(0) @binding(1)  gray scratch      (BT.709 luma from pass 1)
//   @group(0) @binding(2)  destination       (rgba16float, write)
//   @group(1) @binding(0)  uniform params
//   @group(2) @binding(0)  ascii_char_map_16x16 (128×128 atlas, nearest)
//   @group(2) @binding(1)  atlas sampler
//
// Algorithm:
//   1. Snap each pixel to its cell origin (floor division by cell_size).
//   2. Sample the centre of the cell in the grey scratch to get average luma
//      for that cell. A centre sample is a good approximation and avoids an
//      expensive box-filter loop inside a compute shader.
//   3. Map luma [0, 1] → character index [0, 15] (0 = space, 15 = @).
//   4. Compute the pixel's sub-position within its character cell, clamped to
//      [0, 7] to index into the 8×8 character bitmask.
//   5. Sample the 128-wide atlas at u = (char_idx * 8 + sub_x) / 128,
//      v = sub_y / 128 (nearest filter). The atlas encodes ink as white (≈1.0)
//      and background as black (0.0).
//   6. Blend: ink pixels take a tinted version of the cell's average source
//      colour; background pixels use a dark neutral. Both are then mixed with
//      the original source by (1 - strength) to achieve a smooth fade-out.
//
// Identity: when params.strength == 0.0, mix(ascii_result, src, 0.0) = src.

struct AsciiArtParams {
    cell_size: f32,
    strength:  f32,
    _padding:  vec2<f32>,
}

@group(0) @binding(0) var source_texture:  texture_2d<f32>;
@group(0) @binding(1) var gray_texture:    texture_2d<f32>;
@group(0) @binding(2) var output_texture:  texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: AsciiArtParams;
@group(2) @binding(0) var char_atlas:      texture_2d<f32>;
@group(2) @binding(1) var atlas_sampler:   sampler;

// Number of characters in the atlas (one row).
const CHAR_COUNT: f32  = 16.0;
// Width of each character cell in the atlas (pixels).
const CHAR_PIX:   f32  = 8.0;
// Total atlas width in pixels.
const ATLAS_W:    f32  = 128.0;
// Total atlas height in pixels (only one row).
const ATLAS_H:    f32  = 8.0;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(source_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord     = vec2<i32>(global_id.xy);
    let fcoord    = vec2<f32>(global_id.xy);

    // ---------------------------------------------------------------------------
    // 1. Determine cell origin (in image pixels).
    // ---------------------------------------------------------------------------
    let cs        = max(params.cell_size, 1.0);
    let cell_xy   = floor(fcoord / cs) * cs;                 // top-left of cell
    let cell_ctr  = cell_xy + vec2<f32>(cs * 0.5, cs * 0.5); // centre of cell

    // Clamp to valid texture coordinates.
    let ctr_coord = vec2<i32>(clamp(cell_ctr, vec2<f32>(0.0), vec2<f32>(dims) - 1.0));

    // ---------------------------------------------------------------------------
    // 2. Sample luma at cell centre (proxy for average cell brightness).
    // ---------------------------------------------------------------------------
    let luma = textureLoad(gray_texture, ctr_coord, 0).r;

    // ---------------------------------------------------------------------------
    // 3. Map luma to character index [0, 15].
    //    luma is linear light; characters are ordered by visual ink density, so
    //    darker areas (low luma) map to sparse characters (space, dot, comma)
    //    and brighter areas map to denser characters (n, x, #, @).
    //    We invert luma so bright areas become dense characters.
    // ---------------------------------------------------------------------------
    let density   = 1.0 - clamp(luma, 0.0, 1.0);
    let char_idx  = floor(density * (CHAR_COUNT - 1.0) + 0.5);

    // ---------------------------------------------------------------------------
    // 4. Sub-pixel position within the character cell, mapped to [0, 7].
    // ---------------------------------------------------------------------------
    let sub       = fcoord - cell_xy;              // [0, cs)
    let sub_norm  = sub / cs;                      // [0, 1)
    let sub_px    = floor(sub_norm * CHAR_PIX);    // [0, 7]

    // ---------------------------------------------------------------------------
    // 5. Sample atlas.
    //    UV origin is top-left; v points downward in texture space.
    // ---------------------------------------------------------------------------
    let atlas_u   = (char_idx * CHAR_PIX + sub_px.x + 0.5) / ATLAS_W;
    let atlas_v   = (sub_px.y + 0.5) / ATLAS_H;
    let ink_mask  = textureSampleLevel(char_atlas, atlas_sampler,
                                       vec2<f32>(atlas_u, atlas_v), 0.0).r;

    // ---------------------------------------------------------------------------
    // 6. Compose pixel colour.
    //
    //    Cell average colour is sampled from the source at the cell centre
    //    (same proxy approach as luma). Ink pixels get this tinted colour;
    //    background pixels get near-black.
    //
    //    We do NOT clamp the output to [0, 1] to preserve >1.0 linear-light
    //    headroom for downstream shaders.
    // ---------------------------------------------------------------------------
    let src_pixel  = textureLoad(source_texture, coord, 0);
    let src_ctr    = textureLoad(source_texture, ctr_coord, 0);

    // Ink colour: preserve the cell's hue/saturation but scale by a mild factor
    // to give the "printed on white" feel without washing out the image.
    let ink_col    = src_ctr.rgb * 1.1;
    // Background colour: very dark (slightly tinted by source for warmth).
    let bg_col     = src_ctr.rgb * 0.05;

    // Choose ink vs background based on atlas mask (threshold at 0.5).
    let ascii_col  = select(bg_col, ink_col, ink_mask > 0.5);

    // Blend ascii result with original source using strength.
    // When strength = 0.0 this is an identity (output == source).
    let out_rgb    = mix(src_pixel.rgb, ascii_col, params.strength);

    textureStore(output_texture, coord, vec4<f32>(out_rgb, src_pixel.a));
}
