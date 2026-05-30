// Dust and Scratches shader
//
// Simulates aged film damage by compositing two procedurally generated
// artifact layers over the source image:
//
//   1. Scratch lines: Thin, near-vertical dark streaks that run the full
//      height of the frame.  Each column's scratch threshold is derived
//      from the blue-noise texture so scratch positions are stable and
//      spatially uncorrelated.
//
//   2. Dust specks: Small (1–3 px radius) dark blobs scattered randomly
//      across the frame.  Their positions are determined by comparing a
//      two-dimensional blue-noise lookup against the `dust_amount` threshold.
//
// Both layers darken pixels toward black.  The composite damage mask is
// blended back onto the source image using `params.strength`, so
// strength = 0.0 is a pure passthrough (identity).
//
// Blue-noise lookups are wrapped with `fract` to tile the 128×128 texture
// across arbitrarily large images without visible repetition.

struct DustAndScratchesParams {
    strength:        f32,
    scratch_density: f32,
    dust_amount:     f32,
    _padding:        f32,
}

@group(0) @binding(0) var src_texture:    texture_2d<f32>;
@group(0) @binding(1) var dst_texture:    texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: DustAndScratchesParams;
@group(2) @binding(0) var noise_texture:  texture_2d<f32>;
@group(2) @binding(1) var noise_sampler:  sampler;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Cheap integer hash — maps a 2-D cell coordinate to a pseudo-random float in
// [0, 1).  Used to assign each scratch column and dust cell a unique random
// threshold without additional texture lookups.
fn hash2(a: u32, b: u32) -> f32 {
    var h = a * 1664525u + b * 22695477u + 1013904223u;
    h ^= h >> 13u;
    h *= 0x9e3779b9u;
    h ^= h >> 15u;
    return f32(h & 0xFFFFu) / 65536.0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<u32>(gid.xy);
    let src   = textureLoad(src_texture, coord, 0);

    // Normalised UV for blue-noise sampling.  Tile by wrapping with fract so
    // the 128×128 texture covers images of any size.
    let uv = vec2<f32>(gid.xy) / 128.0;

    // ── 1. Scratch layer ────────────────────────────────────────────────────
    //
    // Scratch lines are vertical (constant x) dark streaks.  We partition the
    // image width into narrow columns of fixed pixel width.  For each column
    // we draw a random threshold; if that threshold is below `scratch_density`
    // the column is a scratch.
    //
    // Within a scratch column the scratch has a sub-pixel horizontal position
    // offset (derived from the column hash) so adjacent scratches are staggered.
    // A smooth anti-aliased profile darkens the pixel based on its distance to
    // the scratch centre.

    // Column width in pixels — narrower columns allow more scratches per frame.
    let scratch_col_width: f32 = 4.0;
    let col_idx = u32(f32(gid.x) / scratch_col_width);

    // Primary random value for this column (column presence threshold).
    let col_rnd = hash2(col_idx, 0u);

    // Whether this column is a scratch candidate (0 = no scratch, 1 = scratch).
    let is_scratch_col = select(0.0, 1.0, col_rnd < params.scratch_density * 0.3);

    // Sub-pixel centre offset within the column (0 … scratch_col_width).
    let scratch_centre_x = f32(col_idx) * scratch_col_width + hash2(col_idx, 1u) * scratch_col_width;
    let dist_to_scratch  = abs(f32(gid.x) - scratch_centre_x);

    // Pixel half-width of the scratch (0.4 … 0.8 px).
    let scratch_halfwidth = 0.4 + hash2(col_idx, 2u) * 0.4;

    // Smooth darkening profile: 1 at centre, 0 beyond halfwidth.
    let scratch_profile = is_scratch_col * (1.0 - smoothstep(0.0, scratch_halfwidth, dist_to_scratch));

    // Modulate along the scratch's length via blue-noise to create a slightly
    // irregular, broken-line look rather than a perfectly solid line.
    let noise_along = textureSampleLevel(noise_texture, noise_sampler, fract(uv), 0.0).r;
    // Breaks appear where noise > 0.88 — adjust threshold to control break frequency.
    let scratch_break = select(1.0, 0.0, noise_along > 0.88);

    let scratch_mask = scratch_profile * scratch_break;

    // ── 2. Dust layer ───────────────────────────────────────────────────────
    //
    // Dust specks are small dark blobs.  We partition the image into 4×4 cells;
    // each cell either contains a speck (if its random value is below the dust
    // threshold) or does not.  Pixels within a small radius of the cell centre
    // receive a darkening contribution proportional to proximity.

    let dust_cell_size: f32 = 4.0;
    let cell_x = u32(f32(gid.x) / dust_cell_size);
    let cell_y = u32(f32(gid.y) / dust_cell_size);

    // Per-cell random value for dust presence.
    let dust_rnd = hash2(cell_x * 1234u + cell_y, 42u);
    let has_dust = select(0.0, 1.0, dust_rnd < params.dust_amount * 0.15);

    // Speck centre (jittered within the cell).
    let speck_cx = (f32(cell_x) + 0.2 + hash2(cell_x + cell_y * 97u, 7u) * 0.6) * dust_cell_size;
    let speck_cy = (f32(cell_y) + 0.2 + hash2(cell_x * 53u + cell_y, 11u) * 0.6) * dust_cell_size;

    // Speck radius: 0.5 … 1.2 px.
    let speck_r = 0.5 + hash2(cell_x + cell_y * 31u, 99u) * 0.7;

    let dist_to_speck = length(vec2<f32>(f32(gid.x), f32(gid.y)) - vec2<f32>(speck_cx, speck_cy));
    let dust_profile  = has_dust * (1.0 - smoothstep(0.0, speck_r, dist_to_speck));

    // ── 3. Combine and blend ────────────────────────────────────────────────
    //
    // The combined damage mask darkens toward black.  A mask value of 1.0
    // means fully black at that pixel; 0.0 means no change.  We take the
    // maximum of the two layers so they don't double-darken when they overlap.

    let damage = clamp(max(scratch_mask, dust_profile), 0.0, 1.0);
    let damaged_rgb = src.rgb * (1.0 - damage);

    // Final blend: strength=0 returns the source unchanged (identity).
    let out_rgb = mix(src.rgb, damaged_rgb, params.strength);
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
