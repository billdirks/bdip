// Tintype — Pass 2: coarse procedural grit overlay and final blend
//
// Reads the original source (binding 0) and the vignetted scratch from Pass 1
// (binding 1), overlays a coarse procedural grit texture to simulate the rough
// iron/tin plate surface, then blends the result with the original based on
// `strength`.
//
// Grit approach (distinct from fine grain in Daguerreotype):
//   Tintype plates had a coarser surface than polished daguerreotype silver.
//   The grit is implemented as two-frequency procedural noise: a low-frequency
//   "clumping" term (6×6 block hash) combined with per-pixel high-frequency
//   noise.  The combination gives textured patches rather than even grain, which
//   better resembles the uneven iron collodion surface of period tintypes.
//   Both noise frequencies are derived from coordinate hashes alone — no
//   external textures are required.
//
//   Grit amplitude is intentionally higher than Daguerreotype grain (0.04 vs
//   0.015) and is modulated by (1 − luma) rather than luma, meaning dark areas
//   receive more grit.  This matches the physical reality: the dark, thinly
//   coated regions of the plate show the underlying metal texture most clearly.
//
// Bind-group layout (2-input pass; N=2, so output is binding 2):
//   group(0) binding(0): source texture     (original)
//   group(0) binding(1): vignetted scratch  (Pass 1 output)
//   group(0) binding(2): output storage
//   group(1) binding(0): uniform params

struct TintypeParams {
    strength: f32,
    _pad0:    f32,
    _pad1:    f32,
    _pad2:    f32,
}

@group(0) @binding(0) var src_texture:      texture_2d<f32>;
@group(0) @binding(1) var vignetted_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture:       texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform>           params: TintypeParams;

// Rec. 709 luminance coefficients.
const LUM_R: f32 = 0.2126;
const LUM_G: f32 = 0.7152;
const LUM_B: f32 = 0.0722;

// Maximum grit amplitude in linear [0,1] space.  0.04 corresponds to roughly
// ±2600 u16 at mid-tone — noticeably coarser than fine photographic grain.
const GRIT_AMP: f32 = 0.04;

// Block size for the low-frequency "clumping" noise component.
// A 6-pixel block size produces texture patches visible at typical viewing distances.
const BLOCK_SIZE: u32 = 6u;

// Hash function: combines two u32 coordinates into a pseudo-random f32 in [0, 1].
// PCG-style integer hash chosen for even bit distribution and low GPU register cost.
fn hash2(x: u32, y: u32) -> f32 {
    var v = x * 1664525u + y * 1013904223u;
    v = v ^ (v >> 16u);
    v = v * 2246822519u;
    v = v ^ (v >> 13u);
    v = v * 3266489917u;
    v = v ^ (v >> 16u);
    return f32(v) / 4294967295.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let src_dims = textureDimensions(src_texture);
    if global_id.x >= src_dims.x || global_id.y >= src_dims.y { return; }

    let coord     = vec2<i32>(global_id.xy);
    let src       = textureLoad(src_texture,       coord, 0);
    let vignetted = textureLoad(vignetted_texture, coord, 0);

    // --- Coarse Grit Texture ---
    // Two-frequency noise: low-frequency block term + high-frequency per-pixel term.
    let bx         = global_id.x / BLOCK_SIZE;
    let by         = global_id.y / BLOCK_SIZE;
    let low_freq   = hash2(bx * 7919u, by * 6271u);          // block-level clumping
    let high_freq  = hash2(global_id.x, global_id.y);        // per-pixel detail

    // Combine: 55% block structure + 45% fine detail for visible texture patches.
    let combined = low_freq * 0.55 + high_freq * 0.45;

    // Grit is heavier in dark areas (1 − luma) — dark regions expose the bare plate.
    let luma        = vignetted.r * LUM_R + vignetted.g * LUM_G + vignetted.b * LUM_B;
    let dark_weight = 1.0 - clamp(luma, 0.0, 1.0);
    // Centre noise around 0: [0,1] → [-1, +1], then scale by amplitude and weight.
    let grit        = (combined * 2.0 - 1.0) * GRIT_AMP * dark_weight;

    // Apply grit to all three channels equally (monochrome grit on monochrome image).
    let processed = vec3<f32>(
        vignetted.r + grit,
        vignetted.g + grit,
        vignetted.b + grit,
    );

    // Final blend: mix original source with fully processed result based on strength.
    // At strength=0 the output equals the source (identity).
    let out_rgb = mix(src.rgb, processed, params.strength);
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
