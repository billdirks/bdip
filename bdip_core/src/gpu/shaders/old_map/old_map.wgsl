// Old Map — single-pass antique-map effect.
//
// Processing steps:
//   1. Desaturate the source image using Rec. 709 luminance coefficients.
//   2. Apply the W3C sepia color matrix in linear light to produce a warm
//      brown tone, yielding a `sepia_rgb` value.
//   3. Generate a procedural parchment texture (warm off-white with subtle
//      noise variation) at the current pixel's normalised UV position.
//      The noise is a two-octave fract-sin hash that produces a warm, cream-
//      coloured surface with slight grain, mimicking aged paper.
//   4. Additively blend the parchment texture into the sepia result.
//   5. Mix the blended result back with the original source by `strength`:
//      at strength=0.0 the output is the original unchanged (identity);
//      at strength=1.0 the full antique-map effect is applied.
//
// The parchment texture is generated procedurally — no auxiliary texture
// asset is required.  The base colour is a warm off-white (R≈0.93, G≈0.87,
// B≈0.72 in linear light) with ±0.05 grain variation per octave.
//
// Distinction from the "Parchment" shader: the Parchment shader overlays a
// grain texture (loaded from an aux asset) on the unmodified source. Old Map
// desaturates and sepia-tones the source first, then blends a
// procedurally-generated parchment colour to simulate an antique printed map.

struct OldMapParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: OldMapParams;

// Rec. 709 luminance coefficients for linear-light RGB.
const LUM_R: f32 = 0.2126;
const LUM_G: f32 = 0.7152;
const LUM_B: f32 = 0.0722;

// Parchment base colour in linear light: a warm off-white leaning toward
// cream / aged paper.  Values chosen so the RGB triple reads as a warm
// ivory when displayed in sRGB.
const PARCHMENT_R: f32 = 0.93;
const PARCHMENT_G: f32 = 0.87;
const PARCHMENT_B: f32 = 0.72;

// Maximum grain amplitude (additive, per channel) for each noise octave.
const GRAIN_AMP: f32 = 0.05;

// A deterministic pseudo-random value in [0, 1) seeded by two floats.
// Uses the standard fract-sin hash.
fn hash(a: f32, b: f32) -> f32 {
    return fract(sin(a * 127.1 + b * 311.7) * 43758.5453);
}

// Two-octave procedural grain centred around 0.  Returns a value in
// approximately [-2*GRAIN_AMP, +2*GRAIN_AMP] that is added to the parchment
// base colour.  The second octave uses a perpendicular seed to decorrelate
// it from the first.
fn parchment_grain(uv: vec2<f32>) -> f32 {
    let h0 = hash(uv.x, uv.y);
    let h1 = hash(uv.y + 3.7, uv.x + 1.3);
    // Remap each hash from [0,1) to [-1,+1), then scale.
    return (h0 * 2.0 - 1.0) * GRAIN_AMP
         + (h1 * 2.0 - 1.0) * GRAIN_AMP;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);
    let rgb   = pixel.rgb;

    // --- Step 1 & 2: Desaturate then sepia-tone (W3C matrix, linear light) ---
    // The W3C sepia matrix inherently desaturates and warms the image.
    // Output can exceed 1.0 for bright inputs; headroom is preserved.
    let sepia_r = 0.393 * rgb.r + 0.769 * rgb.g + 0.189 * rgb.b;
    let sepia_g = 0.349 * rgb.r + 0.686 * rgb.g + 0.168 * rgb.b;
    let sepia_b = 0.272 * rgb.r + 0.534 * rgb.g + 0.131 * rgb.b;
    let sepia_rgb = vec3<f32>(sepia_r, sepia_g, sepia_b);

    // --- Step 3: Procedural parchment colour at this pixel's UV ---
    let uv    = vec2<f32>(global_id.xy) / vec2<f32>(dims);
    let grain = parchment_grain(uv);
    let parchment = vec3<f32>(
        PARCHMENT_R + grain,
        PARCHMENT_G + grain,
        PARCHMENT_B + grain,
    );

    // --- Step 4: Additive parchment blend over sepia ---
    // Multiply the sepia colour by the parchment (values near 0.9 warm-toned)
    // to simulate ink printed on aged paper.  Multiplicative blend darkens the
    // image where the parchment colour is low (grain variation) and preserves
    // approximate brightness overall.
    let mapped = sepia_rgb * parchment;

    // --- Step 5: Mix with original by strength (0 = identity) ---
    let out = mix(rgb, mapped, params.strength);

    // Do NOT clamp — preserve headroom above 1.0 for downstream shaders.
    textureStore(output_texture, coord, vec4<f32>(out, pixel.a));
}
