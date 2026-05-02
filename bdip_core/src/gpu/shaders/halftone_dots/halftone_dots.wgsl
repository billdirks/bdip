// Halftone Dots — single-pass sine-wave grid mask.
//
// Algorithm:
//   1. Compute the perceptual luminance of the source pixel (linear-light Rec.709).
//   2. Evaluate a separable sine grid at the pixel coordinate:
//        grid = sin(TAU * frequency * x) * sin(TAU * frequency * y)
//      The product ranges in [-1, 1].
//   3. Derive a per-pixel threshold from luminance:
//        threshold = 1.0 - 2.0 * luminance
//      At luminance=1.0 (white), threshold=-1.0 → almost all grid values exceed the
//      threshold → mostly white output (small effective black dots).
//      At luminance=0.0 (black), threshold=1.0 → almost no grid values exceed the
//      threshold → mostly black output (large effective black dots).
//   4. Select white (1.0) when grid > threshold, black (0.0) otherwise.
//   5. Blend the binary halftone with the source using the `strength` parameter.
//      At strength=0.0 the source is returned unchanged (identity).
//
// The textures operate in linear-light Rgba16Float. No clamping is applied to the
// blended result so that headroom above 1.0 is preserved for downstream shaders.

const TAU: f32 = 6.283185307179586;

struct HalftoneDotParams {
    strength:  f32,
    frequency: f32,
    _padding:  vec2<f32>,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: HalftoneDotParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Perceptual luminance in linear light (Rec.709 coefficients).
    let lum = dot(pixel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Sine grid evaluated at the integer pixel position.
    let fx = f32(global_id.x);
    let fy = f32(global_id.y);
    let grid = sin(TAU * params.frequency * fx) * sin(TAU * params.frequency * fy);

    // Luminance-derived threshold: maps [0, 1] lum to [1, -1] threshold (inverted).
    // Bright areas set a low threshold so most of the grid passes → mostly white output.
    // Dark areas set a high threshold so little of the grid passes → mostly black output.
    let threshold = 1.0 - 2.0 * lum;

    // Binary halftone: white inside the dot, black outside.
    let halftone_value = select(0.0, 1.0, grid > threshold);
    let halftone = vec4<f32>(halftone_value, halftone_value, halftone_value, pixel.a);

    // Blend with source. At strength=0.0 this is a pass-through (identity).
    let out = mix(pixel, halftone, params.strength);
    textureStore(output_texture, coord, out);
}
