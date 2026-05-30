// Graffiti — Pass 2: color quantization + edge darkening + blend.
//
// Produces the final graffiti output from the bleed-blurred scratch texture
// (Pass 1) and the original source texture. The pipeline:
//
//   1. Posterize (quantize) the blurred image to `color_levels` discrete steps
//      per channel, creating bold flat color zones.
//   2. Compute a Sobel edge magnitude on the *source* image so edges stay sharp
//      even when the color zones are flat.
//   3. Darken the quantized result by the edge magnitude scaled by `edge_strength`,
//      producing thick dark outlines at boundaries between color zones.
//   4. Blend the result back with the original source via `strength` (0.0 = identity).
//
// Identity: when strength = 0.0, `mix(src, graffiti, 0.0) = src`.
//
// All GraffitiParams fields must be declared in every pass to satisfy WebGPU's
// uniform binding-size validation requirement.

struct GraffitiParams {
    strength:      f32,
    color_levels:  f32,
    edge_strength: f32,
    bleed:         f32,
}

// 2 inputs: source at binding 0, bleed scratch at binding 1, output at binding 2.
@group(0) @binding(0) var src_texture:   texture_2d<f32>;
@group(0) @binding(1) var bleed_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: GraffitiParams;

// BT.709 luma weights for perceptually-weighted Sobel grayscale conversion.
const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Applies the Sobel operator at `coord` and returns the gradient magnitude in
// [0, 1]. The raw magnitude on normalised [0,1] inputs is in [0, ~4√2]; dividing
// by 4 maps typical values to [0, ~1.4], then clamp brings it to [0, 1].
fn sobel_magnitude(coord: vec2<i32>, dims: vec2<u32>) -> f32 {
    var n: array<f32, 9>;
    var k: i32 = 0;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let s = textureLoad(
                src_texture,
                clamp(coord + vec2<i32>(dx, dy), vec2<i32>(0), vec2<i32>(dims) - 1),
                0,
            );
            n[k] = dot(s.rgb, LUMA);
            k++;
        }
    }
    // Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    // Gy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
    let gx = -n[0] + n[2] - 2.0 * n[3] + 2.0 * n[5] - n[6] + n[8];
    let gy = -n[0] - 2.0 * n[1] - n[2] + n[6] + 2.0 * n[7] + n[8];
    return clamp(length(vec2<f32>(gx, gy)) / 4.0, 0.0, 1.0);
}

// Quantizes a single channel value in [0, 1] to `levels` discrete steps.
fn quantize(v: f32, levels: f32) -> f32 {
    return floor(v * levels) / levels;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(src_texture,   coord, 0);
    let bleed = textureLoad(bleed_texture, coord, 0);

    // --- Step 1: posterize the blurred image ---
    // Quantize each channel of the blurred color to `color_levels` steps.
    // Using the blurred texture for quantization smooths out fine detail before
    // posterization, which widens the flat color bands and suppresses noise —
    // the characteristic look of a spray-painted color zone.
    let levels = max(params.color_levels, 2.0); // guard against degenerate input
    let quant_rgb = vec3<f32>(
        quantize(bleed.r, levels),
        quantize(bleed.g, levels),
        quantize(bleed.b, levels),
    );

    // --- Step 2: edge magnitude on the source ---
    // Using the source (not the blurred texture) preserves crisp edge positions
    // even when the bleed radius is large. The Sobel result is in [0, 1].
    let edge_mag = sobel_magnitude(coord, dims);

    // --- Step 3: darken the quantized color by edge magnitude ---
    // edge_strength multiplies the magnitude before subtracting from the quantized
    // color, producing dark outlines at color-zone boundaries. A cap at 1.0
    // prevents the subtraction from going negative (which would invert).
    let darkening  = clamp(edge_mag * params.edge_strength, 0.0, 1.0);
    let graffiti_rgb = quant_rgb * (1.0 - darkening);

    // --- Step 4: blend with source ---
    // At strength=0.0 this reduces to src.rgb exactly (identity).
    let out_rgb = mix(src.rgb, graffiti_rgb, params.strength);

    // Alpha is preserved from the source image.
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
