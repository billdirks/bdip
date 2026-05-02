// Line Art — single-pass Sobel edge detection with white-background inversion.
//
// Algorithm:
//   1. Convert source pixel to grayscale (BT.709 luma).
//   2. Apply Sobel operator over the 3×3 neighbourhood to compute gradient magnitude.
//   3. Multiply magnitude by `threshold` (range [0.1, 10.0]) to control sensitivity;
//      clamp to [0, 1]. Higher values make faint edges visible as lines.
//   4. Invert so edges are dark lines on a white background: line_value = 1 - magnitude.
//   5. Blend with the source using `strength`: output = mix(src, line_art, strength).
//      At strength=0.0 the output equals the source (identity).

struct LineArtParams {
    // Sensitivity multiplier applied to raw Sobel magnitude. Higher values
    // amplify weak edges. Range [0.1, 10.0]; default 2.0.
    threshold: f32,
    // Blend weight: 0.0 = source unchanged (identity), 1.0 = full line-art effect.
    strength:  f32,
    _padding:  vec2<f32>,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: LineArtParams;

// BT.709 luma coefficients for perceptually-weighted grayscale conversion.
const LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Computes the Sobel gradient vector (gx, gy) at `coord`.
//
// The 3×3 neighbourhood is sampled using clamped coordinates so border pixels
// mirror the nearest edge rather than wrapping. Raw magnitude for normalised
// [0, 1] inputs is in [0, ~4√2]; dividing by 4 in the caller maps typical
// magnitudes to roughly [0, 1] before the threshold multiplier is applied.
fn sobel(coord: vec2<i32>, dims: vec2<u32>) -> vec2<f32> {
    var n: array<f32, 9>;
    var k: i32 = 0;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let s = textureLoad(
                src_texture,
                clamp(coord + vec2<i32>(dx, dy), vec2<i32>(0), vec2<i32>(dims) - 1),
                0,
            );
            n[k] = dot(s.rgb, LUMA_WEIGHTS);
            k++;
        }
    }
    // n layout (row-major, top-left origin):
    //   n[0] n[1] n[2]
    //   n[3] n[4] n[5]
    //   n[6] n[7] n[8]
    //
    // Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    // Gy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
    let gx = -n[0] + n[2] - 2.0 * n[3] + 2.0 * n[5] - n[6] + n[8];
    let gy = -n[0] - 2.0 * n[1] - n[2] + n[6] + 2.0 * n[7] + n[8];
    return vec2<f32>(gx, gy);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(src_texture, coord, 0);

    let gradient  = sobel(coord, dims);

    // Raw Sobel magnitude is nominally in [0, ~4] for normalised inputs.
    // Dividing by 4 puts the unscaled result near [0, 1], then the threshold
    // parameter adjusts sensitivity.
    let magnitude = clamp(length(gradient) * (params.threshold / 4.0), 0.0, 1.0);

    // Invert: strong edges become dark lines; flat regions become white (1.0).
    let line_value = 1.0 - magnitude;
    let line_rgb   = vec3<f32>(line_value, line_value, line_value);

    // Blend between source and line-art result. At strength=0.0 this is the
    // identity transformation — output equals source.
    let out_rgb = mix(src.rgb, line_rgb, params.strength);

    // Do NOT clamp — preserve >1.0 headroom for downstream shaders.
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
