// Pencil Sketch — Pass 1: grayscale conversion and Sobel edge detection.
//
// Converts the source image to grayscale, then applies the Sobel operator to
// compute gradient magnitude and direction. The output is packed into an
// rgba16float scratch texture:
//
//   .r = edge intensity in [0, 1] (Sobel gradient magnitude, scaled by edge_strength)
//   .g = gradient angle in [0, 1] (atan2(gy, gx) mapped to [0, 1])
//   .b = 0 (unused)
//   .a = source alpha (passed through for the final composite)
//
// The gradient angle encodes the direction perpendicular to the edge, which
// is the direction pencil strokes naturally run along the edge line.
//
// All PencilSketchParams fields must be declared in every pass to satisfy
// WebGPU's uniform binding-size validation requirement.

struct PencilSketchParams {
    // Blend factor: 0.0 = source unchanged (identity), 1.0 = full effect.
    strength:        f32,
    // Multiplier on raw Sobel magnitude. Higher values amplify faint edges.
    edge_strength:   f32,
    // Directional blur extent in pass 2. 0.0 = no blur.
    stroke_softness: f32,
    _padding:        f32,
}

// Bindings — 1 input: source at binding 0, output at binding 1.
@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: PencilSketchParams;

// BT.709 luma coefficients for perceptually-weighted grayscale conversion.
const LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Applies the Sobel operator at `coord` and returns the gradient vector (gx, gy).
// The raw Sobel magnitude on normalised [0,1] inputs is in [0, ~4√2]; dividing by
// 4 in the caller maps typical magnitudes to roughly [0, 1].
fn sobel(coord: vec2<i32>, dims: vec2<u32>) -> vec2<f32> {
    // Sample the 3×3 neighbourhood (clamped to texture bounds).
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

    let gradient = sobel(coord, dims);

    // Raw Sobel magnitude is nominally in [0, ~4] for normalised inputs.
    // Dividing by 4 puts the unscaled result near [0, 1], then edge_strength
    // adjusts the sensitivity.
    let magnitude      = length(gradient) * (params.edge_strength / 4.0);
    let edge_intensity = clamp(magnitude, 0.0, 1.0);

    // Gradient angle mapped to [0, 1]: atan2 returns [-π, π]; adding π then
    // dividing by 2π maps the full circle to [0, 1].
    let pi         = 3.14159265358979;
    let angle_raw  = atan2(gradient.y, gradient.x); // [-π, π]
    let angle_norm = (angle_raw + pi) / (2.0 * pi); // [0, 1]

    textureStore(dst_texture, coord, vec4<f32>(edge_intensity, angle_norm, 0.0, src.a));
}
