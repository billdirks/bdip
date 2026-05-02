// Watercolor Edge — Pass 1: Sobel edge detection.
//
// Converts the source image to luma (BT.709) and applies the Sobel operator to
// compute gradient magnitude. The result is stored in the scratch texture:
//
//   .r = edge intensity in [0, 1] (normalised Sobel gradient magnitude)
//   .g = 0 (unused)
//   .b = 0 (unused)
//   .a = source alpha (passed through for the final composite)
//
// A higher Sobel magnitude indicates a stronger contrast edge. The composite
// pass uses this as the basis of a dark multiplication mask.
//
// All WatercolorEdgeParams fields must be declared in every pass to satisfy
// WebGPU's uniform binding-size validation requirement.

struct WatercolorEdgeParams {
    // Blend factor: 0.0 = source unchanged (identity), 1.0 = full dark-edge effect.
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

// Bindings — 1 input: source at binding 0, output at binding 1.
@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: WatercolorEdgeParams;

// BT.709 luma coefficients for perceptually-weighted grayscale conversion.
const LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Applies the Sobel operator at `coord` and returns the gradient vector (gx, gy).
// Uses the 3×3 neighbourhood with coordinates clamped to texture bounds.
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

    let gradient = sobel(coord, dims);

    // Raw Sobel magnitude is nominally in [0, ~4] for normalised [0, 1] luma inputs.
    // Dividing by 4 maps the typical range to approximately [0, 1].
    let magnitude     = length(gradient) / 4.0;
    let edge_intensity = clamp(magnitude, 0.0, 1.0);

    textureStore(dst_texture, coord, vec4<f32>(edge_intensity, 0.0, 0.0, src.a));
}
