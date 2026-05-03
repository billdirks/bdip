// Charcoal Sketch — Pass 1: Sobel edge detection with paper-tone inversion.
//
// Converts the source image to luminance and applies the Sobel operator to
// compute gradient magnitude. The result is inverted so that:
//   - strong edges  → dark values (black charcoal strokes)
//   - flat regions  → near-1.0  (light cream paper background)
//
// Unlike Pencil Sketch, no gradient direction is stored; charcoal strokes are
// broader and isotropic, so directional blur is not used. The texture roughness
// is instead added by procedural grain in pass 2.
//
// The scratch texture stores:
//   .r = inverted edge intensity in [0, 1]  (1 = paper, 0 = dark charcoal stroke)
//   .g = 0 (unused)
//   .b = 0 (unused)
//   .a = source alpha (passed through for the final composite)
//
// All CharcoalSketchParams fields must be declared in every pass to satisfy
// WebGPU's uniform binding-size validation requirement.

struct CharcoalSketchParams {
    // Blend factor: 0.0 = source unchanged (identity), 1.0 = full effect.
    strength:      f32,
    // Multiplier on raw Sobel magnitude. Higher values amplify faint edges
    // into visible charcoal strokes. Range [0.1, 10.0].
    edge_strength: f32,
    // Amplitude of the procedural grain noise. Range [0.0, 1.0].
    // Used in pass 2; declared here for uniform-buffer size parity.
    grain_amount:  f32,
    _padding:      f32,
}

// Bindings — 1 input: source at binding 0, output at binding 1.
@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CharcoalSketchParams;

// BT.709 luma coefficients for perceptually-weighted grayscale conversion.
const LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Applies the Sobel operator at `coord` and returns gradient magnitude.
// Raw Sobel magnitude on normalised [0,1] inputs is nominally in [0, ~4√2].
// Dividing by 4 in the caller maps typical values to roughly [0, 1].
fn sobel_magnitude(coord: vec2<i32>, dims: vec2<u32>) -> f32 {
    // Sample the 3×3 neighbourhood, clamping to texture bounds.
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
    return length(vec2<f32>(gx, gy));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(src_texture, coord, 0);

    // Raw Sobel magnitude is nominally in [0, ~4] for normalised inputs.
    // Dividing by 4 puts it near [0, 1], then edge_strength adjusts sensitivity.
    let mag            = sobel_magnitude(coord, dims);
    let stroke         = clamp(mag * (params.edge_strength / 4.0), 0.0, 1.0);

    // Invert: strong edge → near 0 (dark charcoal stroke); flat → near 1 (paper).
    let paper_value = 1.0 - stroke;

    // Store inverted edge intensity (.r) and preserve source alpha (.a).
    textureStore(dst_texture, coord, vec4<f32>(paper_value, 0.0, 0.0, src.a));
}
