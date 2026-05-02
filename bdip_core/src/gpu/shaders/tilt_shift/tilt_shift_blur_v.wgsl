// Tilt-Shift — vertical separable Gaussian blur pass.
//
// Operates on the horizontally blurred 4× downsampled image from Pass 2 and
// applies the vertical component of the separable Gaussian. The sigma calculation
// is identical to the horizontal pass so the two-pass kernel produces a
// circularly symmetric Gaussian when combined.
//
// All five Tilt-Shift WGSL files declare the full TiltShiftParams struct to satisfy
// WebGPU's uniform binding-size validation.

struct TiltShiftParams {
    focus_center:  f32,
    focus_width:   f32,
    blur_strength: f32,
    _padding:      f32,
}

// Bindings — position-indexed (1 input → input at 0, output at 1).
@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: TiltShiftParams;

const SIGMA_FRACTION: f32 = 0.05;
const RADIUS_CAP: i32 = 40;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let sigma = params.blur_strength * SIGMA_FRACTION * f32(dims.y);
    let radius = min(i32(ceil(3.0 * sigma)), RADIUS_CAP);

    // Guard against sigma=0 to avoid NaN in Gaussian weight calculations.
    if radius == 0 {
        let centre = textureLoad(input_texture, vec2<i32>(gid.xy), 0);
        textureStore(output_texture, vec2<i32>(gid.xy), centre);
        return;
    }

    let two_sigma_sq = 2.0 * sigma * sigma;
    var accum:      vec4<f32> = vec4<f32>(0.0);
    var weight_sum: f32       = 0.0;
    let coord = vec2<i32>(gid.xy);

    for (var t: i32 = -radius; t <= radius; t = t + 1) {
        let offset = vec2<i32>(0, t); // vertical tap
        let s = textureLoad(
            input_texture,
            clamp(coord + offset, vec2<i32>(0), vec2<i32>(dims) - 1),
            0,
        );
        let w = exp(-f32(t * t) / two_sigma_sq);
        accum      = accum + s * w;
        weight_sum = weight_sum + w;
    }

    let out = accum / weight_sum;
    // Alpha is copied from the centre pixel — the blur must not smear alpha.
    let src_alpha = textureLoad(input_texture, coord, 0).a;
    textureStore(output_texture, coord, vec4<f32>(out.rgb, src_alpha));
}
