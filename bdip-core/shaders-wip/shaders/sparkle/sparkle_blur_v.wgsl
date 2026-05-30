// Sparkle — vertical Gaussian blur pass.
//
// Second half of the separable Gaussian blur over the bright-pixel mask.
// Together with `sparkle_blur_h.wgsl` this approximates a 2D Gaussian spread.
// Parameters and sigma computation are identical to the horizontal pass so that
// the two passes are symmetric.
//
// All Sparkle WGSL files declare the full SparkleParams struct to satisfy
// WebGPU's uniform binding-size validation (see specs/adding_a_shader.md
// § "Shared-uniform alignment rule").

struct SparkleParams {
    threshold: f32,
    intensity: f32,
    radius:    f32,
    _padding:  f32,
}

const RADIUS_CAP: i32 = 512;

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: SparkleParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let short_axis   = f32(min(dims.x, dims.y));
    let sigma        = params.radius * short_axis * 0.1;
    let radius       = min(i32(ceil(3.0 * sigma)), RADIUS_CAP);
    let two_sigma_sq = 2.0 * sigma * sigma;

    let coord = vec2<i32>(gid.xy);
    var accum:      vec4<f32> = vec4<f32>(0.0);
    var weight_sum: f32       = 0.0;

    for (var t: i32 = -radius; t <= radius; t = t + 1) {
        let offset = vec2<i32>(0, t);
        let s = textureLoad(
            input_texture,
            clamp(coord + offset, vec2<i32>(0), vec2<i32>(dims) - 1),
            0,
        );
        let w = exp(-f32(t * t) / max(two_sigma_sq, 0.0001));
        accum      = accum + s * w;
        weight_sum = weight_sum + w;
    }

    let src_alpha = textureLoad(input_texture, coord, 0).a;
    let out = accum / max(weight_sum, 0.0001);
    textureStore(output_texture, coord, vec4<f32>(out.rgb, src_alpha));
}
