// Clarity — horizontal separable Gaussian blur pass.
//
// All three Clarity WGSL files declare the full ClarityParams struct to satisfy
// WebGPU's uniform binding-size validation (see specs/multi-pass-plan.md
// § "Bind-group contract (multi-pass passes)").

struct ClarityParams {
    amount:   f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

// Bindings — position-indexed (1 input → input at 0, output at 1).
@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ClarityParams;

// Sigma is derived from image dimensions so the CPU-side ClarityParams struct
// needs only the user-facing `amount` field. The 2% diagonal rule is the
// canonical Clarity blur radius; RADIUS_CAP prevents extreme images from
// ballooning the kernel loop.
const SIGMA_FRACTION: f32 = 0.02;
const RADIUS_CAP: i32 = 360;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let sigma        = SIGMA_FRACTION * f32(max(dims.x, dims.y));
    let radius       = min(i32(ceil(3.0 * sigma)), RADIUS_CAP);
    let two_sigma_sq = 2.0 * sigma * sigma;

    var accum:      vec4<f32> = vec4<f32>(0.0);
    var weight_sum: f32       = 0.0;
    let coord = vec2<i32>(gid.xy);

    for (var t: i32 = -radius; t <= radius; t = t + 1) {
        let offset = vec2<i32>(t, 0); // horizontal tap
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
