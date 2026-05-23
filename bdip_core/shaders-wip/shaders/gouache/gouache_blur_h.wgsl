// Gouache — horizontal separable Gaussian blur pass.
//
// This pass softens the image horizontally to begin the smoothing step that
// produces the flat-color, low-detail appearance of gouache paint. The blur
// radius is derived from image dimensions so that the effect scales to any
// resolution without CPU-side parameter changes.
//
// All Gouache WGSL files declare the full GouacheParams struct to satisfy
// WebGPU's uniform binding-size validation requirement: every pass in a
// multi-pass shader shares the same uniform buffer and struct layout.

struct GouacheParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

// Bindings — position-indexed (1 input → input at 0, output at 1).
@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: GouacheParams;

// SIGMA_FRACTION drives the blur radius relative to the image's longer edge.
// 1.5% of the longer dimension gives a noticeable smoothing that flattens fine
// detail without completely destroying structure.
const SIGMA_FRACTION: f32 = 0.015;
const RADIUS_CAP:     i32 = 270;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    // Scale sigma by strength so that at strength=0 sigma→0, giving identity.
    let base_sigma    = SIGMA_FRACTION * f32(max(dims.x, dims.y));
    let sigma         = base_sigma * params.strength;
    let radius        = min(i32(ceil(3.0 * sigma)), RADIUS_CAP);
    let two_sigma_sq  = 2.0 * sigma * sigma;

    let coord = vec2<i32>(gid.xy);

    // When strength=0, sigma=0 and two_sigma_sq=0, which would cause division by
    // zero in the Gaussian weight. In that case return the source pixel unchanged.
    if two_sigma_sq <= 0.0 {
        let src = textureLoad(input_texture, coord, 0);
        textureStore(output_texture, coord, src);
        return;
    }

    var accum:      vec4<f32> = vec4<f32>(0.0);
    var weight_sum: f32       = 0.0;

    for (var t: i32 = -radius; t <= radius; t = t + 1) {
        let s = textureLoad(
            input_texture,
            clamp(coord + vec2<i32>(t, 0), vec2<i32>(0), vec2<i32>(dims) - 1),
            0,
        );
        let w = exp(-f32(t * t) / two_sigma_sq);
        accum      += s * w;
        weight_sum += w;
    }

    let blurred = accum / weight_sum;
    // Alpha is copied from the centre pixel — the blur must not smear alpha.
    let src_alpha = textureLoad(input_texture, coord, 0).a;
    textureStore(output_texture, coord, vec4<f32>(blurred.rgb, src_alpha));
}
