// Cartoon — bilateral filter pass (edge-preserving smoothing).
//
// A bilateral filter weights each neighbor by both spatial distance and intensity
// similarity. This flattens color within regions while preserving sharp edges,
// producing the flat-shaded areas that define cartoon/cel-shading aesthetics.
//
// The spatial sigma is derived from image size (3% of the longer dimension, capped
// at 15 px). The range sigma is controlled by `params.smoothing`: low values make the
// filter selective (only averaging pixels with very similar intensity), high values
// allow averaging across larger intensity differences.
//
// All four Cartoon WGSL files declare the full CartoonParams struct to satisfy
// WebGPU's uniform binding-size validation (see specs/multi-pass-plan.md
// § "Bind-group contract (multi-pass passes)").

struct CartoonParams {
    strength:       f32,
    levels:         f32,
    smoothing:      f32,
    edge_threshold: f32,
    edge_softness:  f32,
    edge_darkness:  f32,
    _padding0:      f32,
    _padding1:      f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CartoonParams;

const SIGMA_SPATIAL_FRACTION: f32 = 0.03;
const RADIUS_CAP: i32 = 15;

fn luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let center = textureLoad(input_texture, coord, 0);
    let center_luma = luma(center.rgb);

    let sigma_s = SIGMA_SPATIAL_FRACTION * f32(max(dims.x, dims.y));
    let radius = min(i32(ceil(3.0 * sigma_s)), RADIUS_CAP);
    let two_sigma_s_sq = 2.0 * sigma_s * sigma_s;

    // smoothing=0 → sigma_r=0.01 (near-identity), smoothing=1 → sigma_r=0.5 (aggressive).
    let sigma_r = mix(0.01, 0.5, params.smoothing);
    let two_sigma_r_sq = 2.0 * sigma_r * sigma_r;

    var accum = vec3<f32>(0.0);
    var weight_sum = 0.0;

    for (var dy: i32 = -radius; dy <= radius; dy = dy + 1) {
        for (var dx: i32 = -radius; dx <= radius; dx = dx + 1) {
            let sample_coord = clamp(coord + vec2<i32>(dx, dy), vec2<i32>(0), vec2<i32>(dims) - 1);
            let sample = textureLoad(input_texture, sample_coord, 0);
            let sample_luma = luma(sample.rgb);

            let dist_sq = f32(dx * dx + dy * dy);
            let w_spatial = exp(-dist_sq / two_sigma_s_sq);

            let diff = sample_luma - center_luma;
            let w_range = exp(-(diff * diff) / two_sigma_r_sq);

            let w = w_spatial * w_range;
            accum = accum + sample.rgb * w;
            weight_sum = weight_sum + w;
        }
    }

    let filtered = accum / weight_sum;
    textureStore(output_texture, coord, vec4<f32>(filtered, center.a));
}
