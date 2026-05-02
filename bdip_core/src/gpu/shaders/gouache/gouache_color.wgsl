// Gouache — color pass.
//
// Combines the original source with the smoothed image and applies a saturation
// boost to simulate the opaque, high-chroma appearance of gouache paint.
//
// The formula is:
//   C_smooth = lerp(C_src, C_blurred, strength)   — flattens fine detail
//   luma      = dot(C_smooth, Rec.709 weights)
//   C_out     = lerp(luma, C_smooth, 1 + sat_boost) — boosts saturation
//
// where sat_boost = strength * MAX_SAT_BOOST.
//
// At strength=0: C_smooth = C_src and sat_boost=0, so C_out = C_src (identity).
//
// All Gouache WGSL files declare the full GouacheParams struct to satisfy
// WebGPU's uniform binding-size validation requirement.

struct GouacheParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

// Bindings — position-indexed (2 inputs → inputs at 0 and 1, output at 2).
@group(0) @binding(0) var input_source:  texture_2d<f32>;
@group(0) @binding(1) var input_blurred: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: GouacheParams;

// Maximum saturation multiplier added at full strength. A value of 0.6 gives a
// visually distinct but not garish boost when strength=1.0.
const MAX_SAT_BOOST: f32 = 0.6;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord   = vec2<i32>(gid.xy);
    let src     = textureLoad(input_source,  coord, 0);
    let blurred = textureLoad(input_blurred, coord, 0);

    // Step 1: blend towards the smoothed colour to flatten fine detail.
    // At strength=0 this reduces to src.rgb unchanged.
    let c_smooth = mix(src.rgb, blurred.rgb, params.strength);

    // Step 2: boost saturation of the smoothed result.
    // scale = 1 + sat_boost; at strength=0, scale=1 → identity.
    let luma      = dot(c_smooth, vec3<f32>(0.2126, 0.7152, 0.0722));
    let sat_boost = params.strength * MAX_SAT_BOOST;
    let c_sat     = mix(vec3<f32>(luma), c_smooth, 1.0 + sat_boost);

    // Clamp to [0, 1] to keep values in a displayable range after saturation boost.
    let out_rgb = clamp(c_sat, vec3<f32>(0.0), vec3<f32>(1.0));

    textureStore(output_texture, coord, vec4<f32>(out_rgb, src.a));
}
