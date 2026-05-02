struct LowKeyParams {
    strength: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: LowKeyParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    let color = textureLoad(src_texture, coord, 0);

    // Low-key effect: two operations scaled by strength.
    //
    // 1. Exposure drop — multiply by 2^(-2*strength) to darken the image.
    //    At strength=1 this is a -2 stop drop (factor of 0.25).
    //    At strength=0 the factor is 1.0, giving identity.
    //
    // 2. Contrast boost — scale linearly around the post-exposure midpoint (0.25).
    //    scale = 1 + 2*strength means scale=1 at strength=0 (identity) and
    //    scale=3 at strength=1. Shadows below the midpoint are pushed to black
    //    while highlights above it retain some brightness.
    //
    // The contrast midpoint of 0.25 is chosen to sit at the expected linear
    // luminance after a -2 stop exposure drop of a pure 50% gray input,
    // so mid-tones crush to black at full strength while the brightest
    // highlights remain visible.
    //
    // When strength=0: exp_scale=1, contrast_scale=1, midpoint irrelevant
    // — the result is exactly the input (identity).
    //
    // Do NOT clamp to preserve >1.0 headroom for downstream shaders.
    let exp_scale = pow(2.0, -2.0 * params.strength);
    let contrast_scale = 1.0 + 2.0 * params.strength;
    let midpoint = 0.25;

    let darkened = color.rgb * exp_scale;
    let out_rgb = (darkened - vec3<f32>(midpoint)) * contrast_scale + vec3<f32>(midpoint);

    textureStore(dst_texture, coord, vec4<f32>(out_rgb, color.a));
}
