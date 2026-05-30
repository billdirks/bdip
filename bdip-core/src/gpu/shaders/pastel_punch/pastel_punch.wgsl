@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

struct PastelPunchParams {
    // Controls the strength of the pastel effect.
    //   strength = 0.0  → identity (no change)
    //   strength = 1.0  → full pastel: saturates the luminance-driven blend toward white
    strength: f32,
    // Padding to satisfy WebGPU 16-byte alignment rules for uniforms in structs.
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(1) @binding(0) var<uniform> params: PastelPunchParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);

    // Compute luminance using Rec. 709 coefficients (linear light).
    let luminance = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;

    // The blend factor mixes toward white based on luminance and strength.
    // Brighter pixels get a higher blend factor, pushing them more toward white,
    // which reduces saturation while lifting brightness — the pastel look.
    // Clamping luminance guards against out-of-range linear values in the pipeline.
    let luma_clamped = clamp(luminance, 0.0, 1.0);
    let blend = luma_clamped * params.strength;

    // Mix original color toward white (1, 1, 1). When strength is 0 the blend
    // factor is 0 throughout and the output equals the input (identity).
    let white = vec3<f32>(1.0, 1.0, 1.0);
    let new_rgb = mix(color.rgb, white, blend);

    textureStore(dst_texture, coords, vec4<f32>(new_rgb, color.a));
}
