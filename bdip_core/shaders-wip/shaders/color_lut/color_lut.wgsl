struct ColorLutParams {
    intensity: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ColorLutParams;
@group(2) @binding(0) var lut_texture: texture_3d<f32>;
@group(2) @binding(1) var lut_sampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<u32>(global_id.xy);
    let color = textureLoad(src_texture, coord, 0);

    // LUT is authored in sRGB space — convert linear → sRGB, sample, convert back.
    let srgb = pow(clamp(color.rgb, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));

    // Scale and offset to sample from cell centers (half-texel inset) to avoid
    // clamping artifacts at the 0 and 1 boundaries.
    let lut_size = f32(textureDimensions(lut_texture).x);
    let scale = (lut_size - 1.0) / lut_size;
    let offset = 0.5 / lut_size;
    let lut_coord = srgb * scale + offset;

    let graded_srgb = textureSampleLevel(lut_texture, lut_sampler, lut_coord, 0.0).rgb;
    let graded_linear = pow(graded_srgb, vec3(2.2));

    let out = mix(color.rgb, graded_linear, params.intensity);
    textureStore(dst_texture, coord, vec4(out, color.a));
}
