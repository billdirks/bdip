struct ThermalParams {
    intensity: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ThermalParams;
@group(2) @binding(0) var gradient_tex: texture_2d<f32>;
@group(2) @binding(1) var gradient_sampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<u32>(gid.xy);
    let color = textureLoad(src_texture, coord, 0);
    let luma = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Sample the gradient texture at (luma, 0.5) — horizontal axis is brightness.
    let thermal = textureSampleLevel(
        gradient_tex, gradient_sampler, vec2<f32>(clamp(luma, 0.0, 1.0), 0.5), 0.0
    ).rgb;

    let out = mix(color.rgb, thermal, params.intensity);
    textureStore(dst_texture, coord, vec4<f32>(out, color.a));
}
