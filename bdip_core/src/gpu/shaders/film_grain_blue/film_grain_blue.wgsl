struct FilmGrainBlueParams {
    amount: f32,
    variation: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: FilmGrainBlueParams;
@group(2) @binding(0) var noise_texture: texture_2d<f32>;
@group(2) @binding(1) var noise_sampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<u32>(global_id.xy);
    let color = textureLoad(src_texture, coord, 0);

    // Derive a 2D UV offset from the variation parameter to reshuffle the tiled
    // pattern without re-uploading the texture.
    let var_x = fract(params.variation * 12.9898);
    let var_y = fract(params.variation * 78.233);
    let variation_offset = vec2<f32>(var_x, var_y);

    // Tile the 128×128 blue noise across the image and apply the variation offset.
    let uv = fract(vec2<f32>(global_id.xy) / 128.0 + variation_offset);
    let noise_val = textureSampleLevel(noise_texture, noise_sampler, uv, 0.0).r;

    // Center [0, 1] → [-0.5, 0.5].
    let n = noise_val - 0.5;

    // Rec. 709 luma-weighted grain — grain is more visible in midtones/shadows,
    // matching film emulsion behavior.
    let luma = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let weight = sqrt(max(luma, 0.0));

    let out_rgb = color.rgb + vec3<f32>(n * params.amount * weight);
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, color.a));
}
