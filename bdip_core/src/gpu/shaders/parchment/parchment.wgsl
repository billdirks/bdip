struct ParchmentParams {
    intensity: f32,
    scale: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ParchmentParams;
@group(2) @binding(0) var paper_tex: texture_2d<f32>;
@group(2) @binding(1) var paper_sampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<u32>(gid.xy);
    let color = textureLoad(src_texture, coord, 0);

    // Tile the paper texture across the image at the given scale.
    let paper_dims = vec2<f32>(textureDimensions(paper_tex));
    let uv = fract(vec2<f32>(gid.xy) / (paper_dims * params.scale));
    let paper = textureSampleLevel(paper_tex, paper_sampler, uv, 0.0).rgb;

    // Multiplicative blend: paper grain darkens/tones the image where grain is present.
    let parchment = color.rgb * paper;
    let out = mix(color.rgb, parchment, params.intensity);
    textureStore(dst_texture, coord, vec4<f32>(out, color.a));
}
