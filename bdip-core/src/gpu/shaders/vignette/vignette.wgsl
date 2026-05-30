struct VignetteParams {
    radius: f32,
    softness: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: VignetteParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let d = distance(uv, vec2<f32>(0.5, 0.5));

    // V=1 inside (radius - softness), V=0 outside radius.
    let v = 1.0 - smoothstep(params.radius - params.softness, params.radius, d);

    let color = textureLoad(src_texture, coord, 0);
    textureStore(dst_texture, coord, vec4<f32>(color.rgb * v, color.a));
}
