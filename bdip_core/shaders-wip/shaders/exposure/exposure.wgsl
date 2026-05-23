struct ExposureParams {
    exposure: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ExposureParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    let color = textureLoad(src_texture, coord, 0);

    // Multiply by 2^exposure — each +1.0 stop doubles the light.
    let scale = pow(2.0, params.exposure);
    let out_rgb = clamp(color.rgb * scale, vec3<f32>(0.0), vec3<f32>(1.0));

    textureStore(dst_texture, coord, vec4<f32>(out_rgb, color.a));
}
