struct HighKeyParams {
    strength: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: HighKeyParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    let color = textureLoad(src_texture, coord, 0);

    // High-key effect: two operations scaled by strength.
    //
    // 1. Exposure boost — multiply by 2^(2*strength) to lift overall brightness
    //    (each +1.0 stop doubles the light; at strength=1 we boost by ~2 stops).
    // 2. Shadow lift — add a linear floor that lifts dark areas toward white.
    //    The floor is proportional to (1 - channel value), so it has maximum
    //    effect on pure blacks and no effect on whites (preserving highlights).
    //
    // When strength=0 the scale is 2^0=1 and the floor is 0, giving identity.
    let scale = pow(2.0, 2.0 * params.strength);
    let floor = params.strength * 0.3;

    // Apply exposure boost first, then add the lifted floor.
    // Do NOT clamp to preserve >1.0 headroom for downstream shaders.
    let out_rgb = color.rgb * scale + floor * (1.0 - color.rgb);

    textureStore(dst_texture, coord, vec4<f32>(out_rgb, color.a));
}
