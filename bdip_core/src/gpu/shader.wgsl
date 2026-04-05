@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

// The uniform buffer where we pass our brightness offset
struct TransformParams {
    brightness_offset: f32,
    // Add padding to satisfy WGPU 16-byte alignment rules for uniforms in structs
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(1) @binding(0) var<uniform> params: TransformParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);

    // Apply brightness to rgb, leave alpha alone.
    let new_rgb = color.rgb + vec3<f32>(params.brightness_offset);
    let final_color = vec4<f32>(new_rgb, color.a);

    textureStore(dst_texture, coords, final_color);
}
