@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

// Dummy uniform — no user-facing parameters, but the bind group layout requires
// a params group (Group 1, Binding 0) to satisfy the shared pipeline contract.
struct GrayscaleParams {
    _unused: vec4<f32>,
}

@group(1) @binding(0) var<uniform> params: GrayscaleParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);

    // ITU-R BT.709 luminance coefficients, correct for linear-light values.
    let luminance = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
    let final_color = vec4<f32>(luminance, luminance, luminance, color.a);

    textureStore(dst_texture, coords, final_color);
}
