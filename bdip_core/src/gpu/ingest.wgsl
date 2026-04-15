// Ingest pass: sRGB-encoded → linear light.
//
// Reads an Rgba16Unorm texture whose RGB channels hold sRGB-normalized u16
// values uploaded raw from the CPU. The GPU hardware normalizes the u16 values
// to [0.0, 1.0] on textureLoad, so the shader receives the same [0,1] float
// range it did when the texture was Rgba16Float. Writes a new Rgba16Float
// texture whose RGB channels hold the corresponding linear-light values.
// Alpha is copied unchanged.
//
// This pass is the first step of every pipeline run. All transformation shaders
// that follow operate on linear data and have no knowledge of gamma encoding.

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);

    let linear_r = srgb_to_linear(color.r);
    let linear_g = srgb_to_linear(color.g);
    let linear_b = srgb_to_linear(color.b);

    textureStore(dst_texture, coords, vec4<f32>(linear_r, linear_g, linear_b, color.a));
}
