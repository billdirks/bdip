// Presentation pass: linear light → sRGB-encoded.
//
// Reads an Rgba16Float texture whose RGB channels hold linear-light values
// (the output of the transformation chain) and writes a new Rgba16Float
// texture whose RGB channels hold sRGB-encoded values ready for file export
// or display. Alpha is copied unchanged.
//
// This pass is the last step of every pipeline run before CPU readback.
//
// IMPORTANT: the final textureStore is a single statement at the tail of
// main. PR #2 will replace it with a buffer store; keeping it localized
// makes that swap mechanical. Do not factor the write into a helper.

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

fn linear_to_srgb(c: f32) -> f32 {
    if (c <= 0.0031308) {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);

    let srgb_r = linear_to_srgb(color.r);
    let srgb_g = linear_to_srgb(color.g);
    let srgb_b = linear_to_srgb(color.b);
    let final_color = vec4<f32>(srgb_r, srgb_g, srgb_b, color.a);

    textureStore(dst_texture, coords, final_color);
}
