@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

// Dummy uniform — no user-facing parameters, but the bind group layout requires
// a params group (Group 1, Binding 0) to satisfy the shared pipeline contract.
struct SepiaParams {
    _unused: vec4<f32>,
}

@group(1) @binding(0) var<uniform> params: SepiaParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);
    let r = color.r;
    let g = color.g;
    let b = color.b;

    // Standard sepia color matrix applied in linear light.
    // Matrix coefficients derived from the W3C filter specification
    // (https://www.w3.org/TR/filter-effects/#feColorMatrixElement, sepia(1)).
    // Output may exceed 1.0 for bright inputs; do not clamp to preserve headroom
    // for downstream shaders.
    let out_r = 0.393 * r + 0.769 * g + 0.189 * b;
    let out_g = 0.349 * r + 0.686 * g + 0.168 * b;
    let out_b = 0.272 * r + 0.534 * g + 0.131 * b;

    textureStore(dst_texture, coords, vec4<f32>(out_r, out_g, out_b, color.a));
}
