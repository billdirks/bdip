@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

// Dummy uniform — no user-facing parameters, but the bind group layout requires
// a params group (Group 1, Binding 0) to satisfy the shared pipeline contract.
struct XRayParams {
    _unused: vec4<f32>,
}

@group(1) @binding(0) var<uniform> params: XRayParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);

    // Step 1: Invert the RGB channels in linear light (1 − value).
    let inverted = vec3<f32>(1.0 - color.r, 1.0 - color.g, 1.0 - color.b);

    // Step 2: Convert the inverted values to grayscale using ITU-R BT.709
    // luminance coefficients, which are correct for linear-light values.
    let luminance = 0.2126 * inverted.r + 0.7152 * inverted.g + 0.0722 * inverted.b;

    // Step 3: Apply high contrast by raising the luminance to a power of 2.0.
    // Gamma-style exponentiation in linear light compresses midtones and
    // preserves highlights, producing the stark light-on-dark appearance
    // characteristic of X-ray imaging. Values stay in [0, 1] so no clamping
    // is necessary for typical images.
    let contrasted = luminance * luminance;

    let final_color = vec4<f32>(contrasted, contrasted, contrasted, color.a);

    textureStore(dst_texture, coords, final_color);
}
