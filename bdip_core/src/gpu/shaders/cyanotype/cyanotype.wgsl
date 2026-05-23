@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

// Dummy uniform — no user-facing parameters, but the bind group layout requires
// a params group (Group 1, Binding 0) to satisfy the shared pipeline contract.
struct CyanotypeParams {
    _unused: vec4<f32>,
}

@group(1) @binding(0) var<uniform> params: CyanotypeParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);

    // Convert to luminance using Rec. 709 weights (standard for linear-light images).
    let luma = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;

    // Map luminance onto the cyanotype Prussian-blue/cyan palette.
    // Shadows: deep blue (0.0, 0.05, 0.2); highlights: pale blue-white (0.85, 0.93, 1.0).
    // Linear interpolation from dark to light preserves the characteristic gradient.
    // Output may exceed 1.0 for bright inputs; do not clamp to preserve downstream headroom.
    let shadow    = vec3<f32>(0.0,  0.05, 0.20);
    let highlight = vec3<f32>(0.85, 0.93, 1.0);
    let out_rgb = mix(shadow, highlight, luma);

    textureStore(dst_texture, coords, vec4<f32>(out_rgb, color.a));
}
