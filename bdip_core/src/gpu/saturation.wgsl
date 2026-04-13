@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

struct SaturationParams {
    saturation_offset: f32,
    // Padding to satisfy WebGPU 16-byte alignment rules for uniforms in structs.
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(1) @binding(0) var<uniform> params: SaturationParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);

    // Compute luminance using Rec. 709 coefficients (linear light).
    let luminance = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;

    // Interpolate between gray (luminance) and the original color.
    //   saturation_offset = 0.0  → scale = 1.0 → unchanged
    //   saturation_offset = -1.0 → scale = 0.0 → full grayscale
    //   saturation_offset =  1.0 → scale = 2.0 → double saturation
    let new_rgb = mix(vec3<f32>(luminance), color.rgb, 1.0 + params.saturation_offset);
    let final_color = vec4<f32>(new_rgb, color.a);

    textureStore(dst_texture, coords, final_color);
}
