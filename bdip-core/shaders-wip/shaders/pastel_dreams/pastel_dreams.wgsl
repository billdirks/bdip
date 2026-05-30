@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

struct PastelDreamsParams {
    // Controls overall intensity of the effect.
    //   strength = 0.0  → identity (no change)
    //   strength = 1.0  → full pastel: shadows/midtones lifted toward white,
    //                     saturation reduced to near-zero
    strength: f32,
    // Padding to satisfy WebGPU 16-byte alignment rules for uniforms in structs.
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(1) @binding(0) var<uniform> params: PastelDreamsParams;

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

    // --- Brightness lift ---
    // Mix the original RGB toward 1.0 (white) proportional to strength.
    // This lifts shadows and midtones without clipping highlights already near white.
    // The lift is uniform across channels, preserving hue while pushing toward light.
    let brightness_lift = params.strength * 0.5;
    let lifted_rgb = color.rgb + vec3<f32>(brightness_lift);

    // --- Saturation reduction ---
    // Desaturate by interpolating toward the luminance value, scaled by strength.
    // At strength=0 scale=1.0 (identity); at strength=1 scale=0 (full grayscale).
    let luma_clamped = clamp(luminance + brightness_lift, 0.0, 1.0);
    let sat_scale = 1.0 - params.strength;
    let desaturated_rgb = mix(vec3<f32>(luma_clamped), lifted_rgb, sat_scale);

    textureStore(dst_texture, coords, vec4<f32>(desaturated_rgb, color.a));
}
