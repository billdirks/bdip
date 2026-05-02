struct RetroGameBoyParams {
    // Controls blending between fully quantized Game Boy palette (1.0) and original
    // grayscale (0.0). Default 0.0 is identity (no change); 1.0 applies the full effect.
    palette_intensity: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: RetroGameBoyParams;

// The original DMG Game Boy screen used a 4-shade pea-green/olive palette.
// These are the four shades expressed in linear light, converted from the
// canonical DMG sRGB values: #0F380F, #306230, #8BAC0F, #9BBC0F.
//
// Quantization maps linear luma to one of four equal-width buckets [0, 0.25),
// [0.25, 0.5), [0.5, 0.75), [0.75, 1.0], then replaces each bucket with the
// corresponding palette colour. Bucket midpoints are used as representative
// luma values for deriving expected test outputs.
fn game_boy_palette(level: u32) -> vec3<f32> {
    // Level 0 (darkest):  sRGB #0F380F → linear (0.0048, 0.0395, 0.0048)
    // Level 1:            sRGB #306230 → linear (0.0296, 0.1221, 0.0296)
    // Level 2:            sRGB #8BAC0F → linear (0.2582, 0.4125, 0.0048)
    // Level 3 (lightest): sRGB #9BBC0F → linear (0.3278, 0.5029, 0.0048)
    if level == 0u {
        return vec3<f32>(0.0048, 0.0395, 0.0048);
    } else if level == 1u {
        return vec3<f32>(0.0296, 0.1221, 0.0296);
    } else if level == 2u {
        return vec3<f32>(0.2582, 0.4125, 0.0048);
    } else {
        return vec3<f32>(0.3278, 0.5029, 0.0048);
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);

    // Compute luma using Rec. 709 coefficients (standard for linear-light images).
    let luma = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;

    // Quantize to 4 levels by dividing [0, 1] into four equal-width buckets.
    // clamp guards against out-of-range luma values in HDR headroom content.
    let clamped = clamp(luma, 0.0, 1.0);
    let level = u32(min(floor(clamped * 4.0), 3.0));
    let palette_color = game_boy_palette(level);

    // Blend between the original grayscale and the quantized palette colour.
    // When palette_intensity is 0.0 (default), this is an identity transformation.
    // When palette_intensity is 1.0, the full 4-shade pea-green palette is applied.
    let gray = vec3<f32>(luma);
    let out_rgb = mix(gray, palette_color, params.palette_intensity);

    // Do not clamp — preserve >1.0 headroom for downstream shaders.
    textureStore(dst_texture, coords, vec4<f32>(out_rgb, color.a));
}
