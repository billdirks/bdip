// Candy Color — vibrance-based saturation boost.
//
// Unlike uniform saturation (which scales all channels equally away from luminance),
// vibrance boosts each channel proportional to how far it currently is from being
// fully saturated. Channels that are already vivid receive little to no lift; muted
// channels receive the full boost. This preserves the character of already-vivid hues
// while punching up dull tones.
//
// Algorithm per pixel (in linear light):
//   1. Compute the current pixel saturation as (max(R,G,B) - min(R,G,B)), normalised
//      to [0, 1]. A gray pixel has saturation 0; a fully saturated hue has saturation 1.
//   2. Compute a per-pixel vibrance weight = strength * (1 - saturation). This means
//      gray pixels get the full strength, while a fully saturated pixel gets 0 boost.
//   3. Interpolate each channel from the Rec.709 luminance value toward the original
//      channel value using (1 + vibrance_weight) as the mix factor.
//      At weight=0 this is identical to a mix factor of 1.0 — an identity operation.
//
// Identity: when strength = 0, vibrance_weight = 0, mix factor = 1.0 → no change.

struct CandyColorParams {
    strength:  f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CandyColorParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let color = textureLoad(src_texture, coord, 0);

    // Rec. 709 luminance of the linear-light pixel.
    let lum = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;

    // Current pixel saturation in [0, 1]: 0 for neutral grays, 1 for fully saturated.
    // Using the simple (max - min) metric, which is cheap and sufficient for this purpose.
    let ch_max = max(color.r, max(color.g, color.b));
    let ch_min = min(color.r, min(color.g, color.b));
    let saturation = ch_max - ch_min;

    // Vibrance weight: full strength for desaturated pixels, zero boost for already-vivid ones.
    // Clamped to [0, 1] to guard against out-of-range input values.
    let vibrance_weight = clamp(params.strength, 0.0, 1.0) * (1.0 - saturation);

    // Mix factor > 1.0 pushes each channel further from the luminance gray point.
    // At vibrance_weight=0 this equals 1.0 — a mathematical identity.
    let mix_factor = 1.0 + vibrance_weight;
    let boosted = mix(vec3<f32>(lum), color.rgb, mix_factor);

    // Do NOT clamp — preserve >1.0 headroom for downstream shaders.
    textureStore(dst_texture, coord, vec4<f32>(boosted, color.a));
}
