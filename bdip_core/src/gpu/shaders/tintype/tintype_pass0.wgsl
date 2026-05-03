// Tintype — Pass 0: desaturation, high contrast, and warm metallic tint
//
// Converts the source to a near-monochrome image with the dark pewter/steel-grey
// tonal character of Civil War era tintype photography.  The result is stored in a
// scratch texture consumed by Pass 1 (vignette) and Pass 2 (grit overlay).
//
// Processing steps:
//   1. Luminance-weighted desaturation (Rec. 709), with a residual colour bleed
//      of 8% to retain a very faint warm colour cast before tinting.
//   2. S-curve contrast boost: pulls shadows toward dark pewter, pushes highlights
//      toward dull silver — a characteristic of uneven iron plate exposure.
//   3. Warm metallic tint: unlike the blue-grey of daguerreotypes, tintypes have a
//      slightly warm, dark pewter-grey tone — R is boosted more than B, giving
//      a subtle dark-steel warmth rather than a cool silver.

struct TintypeParams {
    strength: f32,
    _pad0:    f32,
    _pad1:    f32,
    _pad2:    f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform>     params: TintypeParams;

// Rec. 709 luminance coefficients.
const LUM_R: f32 = 0.2126;
const LUM_G: f32 = 0.7152;
const LUM_B: f32 = 0.0722;

// S-curve contrast: steeper midtone separation than Daguerreotype to simulate
// the harsh tonal compression of iron-backed photographic emulsions.
fn s_curve(x: f32) -> f32 {
    let t = clamp(x, 0.0, 1.0);
    // Smoothstep-derived cubic: 3t² - 2t³ produces the S-shaped tonal curve.
    let smoothed = t * t * (3.0 - 2.0 * t);
    // Blend toward the steeper curve at 75% weight for high contrast.
    return mix(t, smoothed, 0.75);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let src   = textureLoad(src_texture, coord, 0);

    // 1. Near-complete desaturation: 92% luma + 8% residual colour.
    let luma    = src.r * LUM_R + src.g * LUM_G + src.b * LUM_B;
    let desat_r = mix(luma, src.r, 0.08);
    let desat_g = mix(luma, src.g, 0.08);
    let desat_b = mix(luma, src.b, 0.08);

    // 2. High-contrast S-curve applied to the luminance level.
    let luma_contrasted = s_curve(luma);
    // Apply the same contrast ratio to all channels uniformly.
    let contrast_ratio  = select(luma_contrasted / luma, 1.0, luma < 0.001);
    let contrasted_r    = desat_r * contrast_ratio;
    let contrasted_g    = desat_g * contrast_ratio;
    let contrasted_b    = desat_b * contrast_ratio;

    // 3. Warm dark pewter/steel-grey tint characteristic of iron-plate tintypes.
    //    The tint matrix gives R > G > B for a dark warm-grey (opposite of the
    //    cool blue-grey of daguerreotypes):
    //      R ← value * 1.04  (slight warm red-brown lift)
    //      G ← value * 1.00  (neutral green)
    //      B ← value * 0.92  (slight blue suppression for pewter warmth)
    let tinted_r = contrasted_r * 1.04;
    let tinted_g = contrasted_g * 1.00;
    let tinted_b = contrasted_b * 0.92;

    // Write toned result to scratch — blend with source is handled in Pass 2.
    textureStore(dst_texture, coord, vec4<f32>(tinted_r, tinted_g, tinted_b, src.a));
}
