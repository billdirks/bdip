// Daguerreotype — Pass 0: colour processing
//
// Converts the source to a high-contrast silver-grey tone with a slight blue-grey
// metallic sheen.  The result is stored in a scratch texture consumed by Pass 1.
//
// Processing steps:
//   1. Luminance-weighted desaturation (Rec. 709 coefficients).
//   2. S-curve contrast boost that lifts blacks and crushes highlights — this
//      approximates the harsh tonal compression typical of silver-salt emulsions.
//   3. Metallic tint: a subtle shift that adds more blue to the grey point and
//      a very slight green lift.  The resulting neutral grey trends blue-grey
//      (cool metallic) rather than warm sepia.

struct DaguerreotypeParams {
    strength: f32,
    _pad0:    f32,
    _pad1:    f32,
    _pad2:    f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture:  texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform>      params: DaguerreotypeParams;

// Rec. 709 luminance coefficients.
const LUM_R: f32 = 0.2126;
const LUM_G: f32 = 0.7152;
const LUM_B: f32 = 0.0722;

// S-curve contrast: maps linear [0,1] to a steeper [0,1].
// Uses a simple cubic Hermite blend between an identity and a stronger curve.
fn s_curve(x: f32) -> f32 {
    // Cubic smoothstep-derived curve: 3x² - 2x³ further composed with a boost.
    // Remap 0→0, 0.5→0.5, 1→1 but with steeper flanks.
    let t = clamp(x, 0.0, 1.0);
    // Two-segment contrast: below 0.5 pull down, above 0.5 push up.
    let boosted = t * t * (3.0 - 2.0 * t);
    // Mix between original and boosted to allow partial application.
    return mix(t, boosted, 0.65);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let src   = textureLoad(src_texture, coord, 0);

    // 1. Desaturate to luminance.
    let luma = src.r * LUM_R + src.g * LUM_G + src.b * LUM_B;

    // 2. Apply S-curve contrast boost to the luminance value.
    let contrasted = s_curve(luma);

    // 3. Metallic blue-grey tint.
    //    The tint matrix nudges a neutral grey toward silver-blue:
    //      R ← luma * 0.94  (slight warm-tone reduction)
    //      G ← luma * 0.97  (near-neutral)
    //      B ← luma * 1.06  (slight blue lift for metallic sheen)
    let silver_r = contrasted * 0.94;
    let silver_g = contrasted * 0.97;
    let silver_b = contrasted * 1.06;

    // The toned result is written to scratch regardless of strength; the blend
    // with the original source is applied in Pass 1.
    let toned = vec4<f32>(silver_r, silver_g, silver_b, src.a);
    textureStore(dst_texture, coord, toned);
}
