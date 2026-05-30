// The uniform struct must match the Rust AntiqueGoldParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would make the struct 32 bytes and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct AntiqueGoldParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: AntiqueGoldParams;

// Rec. 709 luminance coefficients for linear-light RGB.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);
    let rgb   = pixel.rgb;

    let lum = luminance(rgb);

    // Antique Gold color matrix — applied in linear light.
    //
    // The matrix is derived from the W3C sepia matrix and then shifted to produce
    // a warmer, more yellow-gold output. Compared to sepia:
    //   - Red channel is boosted to push highlights toward golden yellow.
    //   - Green channel is tuned to add warmth without going too orange.
    //   - Blue channel is reduced to remove the cool cast.
    //
    // At strength=1.0 the full matrix is applied; at strength=0.0 the output
    // equals the input (identity matrix blended in via mix below).
    //
    // Output may exceed 1.0 for bright inputs — do not clamp, to preserve
    // headroom for downstream shaders.
    let out_r_full = 0.50 * rgb.r + 0.82 * rgb.g + 0.18 * rgb.b;
    let out_g_full = 0.38 * rgb.r + 0.62 * rgb.g + 0.12 * rgb.b;
    let out_b_full = 0.12 * rgb.r + 0.24 * rgb.g + 0.08 * rgb.b;
    let tinted = vec3<f32>(out_r_full, out_g_full, out_b_full);

    // Blend the original pixel toward the tinted result proportionally to strength.
    // At strength=0.0, mix returns rgb unchanged (identity).
    // At strength=1.0, mix returns the fully tinted color.
    let blended = mix(rgb, tinted, params.strength);

    // Do NOT clamp — preserve headroom above 1.0 for downstream shaders.
    textureStore(output_texture, coord, vec4<f32>(blended, pixel.a));
}
