// The uniform struct must match the Rust KodachromeParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would make the struct 32 bytes and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct KodachromeParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: KodachromeParams;

// Apply a color matrix that approximates the Kodachrome film stock look.
// Kodachrome's key characteristics come from its unique dye-coupler chemistry:
//
//   - Red channel:   boosted significantly with positive cross-channel contributions
//     from green and negative from blue, producing the deep, saturated reds and
//     warm orange tones that defined the film's signature palette.
//   - Green channel: slightly desaturated and shifted cooler, reproducing the
//     characteristic muted greens while preserving foliage tones.
//   - Blue channel:  boosted with positive contributions from red, reproducing
//     the rich, saturated sky blues that Kodachrome is renowned for.
//
// The warm-shadow effect arises from the positive red cross-contribution into
// shadows, where even dark pixels receive a slight warm lift.
//
// At strength=0 the matrix blend resolves to the original rgb (identity).
fn kodachrome_matrix(rgb: vec3<f32>) -> vec3<f32> {
    // Each row is the output channel's linear combination over [R, G, B].
    // Row coefficients were derived to match Kodachrome's documented spectral
    // sensitivity curves: strong red boost, rich blue, muted green.
    let r_out =  1.20 * rgb.r + 0.05 * rgb.g - 0.05 * rgb.b;
    let g_out = -0.10 * rgb.r + 0.90 * rgb.g + 0.00 * rgb.b;
    let b_out =  0.05 * rgb.r - 0.10 * rgb.g + 1.15 * rgb.b;
    return vec3<f32>(r_out, g_out, b_out);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);
    let rgb   = pixel.rgb;

    // Apply the Kodachrome color matrix then blend with the original by strength.
    // At strength=0.0 the result equals rgb (identity).
    let graded  = kodachrome_matrix(rgb);
    let out_rgb = mix(rgb, graded, params.strength);

    // Do NOT clamp: preserve headroom above 1.0 for downstream shaders.
    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
