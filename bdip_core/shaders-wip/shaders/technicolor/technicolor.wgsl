// The uniform struct must match the Rust TechnicolorParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would make the struct 32 bytes and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct TechnicolorParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: TechnicolorParams;

// Rec. 709 luminance coefficients (linear light), used to preserve perceived
// brightness when the color matrix shifts channel ratios.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Apply a color matrix that approximates the Technicolor 3-strip dye transfer
// process.  In the original process each primary color was captured on a
// separate film strip through a complementary-color filter, then transferred
// using magenta, cyan, and yellow dyes.  The dye transfer introduced
// cross-channel contamination: each output channel received contributions from
// the other two strips.
//
// The matrix below encodes those cross-channel relationships:
//   - Red channel: boosted with a small green contribution and a small blue
//     subtraction, producing the rich, warm reds characteristic of 1930–50s
//     Technicolor prints.
//   - Green channel: slightly boosted with minor red and blue contributions,
//     reproducing the saturated foliage and costume greens.
//   - Blue channel: desaturated slightly via green bleed, reflecting the
//     historical tendency of cyan dye to absorb some green.
//
// At strength=0 the matrix becomes the 3×3 identity and the shader is a no-op.
fn technicolor_matrix(rgb: vec3<f32>) -> vec3<f32> {
    // Column-major: each row is the output channel's weights over [R, G, B].
    let r_out = 1.3  * rgb.r - 0.1  * rgb.g - 0.05 * rgb.b;
    let g_out = -0.05 * rgb.r + 1.2  * rgb.g + 0.05 * rgb.b;
    let b_out = 0.05 * rgb.r - 0.15 * rgb.g + 0.9  * rgb.b;
    return vec3<f32>(r_out, g_out, b_out);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);
    let rgb   = pixel.rgb;

    // Apply the Technicolor color matrix then blend with the original by strength.
    // At strength=0.0 the result equals rgb (identity).
    let graded = technicolor_matrix(rgb);
    let out_rgb = mix(rgb, graded, params.strength);

    // Do NOT clamp: preserve headroom above 1.0 for downstream shaders.
    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
