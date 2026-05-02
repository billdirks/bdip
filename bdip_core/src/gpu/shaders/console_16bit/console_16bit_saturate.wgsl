// 16-bit Console — pass 2: saturation boost and master blend.
//
// Reads both the original source image (binding 0) and the dithered result from
// pass 1 (binding 1). Applies a saturation boost to the dithered image to recreate
// the vivid palette output typical of 16-bit console TV displays, then blends the
// processed result with the original source using `strength` as the mix weight.
//
// Saturation formula (Rec. 709 luminance coefficients, linear light):
//
//   lum = 0.2126·R + 0.7152·G + 0.0722·B
//   sat_scale = 1.0 + saturation_boost
//   out_rgb = mix(vec3(lum), dithered_rgb, sat_scale)
//
// At saturation_boost=0.0, sat_scale=1.0 and the mix reduces to dithered_rgb
// unchanged. At saturation_boost=1.0, sat_scale=2.0, each channel moves twice as
// far from the luminance axis, doubling chroma intensity.
//
// Master blend:
//
//   final_rgb = mix(source_rgb, saturated_rgb, strength)
//
// At strength=0.0 the output equals the source exactly (identity). At strength=1.0
// the full effect is applied. Alpha is taken from the original source throughout.
//
// All passes share the same uniform struct; the full struct is declared here to
// satisfy WebGPU uniform binding-size validation.

struct Console16BitParams {
    color_levels:     f32,
    saturation_boost: f32,
    strength:         f32,
    _padding:         f32,
}

// Binding 0: original source image (for master blend).
@group(0) @binding(0) var source_texture:  texture_2d<f32>;
// Binding 1: dithered result from pass 1.
@group(0) @binding(1) var dither_texture:  texture_2d<f32>;
@group(0) @binding(2) var output_texture:  texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: Console16BitParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(source_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord   = vec2<i32>(gid.xy);
    let src     = textureLoad(source_texture, coord, 0);
    let dithered = textureLoad(dither_texture, coord, 0);

    // Boost saturation of the dithered image using Rec. 709 luminance weights
    // (linear-light coefficients; the pipeline works in linear space).
    let lum       = 0.2126 * dithered.r + 0.7152 * dithered.g + 0.0722 * dithered.b;
    let sat_scale = 1.0 + params.saturation_boost;
    let saturated = mix(vec3<f32>(lum), dithered.rgb, sat_scale);

    // Blend the processed result with the original source image.
    let final_rgb = mix(src.rgb, saturated, params.strength);

    // Alpha is taken from the original source throughout both passes.
    textureStore(output_texture, coord, vec4<f32>(final_rgb, src.a));
}
