// Retro Newspaper — Pass 1: Grayscale conversion.
//
// Converts the linear-light source image to luminance using BT.709 coefficients.
// The result is stored as a grayscale rgba16float scratch texture (R=G=B=luma,
// A=source alpha) for use by the quantisation and halftone passes.
//
// All RetroNewspaperParams fields must be declared in every pass to satisfy
// WebGPU's uniform binding-size validation requirement.

struct RetroNewspaperParams {
    dot_frequency: f32,
    strength:      f32,
    _padding:      vec2<f32>,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: RetroNewspaperParams;

// BT.709 luma coefficients for perceptually-weighted grayscale conversion.
const LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(src_texture, coord, 0);

    let luma = dot(src.rgb, LUMA_WEIGHTS);
    textureStore(dst_texture, coord, vec4<f32>(luma, luma, luma, src.a));
}
