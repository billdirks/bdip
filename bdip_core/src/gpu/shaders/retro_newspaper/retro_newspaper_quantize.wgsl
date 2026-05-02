// Retro Newspaper — Pass 2: Tonal quantisation.
//
// Reduces the grayscale image to a fixed number of discrete gray levels (5),
// simulating the limited tonal range of letterpress or offset newspaper printing.
// The quantised value is written back as R=G=B=quantised_luma to the scratch
// texture for the halftone pass.
//
// Five tonal levels are used, matching the typical newspaper reproduction range:
// black, dark gray, mid gray, light gray, and white.
//
// All RetroNewspaperParams fields must be declared in every pass.

struct RetroNewspaperParams {
    dot_frequency: f32,
    strength:      f32,
    _padding:      vec2<f32>,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: RetroNewspaperParams;

// Number of quantisation levels. 5 levels gives a newspaper-like tonal range.
const LEVELS: f32 = 5.0;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(src_texture, coord, 0);

    // Quantise to LEVELS discrete steps.
    let quantised = floor(src.r * LEVELS + 0.5) / LEVELS;
    textureStore(dst_texture, coord, vec4<f32>(quantised, quantised, quantised, src.a));
}
