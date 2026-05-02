// 16-bit Console — pass 1: ordered Bayer-matrix dithering.
//
// Applies a 4×4 ordered dither (Bayer matrix) before quantizing each channel to
// `color_levels` discrete steps. This distributes quantization error spatially,
// reducing visible banding compared to straight truncation. The technique was used
// in 16-bit era hardware rendering and tile-based game art pipelines.
//
// Bayer 4×4 threshold matrix (values 0–15, normalized to [0, 1) by dividing by 16):
//
//    0   8   2  10
//   12   4  14   6
//    3  11   1   9
//   15   7  13   5
//
// The threshold offsets a channel value by ±(0.5 / (levels - 1)) before rounding,
// nudging borderline values toward the nearest palette entry in a spatially ordered
// pattern. This is equivalent to adding scaled noise before quantization.
//
// `strength` is the master blend from the Rust params struct. At strength=0 the
// output equals the source exactly. The blend is performed in the second pass so
// that the source is available there; this pass outputs the fully dithered result.
//
// Alpha is passed through unmodified.
//
// All passes share the same uniform struct; the full struct is declared here to
// satisfy WebGPU uniform binding-size validation.

struct Console16BitParams {
    color_levels:     f32,
    saturation_boost: f32,
    strength:         f32,
    _padding:         f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: Console16BitParams;

// 4×4 Bayer threshold matrix, row-major, values in [0, 16).
// Normalized to [0, 1) by dividing by 16 at use-time.
fn bayer_threshold(x: u32, y: u32) -> f32 {
    let bayer = array<u32, 16>(
         0u,  8u,  2u, 10u,
        12u,  4u, 14u,  6u,
         3u, 11u,  1u,  9u,
        15u,  7u, 13u,  5u,
    );
    let idx = (y % 4u) * 4u + (x % 4u);
    return f32(bayer[idx]) / 16.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Clamp levels to [2, 256] to avoid division by zero.
    let levels = clamp(params.color_levels, 2.0, 256.0);
    let steps  = levels - 1.0;

    // The Bayer threshold is in [0, 1). Rescale it to one quantization step so
    // it offsets the value by at most ±0.5 steps before rounding. Centering the
    // threshold at 0.5 ensures equal-probability up/down dithering.
    let t      = bayer_threshold(gid.x, gid.y);
    let offset = (t - 0.5) / steps;

    // Apply threshold offset then quantize: round to nearest palette step.
    // Operating in linear-light space matches the working color space of the
    // pipeline (Rgba16Float textures hold linear values).
    let r = round((pixel.r + offset) * steps) / steps;
    let g = round((pixel.g + offset) * steps) / steps;
    let b = round((pixel.b + offset) * steps) / steps;

    // Store the dithered result. The saturate pass reads this and blends
    // it with the original source according to `strength`.
    textureStore(output_texture, coord, vec4<f32>(r, g, b, pixel.a));
}
