// 8-bit Pixel Art — pass 2: color quantization (palette limiting).
//
// Reduces each color channel to `color_levels` discrete steps, simulating the
// limited palette of early 8-bit hardware.
//
// Quantization formula per channel c ∈ [0, 1]:
//
//   quantized = round(c * (levels - 1)) / (levels - 1)
//
// At color_levels == 256 the step size is 1/255 ≈ 0.00392, which matches the
// native u8 resolution and is effectively imperceptible through a u16 round-trip
// (identity within rounding). At color_levels == 2 only two values (0.0 and 1.0)
// survive, collapsing the image to two-tone per channel.
//
// Alpha is passed through unmodified — palette quantization applies to color only.
//
// Color values are not clamped here. The input is linear-light Rgba16Float which
// may exceed [0, 1]; the quantization snaps to the nearest representable step at
// whatever magnitude the value occupies. Values outside [0, 1] are not specially
// handled — they quantize cleanly to the nearest step above 1 or below 0.
//
// Declares the full PixelArt8BitParams struct (matching the pixelate pass) to
// satisfy WebGPU uniform binding-size validation.

struct PixelArt8BitParams {
    pixel_size:   f32,
    color_levels: f32,
    _padding0:    f32,
    _padding1:    f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: PixelArt8BitParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Clamp levels to the valid range [2, 256] to avoid division by zero and
    // to prevent degenerate single-level outputs.
    let levels = clamp(params.color_levels, 2.0, 256.0);
    let steps  = levels - 1.0;

    // Quantize RGB; leave alpha unchanged.
    let r = round(pixel.r * steps) / steps;
    let g = round(pixel.g * steps) / steps;
    let b = round(pixel.b * steps) / steps;

    textureStore(output_texture, coord, vec4<f32>(r, g, b, pixel.a));
}
