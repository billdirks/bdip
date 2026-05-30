// ASCII Art — Pass 1: BT.709 greyscale conversion.
//
// Converts each pixel to luminance using the BT.709 coefficients and writes
// the result as a greyscale Rgba16Float image (R = G = B = luma, A = source A).
// This scratch texture is consumed by the ascii pass to compute per-cell
// average brightness.

struct AsciiArtParams {
    cell_size: f32,
    strength:  f32,
    _padding:  vec2<f32>,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: AsciiArtParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // BT.709 luminance coefficients (linear-light RGB in, linear luma out).
    let luma = dot(pixel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    textureStore(output_texture, coord, vec4<f32>(luma, luma, luma, pixel.a));
}
