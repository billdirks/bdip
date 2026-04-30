// Pop Art — quantize pass.
//
// Posterizes the source image into a small number of discrete tonal levels,
// producing the flat, graphic color regions characteristic of pop art.
// Quantization runs in linear-light space, consistent with the rest of the pipeline.
//
// All Pop Art WGSL files declare the full PopArtParams struct to satisfy
// WebGPU's uniform binding-size validation.

struct PopArtParams {
    strength:  f32,
    levels:    f32,
    dot_scale: f32,
    _padding:  f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: PopArtParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    let L = clamp(params.levels, 2.0, 8.0);
    let quantized = clamp(floor(pixel.rgb * L) / (L - 1.0), vec3<f32>(0.0), vec3<f32>(1.0));

    textureStore(output_texture, coord, vec4<f32>(quantized, pixel.a));
}
