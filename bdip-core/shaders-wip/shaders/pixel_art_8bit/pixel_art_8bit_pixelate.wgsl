// 8-bit Pixel Art — pass 1: pixelate.
//
// Each output pixel snaps to the nearest block-aligned source sample. The block
// origin is computed as floor(coord / pixel_size) * pixel_size, so every output
// pixel within the same block reads the same source texel. This replicates the
// nearest-neighbour downscale + upscale in a single pass without requiring a
// scratch texture at reduced resolution.
//
// When pixel_size == 1.0 each output pixel maps to exactly its own source texel
// (identity). No clamping of color values is performed here; the quantize pass
// handles palette reduction.
//
// Declares the full PixelArt8BitParams struct to satisfy WebGPU uniform
// binding-size validation (the quantize pass reads both fields from the same
// uniform buffer).

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

    // Snap each coordinate down to the nearest block-aligned origin.
    let block_size = max(params.pixel_size, 1.0);
    let bx = i32(floor(f32(gid.x) / block_size) * block_size);
    let by = i32(floor(f32(gid.y) / block_size) * block_size);

    // Clamp to image bounds so border blocks never read outside.
    let src_coord = clamp(
        vec2<i32>(bx, by),
        vec2<i32>(0),
        vec2<i32>(dims) - 1,
    );

    let pixel = textureLoad(input_texture, src_coord, 0);
    textureStore(output_texture, vec2<i32>(gid.xy), pixel);
}
