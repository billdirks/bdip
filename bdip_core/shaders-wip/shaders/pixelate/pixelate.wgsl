struct PixelateParams {
    // Block size in pixels. 1.0 = identity (each output pixel reads its own
    // source pixel). Larger values produce coarser, blockier results.
    block_size: f32,
    _padding0:  f32,
    _padding1:  f32,
    _padding2:  f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: PixelateParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // UV in [0, 1] for the current output pixel.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Derive the number of grid cells from block_size.
    //   pixel_count = dims / block_size
    // Snapping: floor(uv * pixelCount) / pixelCount
    // Substituting pixel_count:
    //   snapped_uv = floor(uv * dims / block_size) * block_size / dims
    //
    // At block_size = 1.0 this reduces to floor(uv * dims) / dims, which maps
    // each output pixel to its own source pixel — the identity transformation.
    let safe_block = max(params.block_size, 1.0);
    let pixel_count = vec2<f32>(dims) / safe_block;
    let snapped_uv = floor(uv * pixel_count) / pixel_count;

    // Convert snapped UV back to integer texture coordinates and clamp to
    // valid range to guard against floating-point edge cases at uv = 1.0.
    let src_coord = vec2<i32>(snapped_uv * vec2<f32>(dims));
    let clamped = vec2<i32>(
        clamp(src_coord.x, 0, i32(dims.x) - 1),
        clamp(src_coord.y, 0, i32(dims.y) - 1),
    );

    let color = textureLoad(src_texture, clamped, 0);
    textureStore(dst_texture, coord, color);
}
