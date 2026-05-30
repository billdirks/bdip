struct MosaicParams {
    // Tile width in pixels. 1.0 = identity (each pixel reads from its own tile center,
    // which is itself). Larger values produce wider rectangular tiles.
    tile_width:  f32,
    // Tile height in pixels. 1.0 = identity. Larger values produce taller tiles.
    tile_height: f32,
    _padding0:   f32,
    _padding1:   f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: MosaicParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Clamp tile dimensions to at least 1 px to avoid division by zero.
    let tw = max(params.tile_width,  1.0);
    let th = max(params.tile_height, 1.0);

    // Determine which tile this output pixel belongs to by flooring its pixel
    // coordinates to the nearest tile boundary.
    let tile_origin_x = floor(f32(coord.x) / tw) * tw;
    let tile_origin_y = floor(f32(coord.y) / th) * th;

    // Sample from the center of the tile rather than its top-left corner.
    // This avoids the visual bias toward one corner that top-left sampling produces,
    // giving each tile a color representative of its middle.
    let center_x = tile_origin_x + tw * 0.5;
    let center_y = tile_origin_y + th * 0.5;

    // Convert to integer coordinates, clamping to valid texture range to guard
    // against floating-point rounding at image edges.
    let src_coord = vec2<i32>(
        clamp(i32(center_x), 0, i32(dims.x) - 1),
        clamp(i32(center_y), 0, i32(dims.y) - 1),
    );

    let color = textureLoad(src_texture, src_coord, 0);
    textureStore(dst_texture, coord, color);
}
