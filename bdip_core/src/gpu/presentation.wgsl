// Presentation pass: linear light → sRGB-encoded, packed into a u16 storage buffer.
//
// Reads an Rgba16Float texture whose RGB channels hold linear-light values
// (the output of the transformation chain) and writes them as sRGB-encoded u16
// values into a tightly packed storage buffer. Alpha is copied unchanged.
// Values above 1.0 in linear space (headroom from chained transforms) are
// clamped to [0, 1] before gamma encoding.
//
// This pass is the last step of every pipeline run before CPU readback.
//
// For images that exceed the max_storage_buffer_binding_size limit, the CPU
// dispatches this shader in row-tiles. Each tile gets its own dst_buffer
// (sized to tile_height × width × 2 u32s) and a y_offset uniform that shifts
// the texture read. Buffer writes use the tile-local y (global_id.y), not
// the full-image y.
//
// Buffer layout per tile: 2 u32s per pixel, tightly packed, no row padding.
//   dst_buffer[base]     = R in bits [ 0..15] | G in bits [16..31]
//   dst_buffer[base + 1] = B in bits [ 0..15] | A in bits [16..31]
// where base = (local_y * width + x) * 2.
//
// When the CPU reads the buffer back as &[u16] via bytemuck::cast_slice,
// the result is interleaved [R, G, B, A, R, G, B, A, ...] that
// Rgba16Image::from_raw accepts directly — no per-pixel CPU loop required.

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst_buffer: array<u32>;

struct PresentParams {
    width: u32,
    y_offset: u32,
    tile_height: u32,
    _padding: u32,
}
@group(1) @binding(0) var<uniform> params: PresentParams;

fn linear_to_srgb(c: f32) -> f32 {
    if (c <= 0.0031308) {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);

    // Bounds-check against the tile, not the full texture. The dispatched
    // workgroup grid covers tile_height rows; threads beyond that must not
    // write past the tile buffer.
    if (global_id.x >= dimensions.x || global_id.y >= params.tile_height) {
        return;
    }

    // Texture coordinate: y_offset shifts reads when tiling large images.
    let coords = vec2<u32>(global_id.x, params.y_offset + global_id.y);

    let color = textureLoad(src_texture, coords, 0);

    // Clamp to [0, 1] before gamma encoding; values above 1.0 are headroom
    // that gets clipped at presentation time.
    let srgb_r = linear_to_srgb(clamp(color.r, 0.0, 1.0));
    let srgb_g = linear_to_srgb(clamp(color.g, 0.0, 1.0));
    let srgb_b = linear_to_srgb(clamp(color.b, 0.0, 1.0));
    let alpha   = clamp(color.a, 0.0, 1.0);

    let r_u16 = u32(srgb_r * 65535.0);
    let g_u16 = u32(srgb_g * 65535.0);
    let b_u16 = u32(srgb_b * 65535.0);
    let a_u16 = u32(alpha  * 65535.0);

    // Write to tile-local buffer position (global_id.y, not coords.y).
    let base = (global_id.y * params.width + global_id.x) * 2u;
    dst_buffer[base]      = (r_u16 & 0xFFFFu) | ((g_u16 & 0xFFFFu) << 16u);
    dst_buffer[base + 1u] = (b_u16 & 0xFFFFu) | ((a_u16 & 0xFFFFu) << 16u);
}
