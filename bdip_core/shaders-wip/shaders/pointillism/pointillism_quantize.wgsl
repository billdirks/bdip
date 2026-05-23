// Pointillism — Pass 1: grid-cell colour quantization.
//
// For each output pixel, samples the source image at the centre of the pixel's
// grid cell and stores that colour. The output scratch texture is a colour-
// quantized version of the source where every pixel within a grid cell holds
// the same colour (the colour at the cell centre).
//
// Output layout (rgba16float scratch texture):
//   .rgb = source colour sampled at the grid-cell centre
//   .a   = source alpha at the current pixel (passed through for compositing)
//
// All PointillismParams fields must be declared in every pass to satisfy
// WebGPU's uniform binding-size validation requirement.

struct PointillismParams {
    // Blend factor: 0.0 = source unchanged (identity), 1.0 = full effect.
    strength:  f32,
    // Grid cell size in pixels. Determines spacing between dot centres.
    grid_size: f32,
    // Dot radius as a fraction of the grid cell half-size. 1.0 fills the cell.
    dot_size:  f32,
    _padding:  f32,
}

// Bindings — 1 input: source at binding 0, output at binding 1.
@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: PointillismParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);

    // Clamp grid_size to at least 1 pixel to avoid divide-by-zero.
    let gs = max(params.grid_size, 1.0);

    // Determine the centre of the grid cell that this pixel belongs to.
    // Cell column/row index: floor(coord / gs).
    // Cell centre coordinate: (cell_index + 0.5) * gs.
    let cell_col = floor(f32(coord.x) / gs);
    let cell_row = floor(f32(coord.y) / gs);
    let centre_x = i32((cell_col + 0.5) * gs);
    let centre_y = i32((cell_row + 0.5) * gs);

    // Clamp the sample coordinate to valid texture bounds.
    let sample_coord = clamp(
        vec2<i32>(centre_x, centre_y),
        vec2<i32>(0),
        vec2<i32>(dims) - 1,
    );

    // Sample source colour at the cell centre.
    let cell_colour = textureLoad(src_texture, sample_coord, 0);

    // Preserve per-pixel alpha from the original source for later compositing.
    let src_alpha = textureLoad(src_texture, coord, 0).a;

    textureStore(dst_texture, coord, vec4<f32>(cell_colour.rgb, src_alpha));
}
