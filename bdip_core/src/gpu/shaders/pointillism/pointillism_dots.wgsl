// Pointillism — Pass 2: circle rendering and final compositing.
//
// For each pixel, checks whether it falls within the circular dot centred on
// its grid cell. Pixels inside the dot receive the quantized cell colour (from
// the scratch texture produced by pass 1); pixels outside the dot receive white
// (simulating blank canvas/paper). The resulting pointillist image is then
// blended with the original source via `params.strength`.
//
// Identity: when strength = 0.0, mix(src, dots, 0.0) = src, so the output
// equals the source image regardless of grid_size or dot_size.
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

// Bindings — 2 inputs: source at binding 0, quantized scratch at binding 1,
// output at binding 2.
@group(0) @binding(0) var src_texture:        texture_2d<f32>;
@group(0) @binding(1) var quantized_texture:  texture_2d<f32>;
@group(0) @binding(2) var dst_texture:        texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params:    PointillismParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(src_texture,       coord, 0);
    let cell  = textureLoad(quantized_texture, coord, 0);

    // Clamp grid_size to at least 1 pixel to avoid divide-by-zero.
    let gs = max(params.grid_size, 1.0);

    // Recompute this pixel's grid-cell centre (same formula as pass 1).
    let cell_col = floor(f32(coord.x) / gs);
    let cell_row = floor(f32(coord.y) / gs);
    let centre   = vec2<f32>((cell_col + 0.5) * gs, (cell_row + 0.5) * gs);

    // Distance from this pixel to its cell centre (Euclidean).
    let dist = length(vec2<f32>(coord) - centre);

    // Maximum dot radius is dot_size * (grid_size / 2).
    // dot_size = 1.0 fills the cell up to its half-width; smaller values leave gaps.
    let dot_radius = params.dot_size * (gs * 0.5);

    // Select dot colour or white paper based on distance from cell centre.
    // White in linear-light Rgba16Float is vec3(1.0).
    var pointillist_rgb: vec3<f32>;
    if dist <= dot_radius {
        pointillist_rgb = cell.rgb;
    } else {
        pointillist_rgb = vec3<f32>(1.0, 1.0, 1.0);
    }

    // Blend: strength=0.0 → source unchanged (identity), strength=1.0 → full effect.
    let out_rgb = mix(src.rgb, pointillist_rgb, params.strength);

    // Alpha is preserved from the source image.
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
