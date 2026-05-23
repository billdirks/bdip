// Stained Glass — Voronoi pass.
//
// For each output pixel this pass finds the nearest Voronoi site within a grid
// of randomised points and samples the source image at that site. It also
// computes the distance ratio to the nearest cell boundary (nearest / second-
// nearest site distance), which the edge pass uses to draw cell borders.
//
// Output layout (rgba16float scratch texture):
//   .rgb = colour sampled from source at the nearest Voronoi site
//   .a   = boundary proximity in [0, 1]; values near 0 are on a cell border
//
// The params struct is identical across both passes (WebGPU uniform binding-
// size validation requires every pass in a shader to declare the same struct).

struct StainedGlassParams {
    // Blend factor between the original image and the stained-glass effect.
    // 0.0 = source image unchanged (identity), 1.0 = full effect.
    // Range [0.0, 1.0].
    strength:   f32,
    // Voronoi cell size as a fraction of the shorter image dimension.
    // Larger values produce bigger, more visible cells.
    // Range [0.01, 0.25].
    cell_size:  f32,
    // Relative width of the dark edge lines, as a fraction of the cell size.
    // Range [0.0, 1.0].
    edge_width: f32,
    _padding:   f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: StainedGlassParams;

// ── Hash helpers ────────────────────────────────────────────────────────────
// A simple two-input hash that maps a grid cell index to a pseudo-random
// offset in [0, 1]². Uses integer bit-mixing rather than transcendental
// functions so it runs efficiently on all GPU tiers.

fn hash2(p: vec2<i32>) -> vec2<f32> {
    var h = vec2<u32>(u32(p.x), u32(p.y));
    // Bit-mix (adapted from Murmur / xxHash style).
    h.x = h.x ^ (h.x >> 16u);
    h.x = h.x * 0x45d9f3bu;
    h.x = h.x ^ (h.y * 0x119de1f3u);
    h.x = h.x ^ (h.x >> 16u);

    h.y = h.y ^ (h.y >> 16u);
    h.y = h.y * 0x45d9f3bu;
    h.y = h.y ^ (h.x * 0x119de1f3u);
    h.y = h.y ^ (h.y >> 16u);

    return vec2<f32>(h) / f32(0xffffffffu);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);

    // Pixel UV in [0, 1].
    let uv = (vec2<f32>(coord) + 0.5) / vec2<f32>(dims);

    // Cell grid: one Voronoi site per grid cell of side `cell_size`.
    // The grid is defined in UV space.
    let short_dim = f32(min(dims.x, dims.y));
    // cell_size_uv: cell dimensions in UV coordinates, per axis.
    let cell_size_uv = vec2<f32>(
        params.cell_size * short_dim / f32(dims.x),
        params.cell_size * short_dim / f32(dims.y),
    );

    // Grid cell the current pixel falls in.
    let cell = vec2<i32>(floor(uv / cell_size_uv));

    // Search the 3×3 neighbourhood of grid cells.
    var min_dist  = 1e9;
    var min2_dist = 1e9;
    var nearest_site = uv; // UV of the nearest Voronoi site

    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let ncell = cell + vec2<i32>(dx, dy);
            // Site UV: grid cell origin + random fractional offset.
            let site_uv = (vec2<f32>(ncell) + hash2(ncell)) * cell_size_uv;
            let d = distance(uv, site_uv);
            if d < min_dist {
                min2_dist    = min_dist;
                min_dist     = d;
                nearest_site = site_uv;
            } else if d < min2_dist {
                min2_dist = d;
            }
        }
    }

    // Boundary proximity: ratio of nearest / second-nearest distance.
    // Pixels near a cell boundary have a ratio close to 1.0.
    // We map to [0, 1] where 0 = on the boundary, 1 = cell centre.
    let boundary_prox = clamp(1.0 - (min_dist / max(min2_dist, 1e-6)), 0.0, 1.0);

    // Sample source colour at the Voronoi site (clamped to texture bounds).
    let site_coord = vec2<i32>(clamp(
        vec2<i32>(nearest_site * vec2<f32>(dims)),
        vec2<i32>(0),
        vec2<i32>(dims) - vec2<i32>(1),
    ));
    let cell_color = textureLoad(src_texture, site_coord, 0);

    // Pack: RGB = cell colour, A = boundary proximity.
    textureStore(dst_texture, coord, vec4<f32>(cell_color.rgb, boundary_prox));
}
