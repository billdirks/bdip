// Polygon — Pass 1: Voronoi cell colour assignment.
//
// Divides the image into a grid of `density × density` cells. Each cell has a
// jittered seed point: the regular grid centre displaced by a pseudo-random
// offset whose magnitude is controlled by `jitter` (0.0 = regular grid,
// 1.0 = maximum displacement up to the cell half-size).
//
// For each output pixel the shader finds the nearest seed point among a 3×3
// neighbourhood of candidate cells (sufficient to catch the true nearest seed
// for any jitter value ≤ 1.0), then writes the source colour at that seed to
// the scratch texture.
//
// The seed-point hash uses a pair of integer hash functions that are:
//   - deterministic across invocations (no random state)
//   - fast (three multiply-add steps)
//   - low-correlation between neighbouring cells
//
// All PolygonParams fields must be declared in every pass to satisfy WebGPU's
// uniform binding-size validation requirement.

struct PolygonParams {
    // Blend factor used in the second pass; not read here.
    strength: f32,
    // Number of seed points per axis (grid is density × density cells).
    density:  f32,
    // Maximum random seed displacement as a fraction of the cell half-size.
    jitter:   f32,
    _padding: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: PolygonParams;

// ---------------------------------------------------------------------------
// Hash helpers
// ---------------------------------------------------------------------------

// Maps a 2-D integer cell index (cx, cy) to a pseudo-random float in [0, 1)
// for each axis independently.  Uses a simple but effective integer mixing
// approach: multiply by large primes, XOR-fold, and mask to 23 mantissa bits.
fn hash2(cx: i32, cy: i32) -> vec2<f32> {
    // Pack the two cell coordinates into a single u32 seed per axis.
    let ux = u32(cx);
    let uy = u32(cy);

    // Axis-0 hash (for x displacement).
    var hx: u32 = ux * 1664525u + uy * 22695477u + 2891336453u;
    hx ^= hx >> 16u;
    hx *= 0x45d9f3bu;
    hx ^= hx >> 16u;

    // Axis-1 hash (for y displacement) — different mixing constants to
    // avoid the two axes being correlated with each other.
    var hy: u32 = ux * 214013u + uy * 2531011u + 1013904223u;
    hy ^= hy >> 16u;
    hy *= 0xb5297a4du;
    hy ^= hy >> 16u;

    // Map to [0, 1) by extracting the lower 23 bits and dividing.
    let fx = f32(hx & 0x7fffffu) / f32(0x800000u);
    let fy = f32(hy & 0x7fffffu) / f32(0x800000u);
    return vec2<f32>(fx, fy);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<f32>(f32(gid.x), f32(gid.y));
    let fDims = vec2<f32>(f32(dims.x), f32(dims.y));

    // Clamp density to at least 1 cell to avoid divide-by-zero.
    let d = max(params.density, 1.0);

    // Cell size in pixels.
    let cell_size = fDims / d;
    let half_cell = cell_size * 0.5;

    // Normalised position of this pixel in cell-grid space: [0, density).
    let uv_grid = coord / cell_size;

    // Integer cell index of the cell that contains this pixel.
    let cell_base = vec2<i32>(i32(floor(uv_grid.x)), i32(floor(uv_grid.y)));

    // Search a 3×3 neighbourhood of cells (offset -1 to +1 on each axis).
    // This guarantees we check every seed point that could possibly be the
    // nearest neighbour when jitter is within [0, 1] of the half-cell size.
    var best_dist: f32 = 1.0e30;
    var best_seed: vec2<f32> = vec2<f32>(0.0);

    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let cx = cell_base.x + dx;
            let cy = cell_base.y + dy;

            // Seed position: grid-cell centre + jittered offset.
            // Grid-cell centre in pixel space: (cx + 0.5) * cell_size.
            let centre = (vec2<f32>(f32(cx), f32(cy)) + 0.5) * cell_size;

            // Random offset in [-half_cell, +half_cell], scaled by jitter.
            // hash2 returns [0, 1); map to [-0.5, +0.5) then scale.
            let rnd   = hash2(cx, cy) - 0.5;  // [-0.5, +0.5)
            let seed  = centre + rnd * (half_cell * 2.0 * params.jitter);

            let dist = length(coord - seed);
            if dist < best_dist {
                best_dist = dist;
                best_seed = seed;
            }
        }
    }

    // Sample the source image at the nearest seed point.
    let seed_coord = clamp(
        vec2<i32>(i32(best_seed.x), i32(best_seed.y)),
        vec2<i32>(0),
        vec2<i32>(dims) - 1,
    );
    let cell_color = textureLoad(src_texture, seed_coord, 0);

    // Preserve per-pixel alpha from the source for later compositing.
    let src_alpha = textureLoad(src_texture, vec2<i32>(gid.xy), 0).a;

    textureStore(dst_texture, vec2<i32>(gid.xy), vec4<f32>(cell_color.rgb, src_alpha));
}
