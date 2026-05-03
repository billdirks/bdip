// Bokeh Shapes — polygon blur pass.
//
// Operates on the 4× downsampled scratch texture produced by the downsample pass.
// For each output pixel, samples the input in a polygon-shaped gather kernel and
// averages the included samples. The kernel radius is `params.radius / 4` to
// account for the 4× downsample factor, keeping the user-facing radius parameter
// in full-resolution pixel units.
//
// The aperture shape is determined by a regular polygon signed-distance function
// keyed on `params.sides`:
//
//   sides == 0  →  circular aperture (Euclidean distance)
//   sides >= 3  →  regular polygon with `floor(sides)` sides
//
// RADIUS_CAP bounds the loop to ceil(50/4) = 13, matching the maximum user radius
// of 50 px mapped into the 4× downsampled coordinate system.
//
// All BokehShapes WGSL files declare the full BokehShapesParams struct to satisfy
// WebGPU's uniform binding-size validation.

struct BokehShapesParams {
    radius:   f32,
    sides:    f32,
    strength: f32,
    _padding: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: BokehShapesParams;

// Maximum kernel radius in downsampled pixels: ceil(50 / 4) = 13.
const RADIUS_CAP: i32 = 13;
const PI: f32 = 3.14159265358979323846;
// Downsample factor must match the PassScale::Down(N) value in mod.rs.
const DOWNSAMPLE_FACTOR: f32 = 4.0;

// Regular polygon SDF for a convex n-gon centred at the origin with circumradius r.
// Returns a value <= 0 for points inside the polygon, > 0 outside.
// For n < 3 (circle mode) this function is not called.
//
// The polygon is oriented with one vertex pointing up. The formula projects each
// sample offset onto the nearest polygon face half-plane using the per-sector
// angular alignment.
fn polygon_sdf(p: vec2<f32>, n: f32, r: f32) -> f32 {
    let angle   = atan2(p.y, p.x);
    let sector  = PI / n;
    let snapped = round(angle / (2.0 * sector)) * (2.0 * sector);
    let edge_dist = cos(sector) * r;
    let proj = dot(p, vec2<f32>(cos(snapped), sin(snapped)));
    return proj - edge_dist;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);

    // Convert full-resolution pixel radius to downsampled-image pixel radius.
    let r_ds = params.radius / DOWNSAMPLE_FACTOR;
    let r    = min(i32(ceil(r_ds)), RADIUS_CAP);

    // When the effective downsampled radius is 0, copy the pixel unchanged.
    if r == 0 {
        let src = textureLoad(input_texture, coord, 0);
        textureStore(output_texture, coord, src);
        return;
    }

    let n_sides   = floor(params.sides);
    let use_circle = n_sides < 3.0;

    var accum = vec4<f32>(0.0);
    var count = 0.0;

    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy));

            var inside = false;
            if use_circle {
                // Circular aperture: Euclidean distance against the downsampled radius.
                inside = length(offset) <= r_ds;
            } else {
                // Polygon aperture: SDF test against the downsampled radius.
                inside = polygon_sdf(offset, n_sides, r_ds) <= 0.0;
            }

            if inside {
                let sample_coord = clamp(
                    coord + vec2<i32>(dx, dy),
                    vec2<i32>(0),
                    vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1),
                );
                accum += textureLoad(input_texture, sample_coord, 0);
                count += 1.0;
            }
        }
    }

    var out: vec4<f32>;
    if count > 0.0 {
        out = accum / count;
    } else {
        out = textureLoad(input_texture, coord, 0);
    }

    textureStore(output_texture, coord, out);
}
