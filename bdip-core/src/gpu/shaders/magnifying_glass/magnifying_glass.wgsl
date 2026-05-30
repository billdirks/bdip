/// Magnifying Glass — circular UV scale distortion.
///
/// Inside the lens circle, each output pixel samples the source at a UV coordinate
/// that has been scaled toward the lens centre by `1 / zoom`. This pulls in pixels
/// from closer to the centre, making them appear enlarged. Outside the circle the
/// image is passed through unchanged.
///
/// Aspect-ratio correction is applied before the distance test so the lens boundary
/// is a true circle in screen space, not an ellipse. The radius parameter is a
/// fraction of the shorter image dimension (whichever of width or height is smaller),
/// which keeps the lens the same relative size regardless of image orientation.
///
/// Identity condition: when `zoom == 1.0`, the scaled UV equals the original UV
/// for every point inside the circle, so the output is pixel-for-pixel identical
/// to the source.

struct MagnifyingGlassParams {
    // Magnification factor inside the lens. 1.0 = identity (no-op).
    zoom:     f32,
    // Lens circle radius as a fraction of the shorter image dimension.
    radius:   f32,
    // Lens centre in normalised [0, 1] UV space.
    center_x: f32,
    center_y: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: MagnifyingGlassParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Normalised UV in [0, 1] at the pixel centre.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Displacement from the lens centre in UV space.
    let delta = uv - vec2<f32>(params.center_x, params.center_y);

    // Correct for aspect ratio so the distance test produces a circular lens.
    // The shorter dimension is used as the radius reference so the lens size
    // is consistent regardless of whether the image is landscape or portrait.
    let aspect = f32(dims.x) / f32(dims.y);
    let shorter = f32(min(dims.x, dims.y));
    let longer  = f32(max(dims.x, dims.y));

    // Scale delta into a space where both axes have the same pixel density as
    // the shorter dimension.  This makes `length(delta_corrected)` measure
    // distance in units of the shorter dimension.
    let delta_corrected = delta * vec2<f32>(aspect, 1.0) * (shorter / longer);

    // Radius in the same corrected space: params.radius is already a fraction
    // of the shorter dimension, so convert it to corrected UV units.
    let radius_uv = params.radius * (shorter / longer);

    let dist = length(delta_corrected);

    var src_uv: vec2<f32>;

    if dist < radius_uv && params.zoom > 1.0 {
        // Inside the lens: scale UV toward the lens centre.
        // Dividing delta by zoom pulls the sample point closer to the centre,
        // which makes the region around the centre appear magnified.
        src_uv = vec2<f32>(params.center_x, params.center_y) + delta / params.zoom;
    } else {
        // Outside the lens (or zoom == 1.0): pass through unchanged.
        src_uv = uv;
    }

    // Clamp to valid texture coordinates in case the zoomed UV escapes the image.
    let src_coord = vec2<i32>(clamp(
        src_uv * vec2<f32>(dims),
        vec2<f32>(0.0),
        vec2<f32>(dims) - vec2<f32>(1.0),
    ));

    let color = textureLoad(src_texture, src_coord, 0);
    textureStore(dst_texture, coord, color);
}
