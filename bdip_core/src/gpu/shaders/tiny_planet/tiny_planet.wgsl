struct TinyPlanetParams {
    // Zoom level controlling how much of the image wraps around the sphere.
    // 0.0 = identity (flat pass-through); positive values increase the planet
    // effect. Range [0.0, 1.0].
    zoom:     f32,
    // Rotation of the source image around the vertical axis before projection,
    // in degrees. Range [-180.0, 180.0].
    rotation: f32,
    _padding0: f32,
    _padding1: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: TinyPlanetParams;

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Normalised UV in [0, 1], pixel-centre aligned.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // At zoom=0 the effect is disabled; pass through without any projection.
    if params.zoom <= 0.0 {
        let src_coord = vec2<i32>(clamp(
            vec2<i32>(uv * vec2<f32>(dims)),
            vec2<i32>(0),
            vec2<i32>(dims) - vec2<i32>(1),
        ));
        textureStore(dst_texture, coord, textureLoad(src_texture, src_coord, 0));
        return;
    }

    // Map output pixel to centred [-1, 1] coordinates (aspect-ratio corrected).
    let aspect = f32(dims.x) / f32(dims.y);
    let centred = vec2<f32>(
        (uv.x * 2.0 - 1.0) * aspect,
        uv.y * 2.0 - 1.0,
    );

    // Stereographic projection: from a 2-D output plane to a sphere viewed from
    // below.  A zoom scale < 1 zooms in (tighter planet); we map the user-facing
    // [0,1] zoom to a sphere scale [2.5, 0.3] so that zoom=1 gives a compact
    // planet and zoom=0 is the identity pass-through handled above.
    //
    // Scale is applied to the centred plane coordinates before projecting.
    let scale = 2.5 - params.zoom * 2.2;
    let px = centred.x * scale;
    let py = centred.y * scale;

    // Inverse stereographic projection from (px, py) on the 2-D plane to a
    // point on the unit sphere:
    //   r² = px² + py²
    //   longitude = atan2(py, px)
    //   latitude  = π/2 - 2·atan(r)   (south-pole projection → bottom of image)
    let r2 = px * px + py * py;
    let r  = sqrt(r2);

    // atan2 is undefined at r=0; default to 0 longitude (maps to left edge of image).
    var longitude: f32;
    if r < 1e-6 {
        longitude = 0.0;
    } else {
        longitude = atan2(py, px);
    }

    // Latitude: π/2 at the nadir (pole), decreasing toward 0 / -π/2 at horizon.
    let latitude = PI * 0.5 - 2.0 * atan(r);

    // Map spherical coordinates to equirectangular UV.
    //   longitude ∈ [-π, π]  → u ∈ [0, 1]
    //   latitude  ∈ [-π/2, π/2] → v ∈ [0, 1]  (v=0 = top = equator in planet mode)
    var src_u = (longitude / TWO_PI) + 0.5;
    let src_v = 0.5 - latitude / PI;

    // Apply rotation by shifting the u coordinate (horizontal longitude offset).
    let rot_offset = params.rotation / 360.0;
    src_u = src_u + rot_offset;

    // Wrap u into [0, 1] for seamless horizontal panorama tiling.
    src_u = src_u - floor(src_u);

    // Pixels with latitude above π/2 (above the nadir) map to the area behind
    // the sphere — fill with black.
    if latitude > PI * 0.5 || src_v < 0.0 || src_v > 1.0 {
        textureStore(dst_texture, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let src_coord = vec2<i32>(clamp(
        vec2<i32>(vec2<f32>(src_u, src_v) * vec2<f32>(dims)),
        vec2<i32>(0),
        vec2<i32>(dims) - vec2<i32>(1),
    ));

    let color = textureLoad(src_texture, src_coord, 0);
    textureStore(dst_texture, coord, color);
}
