struct SwirlParams {
    // Maximum rotation angle (radians) applied at the centre. 0.0 = identity.
    // Positive values rotate counter-clockwise; negative values rotate clockwise.
    angle:    f32,
    // Distance from centre (in normalised half-diagonal units) at which the
    // rotation reaches zero. Values <= 0.0 are treated as no-op.
    radius:   f32,
    _padding0: f32,
    _padding1: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: SwirlParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Early-out: when angle is zero the transform is an identity.
    if params.angle == 0.0 {
        let color = textureLoad(src_texture, coord, 0);
        textureStore(dst_texture, coord, color);
        return;
    }

    // Normalised UV in [0, 1] with half-pixel offset for pixel centres.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Centred coordinates in [-1, 1], aspect-corrected so that distance is
    // measured in circular rather than elliptical units.
    let aspect = f32(dims.x) / f32(dims.y);
    let centred = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);

    let dist = length(centred);

    // When radius <= 0 treat as identity (avoid division by zero).
    if params.radius <= 0.0 {
        let color = textureLoad(src_texture, coord, 0);
        textureStore(dst_texture, coord, color);
        return;
    }

    // Rotation angle for this pixel falls off linearly with distance from
    // centre. At dist=0 the full angle is applied; at dist>=radius it is zero.
    // The linear falloff (rather than smooth-step) keeps the implementation
    // simple and gives a tight-spiral look; callers can soften it via the
    // radius parameter.
    let t = clamp(1.0 - dist / params.radius, 0.0, 1.0);
    let theta = params.angle * t;

    // Apply a 2-D rotation matrix to the aspect-corrected centred coordinates.
    // [cos θ  -sin θ] [cx]
    // [sin θ   cos θ] [cy]
    let s = sin(theta);
    let c = cos(theta);
    let rotated = vec2<f32>(
        c * centred.x - s * centred.y,
        s * centred.x + c * centred.y,
    );

    // Undo aspect correction and map back to [0, 1] UV space.
    let src_uv = vec2<f32>(rotated.x / aspect + 0.5, rotated.y + 0.5);

    // Pixels that map outside [0, 1] after distortion are filled with black
    // (opaque). Clamping is not used because it would replicate edge pixels
    // across the swirled region, creating smearing artifacts at the borders.
    if src_uv.x < 0.0 || src_uv.x > 1.0 || src_uv.y < 0.0 || src_uv.y > 1.0 {
        textureStore(dst_texture, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Convert UV back to integer texture coordinates for nearest-neighbour sample.
    let src_coord = vec2<i32>(src_uv * vec2<f32>(dims));
    let clamped = vec2<i32>(
        clamp(src_coord.x, 0, i32(dims.x) - 1),
        clamp(src_coord.y, 0, i32(dims.y) - 1),
    );

    let color = textureLoad(src_texture, clamped, 0);
    textureStore(dst_texture, coord, color);
}
