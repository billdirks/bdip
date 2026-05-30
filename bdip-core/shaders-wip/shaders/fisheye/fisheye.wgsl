struct FisheyeParams {
    // Barrel distortion strength. 0.0 = no-op, >0 = barrel (fisheye bulge),
    // <0 = pincushion (inverse fisheye). Range [-1.0, 1.0].
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: FisheyeParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Normalized UV in [0, 1] with half-pixel offset for pixel centres.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Shift to centred coordinates in [-1, 1].
    let centred = uv * 2.0 - vec2<f32>(1.0);

    // Radial distance from centre (0 at centre, 1 at corners of the centred
    // square). Correct for aspect ratio so the distortion is circular.
    let aspect = f32(dims.x) / f32(dims.y);
    let p = vec2<f32>(centred.x * aspect, centred.y);
    let r = length(p);

    // Barrel/pincushion distortion factor. At strength=0 the factor is 1.0
    // (identity). Positive strength pushes pixels toward the edges (barrel);
    // negative pulls them toward the centre (pincushion).
    let factor = 1.0 + params.strength * r * r;

    // Apply distortion in the aspect-corrected space, then un-correct.
    let distorted_p = p * factor;
    let distorted_centred = vec2<f32>(distorted_p.x / aspect, distorted_p.y);

    // Map back to [0, 1] UV space.
    let src_uv = (distorted_centred + vec2<f32>(1.0)) * 0.5;

    // Pixels that map outside [0, 1] after distortion are filled with black.
    if src_uv.x < 0.0 || src_uv.x > 1.0 || src_uv.y < 0.0 || src_uv.y > 1.0 {
        textureStore(dst_texture, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Convert back to integer texture coordinates for the source sample.
    let src_coord = vec2<i32>(src_uv * vec2<f32>(dims));
    let clamped = vec2<i32>(
        clamp(src_coord.x, 0, i32(dims.x) - 1),
        clamp(src_coord.y, 0, i32(dims.y) - 1),
    );

    let color = textureLoad(src_texture, clamped, 0);
    textureStore(dst_texture, coord, color);
}
