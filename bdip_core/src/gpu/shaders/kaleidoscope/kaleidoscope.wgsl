struct KaleidoscopeParams {
    // Number of mirror segments. At 1.0 the effect is a single reflection (identity-
    // equivalent visually), so the identity default is 1.0. Range [1.0, 32.0].
    segments: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: KaleidoscopeParams;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Normalized UV in [0, 1] with half-pixel offset for pixel centres.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Shift to centred coordinates in [-1, 1], with aspect-ratio correction so
    // the polar mapping is circular rather than elliptical.
    let aspect = f32(dims.x) / f32(dims.y);
    let centred = uv * 2.0 - vec2<f32>(1.0);
    let p = vec2<f32>(centred.x * aspect, centred.y);

    // Convert to polar coordinates. atan2 returns [-PI, PI].
    let r = length(p);
    let theta = atan2(p.y, p.x);

    // Fold the angular space by the segment count.
    // Each segment spans a wedge of width (PI / segments).
    // We quantise theta into that wedge, then mirror within it so the folded
    // angle always lies in [0, wedge_half], producing the kaleidoscope symmetry.
    let seg = params.segments;
    let wedge = PI / seg;

    // Wrap theta to [0, 2*PI) then fold into [0, wedge].
    let theta_pos = (theta + PI) % (2.0 * wedge);  // period = one full segment pair
    let folded = abs(theta_pos - wedge);             // mirror within the wedge

    // Reconstruct Cartesian UV from the folded angle and original radius.
    let folded_p = vec2<f32>(cos(folded) * r / aspect, sin(folded) * r);

    // Map back to [0, 1] UV space.
    let src_uv = (folded_p + vec2<f32>(1.0)) * 0.5;

    // Pixels that map outside [0, 1] are filled with black (can occur for large r
    // values near image corners when aspect != 1).
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
