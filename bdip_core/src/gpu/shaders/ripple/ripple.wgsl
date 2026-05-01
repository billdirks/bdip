struct RippleParams {
    // Displacement magnitude in UV space. 0.0 = identity (no ripple).
    // Range [0.0, 0.5]; a value of 0.1 shifts UVs by up to 10% of image width/height.
    amplitude: f32,
    // Number of sine wave cycles across the image. Higher values produce more waves.
    // Range [0.5, 20.0].
    frequency: f32,
    // Phase offset in radians, shifting the wave pattern without changing shape.
    // Range [0.0, 6.283] (one full cycle).
    phase: f32,
    _padding: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: RippleParams;

const TAU: f32 = 6.2831853;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Normalized UV in [0, 1] with half-pixel offset for pixel centres.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Apply a sine wave to the horizontal UV axis driven by the vertical position,
    // and a second sine wave to the vertical axis driven by the horizontal position.
    // Both waves share the same frequency and phase. When amplitude is 0.0 the
    // displacement is zero and this is a strict identity transformation.
    let wave_u = params.amplitude * sin(params.frequency * TAU * uv.y + params.phase);
    let wave_v = params.amplitude * sin(params.frequency * TAU * uv.x + params.phase);
    let src_uv = uv + vec2<f32>(wave_u, wave_v);

    // Pixels that map outside [0, 1] after distortion are filled with black.
    // Clamping is not used here because clamp would replicate edge pixels across
    // the distorted region, creating visible smearing artifacts at borders.
    if src_uv.x < 0.0 || src_uv.x > 1.0 || src_uv.y < 0.0 || src_uv.y > 1.0 {
        textureStore(dst_texture, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Convert UV back to integer texture coordinates for the nearest-neighbour sample.
    let src_coord = vec2<i32>(src_uv * vec2<f32>(dims));
    let clamped = vec2<i32>(
        clamp(src_coord.x, 0, i32(dims.x) - 1),
        clamp(src_coord.y, 0, i32(dims.y) - 1),
    );

    let color = textureLoad(src_texture, clamped, 0);
    textureStore(dst_texture, coord, color);
}
