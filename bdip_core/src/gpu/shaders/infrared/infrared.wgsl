struct InfraredParams {
    strength:  f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: InfraredParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Infrared simulation: swap the red and green channels.
    // In infrared photography, foliage (strong IR reflector) records as bright,
    // while clear sky (absorbs IR) records very dark. Swapping red and green
    // approximates the characteristic false-colour look of IR film.
    // When params.strength is 0.0, mix() returns the original pixel (identity).
    let infrared = vec4<f32>(pixel.g, pixel.r, pixel.b, pixel.a);
    let out = mix(pixel, infrared, params.strength);

    textureStore(output_texture, coord, out);
}
