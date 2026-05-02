struct MonochromeGreenParams {
    strength:  f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: MonochromeGreenParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Compute Rec.709 luminance from linear-light RGB.
    // Coefficients: R=0.2126, G=0.7152, B=0.0722.
    let luminance = 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;

    // Map luminance to the green channel only, producing the characteristic
    // phosphor-green appearance of old monochrome CRT monitors.
    // When params.strength is 0.0, mix() returns the original pixel (identity).
    let green_mono = vec4<f32>(0.0, luminance, 0.0, pixel.a);
    let out = mix(pixel, green_mono, params.strength);

    textureStore(output_texture, coord, out);
}
