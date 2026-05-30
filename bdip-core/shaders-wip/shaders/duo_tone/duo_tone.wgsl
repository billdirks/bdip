struct DuoToneParams {
    shadow_r:    f32,
    shadow_g:    f32,
    shadow_b:    f32,
    _padding0:   f32,
    highlight_r: f32,
    highlight_g: f32,
    highlight_b: f32,
    _padding1:   f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: DuoToneParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Rec. 709 luminance weights applied to linear-light RGB.
    let lum = dot(pixel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    let shadow    = vec3<f32>(params.shadow_r,    params.shadow_g,    params.shadow_b);
    let highlight = vec3<f32>(params.highlight_r, params.highlight_g, params.highlight_b);

    // Map dark tones to shadow color and bright tones to highlight color.
    // When shadow == black (0,0,0) and highlight == white (1,1,1), this is
    // an identity transformation, preserving the original luminance as grayscale.
    let mapped = mix(shadow, highlight, lum);

    textureStore(output_texture, coord, vec4<f32>(mapped, pixel.a));
}
