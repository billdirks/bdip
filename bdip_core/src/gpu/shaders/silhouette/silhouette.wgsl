struct SilhouetteParams {
    threshold:    f32,
    softness:     f32,
    fg_r:         f32,
    fg_g:         f32,
    fg_b:         f32,
    _padding0:    f32,
    bg_r:         f32,
    bg_g:         f32,
    bg_b:         f32,
    _padding1:    f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: SilhouetteParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Rec. 709 luminance weights applied to linear-light RGB.
    let lum = dot(pixel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    let fg = vec3<f32>(params.fg_r, params.fg_g, params.fg_b);
    let bg = vec3<f32>(params.bg_r, params.bg_g, params.bg_b);

    // smoothstep maps luminance through a soft transition centered on the threshold.
    // When softness is 0 the transition is a hard step; larger values widen the
    // feathered zone symmetrically around the threshold. Softness is clamped to a
    // small epsilon to avoid a zero-width smoothstep range, which would produce a
    // discontinuous step at threshold.
    let half_soft = max(params.softness * 0.5, 0.0001);
    let t = smoothstep(params.threshold - half_soft, params.threshold + half_soft, lum);

    // t == 0  → pixel is below threshold → foreground (dark) color.
    // t == 1  → pixel is above threshold → background (light) color.
    let mapped = mix(fg, bg, t);

    textureStore(output_texture, coord, vec4<f32>(mapped, pixel.a));
}
