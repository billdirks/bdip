struct HighlightsParams {
    amt: f32,
    range: f32,
    end: f32,
    _padding: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: HighlightsParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    let color = textureLoad(src_texture, coord, 0);
    let L = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // W_h=1 for pure highlights, 0 for mid-tones and below.
    let w_h = smoothstep(params.range, params.end, L);
    let out_rgb = color.rgb + (params.amt * w_h * color.rgb);

    textureStore(dst_texture, coord, vec4<f32>(clamp(out_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), color.a));
}
