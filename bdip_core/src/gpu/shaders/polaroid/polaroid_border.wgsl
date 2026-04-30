struct PolaroidParams {
    grade:    f32,
    border:   f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: PolaroidParams;

// Classic Polaroid 600/OneStep frame proportions (approximate).
// Sides and top are a thin equal margin; bottom is the wide "writing space".
const BORDER_SIDE:   f32 = 0.058;
const BORDER_TOP:    f32 = 0.058;
const BORDER_BOTTOM: f32 = 0.186;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<u32>(global_id.xy);
    let color = textureLoad(src_texture, coord, 0);

    let uv = vec2<f32>(coord) / vec2<f32>(dims);

    // 1.0 inside the photo area, 0.0 in the surrounding border region.
    let in_photo = uv.x > BORDER_SIDE
        && uv.x < (1.0 - BORDER_SIDE)
        && uv.y > BORDER_TOP
        && uv.y < (1.0 - BORDER_BOTTOM);
    let photo_mask = select(0.0, 1.0, in_photo);

    // Border pixels blend toward white; photo pixels are unchanged.
    let border_blend = (1.0 - photo_mask) * params.border;
    let out_rgb = mix(color.rgb, vec3<f32>(1.0), border_blend);

    textureStore(dst_texture, coord, vec4<f32>(out_rgb, color.a));
}
