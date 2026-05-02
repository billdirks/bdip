struct LomoParams {
    strength: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: LomoParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    let color = textureLoad(src_texture, coord, 0);

    // Normalized UV in [0,1] with pixel center at +0.5.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Radial distance from center, scaled so the corner of a 1:1 image is ~0.707.
    // smoothstep maps this to a vignette weight in [0,1].
    let d = distance(uv, vec2<f32>(0.5, 0.5));

    // Vignette falloff: starts dimming at ~0.35 from center, reaches black by ~0.75.
    // These constants were chosen to reproduce the characteristic lomo look:
    // a bright, vivid center that fades quickly toward the corners.
    let vignette = 1.0 - smoothstep(0.35, 0.75, d);

    // Saturation boost: interpolate away from luminance using Rec. 709 weights.
    //   strength = 0.0 → sat_scale = 1.0 → no change (identity)
    //   strength = 1.0 → sat_scale = 1.5 → 50 % saturation boost
    let luminance = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
    let sat_scale = 1.0 + 0.5 * params.strength;
    let saturated = mix(vec3<f32>(luminance), color.rgb, sat_scale);

    // Blend vignette and saturation effects by strength.
    // At strength=0 the vignette has no effect (vignette_blend=1) and saturation
    // is also identity — the result is an unmodified pass-through.
    let vignette_blend = 1.0 - params.strength * (1.0 - vignette);
    let out_rgb = saturated * vignette_blend;

    textureStore(dst_texture, coord, vec4<f32>(out_rgb, color.a));
}
