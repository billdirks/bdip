// The uniform struct must match the Rust MoodyBlueParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would make the struct 32 bytes and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct MoodyBlueParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: MoodyBlueParams;

// Rec. 709 luminance coefficients (linear light).
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);
    let rgb   = pixel.rgb;

    let lum = luminance(rgb);

    // Shadow tint weight: peaks at lum=0 (pure shadows) and fades smoothly to
    // zero at lum=1.0 (pure highlights). smoothstep(1.0, 0.0, lum) maps lum=0
    // to 1.0 and lum=1 to 0.0 with an S-curve transition, so highlights receive
    // no tint and shadows receive the full effect.
    //
    // The upper edge is set at 1.0 (not a midpoint like 0.5) so that the blue
    // tint fades gradually across the full tonal range, matching the "moody" look
    // where even midtones retain a hint of cool color while bright highlights stay
    // neutral.
    let shadow_w = smoothstep(1.0, 0.0, lum);

    // Cool blue tint target in linear light. A desaturated blue-indigo with
    // luminance ≈ 0.065 (very dark) ensures blending into shadows does not
    // produce a significant brightness lift.
    let blue_target = vec3<f32>(0.02, 0.05, 0.18);

    // Blend the pixel toward the blue target, weighted by shadow zone and
    // user-controlled strength. At strength=0 the blend weight is 0 — no tint.
    // At strength=1 and lum=0, the pixel is fully replaced by blue_target.
    let tint_blend = params.strength * shadow_w;
    let out_rgb    = mix(rgb, blue_target, tint_blend);

    // Do NOT clamp: preserve headroom above 1.0 for downstream shaders.
    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
