// The uniform struct must match the Rust TealAndOrangeParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would make the struct 32 bytes and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct TealAndOrangeParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: TealAndOrangeParams;

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

    // Rec. 709 luminance in linear light: drives the shadow/highlight split.
    let lum = luminance(rgb);

    // Shadow weight: smoothstep rising from black to mid-grey.
    // Pixels near lum=0 receive full teal blend; weight falls to 0 at lum>=0.5.
    let shadow_w = smoothstep(0.5, 0.0, lum);

    // Highlight weight: smoothstep rising from mid-grey to white.
    // Pixels near lum=1 receive full orange blend; weight falls to 0 at lum<=0.5.
    let highlight_w = smoothstep(0.5, 1.0, lum);

    // Target hues in linear light:
    //   teal   — equal green and blue, no red (a blue-green).
    //   orange — full red, half green, no blue.
    // These values are perceptually matched so both targets share roughly equal
    // luminance (~0.25), which avoids a brightness shift when blending.
    let teal_target   = vec3<f32>(0.0,  0.25, 0.25);
    let orange_target = vec3<f32>(0.37, 0.18, 0.0);

    // Lerp the pixel toward each target, weighted by the luminance zone and
    // the user-controlled strength parameter.  The two weights are exclusive by
    // construction (their overlap at lum=0.5 is zero), so they can be added
    // without double-counting.
    let blend = params.strength * (shadow_w + highlight_w);
    let teal_contrib   = params.strength * shadow_w    * (teal_target   - rgb);
    let orange_contrib = params.strength * highlight_w * (orange_target - rgb);
    let out_rgb = rgb + teal_contrib + orange_contrib;

    // Do NOT clamp: preserve headroom above 1.0 for downstream shaders.
    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
