// The uniform struct must match the Rust GoldenHourParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would make the struct 32 bytes and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct GoldenHourParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: GoldenHourParams;

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

    // Global channel scaling: warm color temperature shift applied uniformly
    // to all pixels. Boost red slightly, boost green less, reduce blue.
    // At strength=0 the multipliers collapse to vec3(1.0, 1.0, 1.0) — identity.
    let channel_scale = vec3<f32>(
        1.0 + params.strength * 0.15,   // red:   +15% at full strength
        1.0 + params.strength * 0.05,   // green: +5% at full strength
        1.0 - params.strength * 0.20,   // blue:  -20% at full strength
    );
    let scaled_rgb = rgb * channel_scale;

    // Warm tint in shadows and midtones: blend toward a golden-amber color for
    // dark and mid-luminance pixels. The tint target is chosen so that at full
    // strength the shadows glow amber without blowing out.
    //
    // Shadow/midtone weight peaks at lum=0 and falls to 0 at lum=0.7, covering
    // both shadows and the lower half of midtones — the zones most affected by
    // golden-hour light wrapping around subjects.
    let warm_weight = smoothstep(0.7, 0.0, lum);

    // Golden-amber tint target in linear light. Equal red and slightly less green,
    // zero blue: a saturated amber that matches late-afternoon sunlight.
    // This target has luminance ≈ 0.22 (similar to an 18% grey card) so blending
    // into shadows does not produce a large brightness shift.
    let warm_target = vec3<f32>(0.30, 0.18, 0.02);

    // Blend the channel-scaled pixel toward the warm target in shadow/midtone zones.
    // At strength=0 the blend weight is 0 — no tint applied.
    let tint_blend   = params.strength * warm_weight * 0.4;
    let tinted_rgb   = mix(scaled_rgb, warm_target, tint_blend);

    // Do NOT clamp: preserve headroom above 1.0 for downstream shaders.
    textureStore(output_texture, coord, vec4<f32>(tinted_rgb, pixel.a));
}
