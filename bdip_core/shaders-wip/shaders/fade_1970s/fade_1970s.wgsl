// The uniform struct must match the Rust Fade1970sParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would make the struct 32 bytes and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct Fade1970sParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: Fade1970sParams;

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

    // ── Black point lift ─────────────────────────────────────────────────────
    //
    // 1970s film stocks had a raised black floor: shadows settle at a grey-brown
    // rather than pure black. This is implemented by compressing the tonal range
    // upward so that linear 0.0 maps to a lifted value.
    //
    // The lift target is a warm dark-grey (more red and green than blue),
    // approximating the aged-emulsion base-fog color. At strength=1.0 the black
    // point is raised to this target; at strength=0.0 no lift is applied.
    //
    // Black-point lift: from (0,0,0) toward the warm-grey floor.
    let lift_target = vec3<f32>(0.045, 0.038, 0.025);

    // Shadow weight: peaks at lum=0, falls linearly to 0 at lum=0.3.
    // Only the darkest tones receive the full lift; midtones receive a partial lift.
    let shadow_weight = clamp(1.0 - lum / 0.3, 0.0, 1.0);
    let lifted = rgb + params.strength * shadow_weight * lift_target;

    // ── Global warm channel scale ─────────────────────────────────────────────
    //
    // 1970s color grading is characterised by a warm orange-brown cast across the
    // whole image. Boost red and green, reduce blue, to produce a warm orange base
    // tone. At strength=0 the multipliers collapse to vec3(1.0, 1.0, 1.0) — identity.
    let channel_scale = vec3<f32>(
        1.0 + params.strength * 0.12,   // red:   +12% at full strength
        1.0 + params.strength * 0.05,   // green: +5%  at full strength
        1.0 - params.strength * 0.18,   // blue:  -18% at full strength
    );
    let scaled = lifted * channel_scale;

    // ── Highlight yellow-green tint ───────────────────────────────────────────
    //
    // Bright areas of aged 1970s film often exhibit a faint yellow-green cast from
    // dye fading in the cyan and magenta layers. This adds a subtle tint to the
    // highlights by blending toward a pale yellow-green for high-luminance pixels.
    //
    // The tint weight peaks at lum=1.0 and falls to 0 at lum=0.6, so only the
    // upper part of the tonal range is affected.
    let highlight_weight = smoothstep(0.6, 1.0, lum);
    let highlight_tint   = vec3<f32>(0.88, 0.92, 0.72); // pale yellow-green in linear light
    let tint_blend       = params.strength * highlight_weight * 0.12;
    let tinted           = mix(scaled, highlight_tint, tint_blend);

    // ── Saturation reduction ──────────────────────────────────────────────────
    //
    // Aged film stocks lose saturation over time. Desaturate slightly toward
    // luminance at full strength to reproduce the soft, muted look.
    let desaturated   = mix(tinted, vec3<f32>(lum), params.strength * 0.10);

    // Do NOT clamp: preserve headroom above 1.0 for downstream shaders.
    textureStore(output_texture, coord, vec4<f32>(desaturated, pixel.a));
}
