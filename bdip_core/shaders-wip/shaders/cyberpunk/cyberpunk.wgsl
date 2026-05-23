// The uniform struct must match the Rust CyberpunkParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would make the struct 32 bytes and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct CyberpunkParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CyberpunkParams;

// Rec. 709 luminance coefficients (linear light).
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Raise a per-channel value using a power curve. exponent > 1 darkens midtones;
// exponent < 1 brightens them. Applied before blending so the curve affects only
// the graded copy, not the original.
fn apply_curve(v: f32, exponent: f32) -> f32 {
    return pow(max(v, 0.0), exponent);
}

// Boost saturation in linear light around the neon range.
// The neon range is loosely defined as highly saturated colours (high chroma).
// Strategy: measure chroma as deviation from luminance; amplify that deviation
// by a fixed neon_boost factor. The boost is strongest for already-saturated
// colours and falls off naturally for neutrals and near-neutrals.
fn boost_neon_saturation(rgb: vec3<f32>, lum: f32, neon_boost: f32) -> vec3<f32> {
    let chroma_vec = rgb - vec3<f32>(lum);
    return rgb + chroma_vec * neon_boost;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);
    let rgb   = pixel.rgb;

    // ── Step 1: Shadow deepening ────────────────────────────────────────────
    // Apply a slightly steeper-than-linear curve (exponent > 1) per channel.
    // This pulls shadows down while leaving highlights mostly intact, creating
    // the high-contrast look characteristic of cyberpunk imagery.
    // The curve is chosen so that strength=1 deepens shadows noticeably but
    // does not crush the entire midtone range.
    let shadow_exponent = 1.0 + params.strength * 0.6;
    let curved = vec3<f32>(
        apply_curve(rgb.r, shadow_exponent),
        apply_curve(rgb.g, shadow_exponent),
        apply_curve(rgb.b, shadow_exponent),
    );

    // ── Step 2: Cyan and magenta channel boost ──────────────────────────────
    // Cyan = G + B with suppressed R; Magenta = R + B with suppressed G.
    // Rather than a full colour matrix, a targeted per-channel adjustment
    // achieves the same effect: slightly reduce R to push the image toward
    // cyan-magenta balance, and boost B to reinforce both neon hues.
    // The coefficients are empirically chosen to be perceptually balanced.
    let cm_r = curved.r * (1.0 - params.strength * 0.08);
    let cm_g = curved.g * (1.0 + params.strength * 0.04);
    let cm_b = curved.b * (1.0 + params.strength * 0.14);
    let neon_rgb = vec3<f32>(cm_r, cm_g, cm_b);

    // ── Step 3: Teal-to-orange split tone ──────────────────────────────────
    // Shadows tilt toward teal (blue-green); highlights tilt toward orange.
    // This is the same luminance-driven split used by the teal_and_orange
    // shader, but with smaller magnitudes suited to the cyberpunk palette.
    let lum = luminance(neon_rgb);

    let shadow_w    = smoothstep(0.5, 0.0, lum);
    let highlight_w = smoothstep(0.5, 1.0, lum);

    // Teal: blue-green tint for shadows.
    // Orange: warm tint for highlights (less intense than teal to keep neon feel).
    let teal_target   = vec3<f32>(0.0,  0.22, 0.30);
    let orange_target = vec3<f32>(0.32, 0.14, 0.0);

    let split_strength = params.strength * 0.5;
    let teal_contrib   = split_strength * shadow_w    * (teal_target   - neon_rgb);
    let orange_contrib = split_strength * highlight_w * (orange_target - neon_rgb);
    let split_rgb = neon_rgb + teal_contrib + orange_contrib;

    // ── Step 4: Neon saturation boost ──────────────────────────────────────
    // Amplify chroma (deviation from luminance) to push already-saturated
    // colours further toward their neon peaks without affecting neutrals.
    let split_lum = luminance(split_rgb);
    let boosted = boost_neon_saturation(split_rgb, split_lum, params.strength * 0.35);

    // ── Step 5: Blend with original ────────────────────────────────────────
    // At strength=0.0 the output equals the input exactly (identity).
    // At strength=1.0 the full graded result is used.
    let out_rgb = mix(rgb, boosted, params.strength);

    // Do NOT clamp: preserve headroom above 1.0 for downstream shaders.
    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
