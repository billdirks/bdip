// The uniform struct must match the Rust CrossProcessParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would make the struct 32 bytes and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct CrossProcessParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CrossProcessParams;

// Per-channel cross-process curve approximations.
//
// These are polynomial/power-curve approximations of the characteristic tone
// curves that result from developing film in the wrong chemistry (e.g. slide
// film in C-41 negative chemistry).
//
// Red channel: boost in highlights via a power curve that lifts values above
// midtone. A power < 1.0 lifts shadows/midtones, but we want primarily a
// highlight boost with a mild warm cast throughout.
//
// We use pow(v, 0.85) which brightens across the range and adds a warm red cast.
fn curve_red(v: f32) -> f32 {
    // pow(v, 0.85): lifts overall with stronger effect in highlights.
    // Clamping prevents negative inputs from producing NaN in pow().
    return pow(clamp(v, 0.0, 1e9), 0.85);
}

// Green channel: S-curve approximation that boosts midtones and crushes shadows.
// A cubic S-curve: 3*v^2 - 2*v^3 (smoothstep) passes through (0,0) and (1,1)
// and has a steeper slope through midtones — boosting midtone contrast.
// We apply it only to [0,1] for the S-curve and fall back to linear for values
// above 1.0 to preserve headroom.
fn curve_green(v: f32) -> f32 {
    let t = clamp(v, 0.0, 1.0);
    // Smoothstep S-curve: lifts midtones, slightly crushes shadows.
    let s = t * t * (3.0 - 2.0 * t);
    // For headroom values above 1.0, pass through linearly.
    return select(s, v, v > 1.0);
}

// Blue channel: inverted/shifted in shadows to introduce a cool cast.
// Cross-processed slide film typically exhibits cyan/blue shadows.
// We use: 1.0 - pow(1.0 - v, 1.3), which pulls the shadow end toward higher
// blue values (cool shadows) while compressing the highlights slightly.
// This maps 0->0 and 1->1 but inverts the curve shape relative to identity.
fn curve_blue(v: f32) -> f32 {
    let t = clamp(v, 0.0, 1.0);
    // Shadow lift: complementary power curve — raises the shadow floor.
    let b = 1.0 - pow(1.0 - t, 1.3);
    // For headroom values above 1.0, pass through linearly.
    return select(b, v, v > 1.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);
    let rgb   = pixel.rgb;

    // Apply per-channel cross-process curves.
    let processed = vec3<f32>(
        curve_red(rgb.r),
        curve_green(rgb.g),
        curve_blue(rgb.b),
    );

    // Blend between the original and processed image based on strength.
    // At strength=0.0 the output equals the input (identity).
    // At strength=1.0 the full cross-process look is applied.
    let out_rgb = mix(rgb, processed, params.strength);

    // Do NOT clamp: preserve headroom above 1.0 for downstream shaders.
    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
