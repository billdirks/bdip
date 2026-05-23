// Underwater — caustic pass (pass 2 of 2).
//
// Generates a procedural caustic light pattern and blends the full underwater
// effect (tinted image + caustic overlay) against the original source image,
// controlled by `strength`.
//
// Caustic generation:
//   Caustics are the bright wavy lines seen on the seafloor from refracted
//   sunlight. They are approximated via sine-based interference: two sets of
//   sine waves at different frequencies and orientations are multiplied together.
//   This produces a network of bright ridges that mimics the intersection
//   pattern of refracted light rays without requiring an external texture.
//
//   The pattern is intentionally computed in UV space (normalised pixel
//   coordinates) so that it scales correctly with image resolution. UV
//   coordinates are scaled to increase the visual frequency of the caustic
//   to a density that reads as water light on typical image sizes.
//
// Final composite formula:
//   caustic_rgb = tinted_rgb + caustic_pattern * caustic_intensity
//   out_rgb     = mix(source_rgb, caustic_rgb, strength)
//
// At `strength=0.0` the formula reduces to `source_rgb`, making the entire
// two-pass chain a strict identity regardless of depth or caustic_intensity.
//
// Do NOT clamp the upper bound — preserve >1.0 linear-light headroom for
// downstream shaders in the pipeline.

struct UnderwaterParams {
    depth:             f32,
    caustic_intensity: f32,
    strength:          f32,
    _padding:          f32,
}

// Two inputs (source + tinted scratch), so output is at binding 2.
@group(0) @binding(0) var input_source:  texture_2d<f32>;
@group(0) @binding(1) var input_tinted:  texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: UnderwaterParams;

// Procedural caustic value at normalised UV coordinates.
//
// Two sine-wave pairs are evaluated at different frequencies and diagonal
// angles. Their products are then summed and remapped to [0, 1]. The
// interference between the two pairs creates the characteristic network
// of bright filaments seen in shallow water.
fn caustic(uv: vec2<f32>) -> f32 {
    // Scale UV to raise spatial frequency to a visible caustic density.
    let s = uv * 8.0;

    // Wave pair A: roughly diagonal at 45°.
    let a = sin(s.x * 1.7 + s.y * 1.1) * sin(s.x * 1.1 - s.y * 1.7);
    // Wave pair B: steeper angle to produce crossing interference.
    let b = sin(s.x * 2.3 - s.y * 0.9) * sin(s.x * 0.9 + s.y * 2.3);

    // Sum, remap from [-1, 1] to [0, 1], then raise to a power to sharpen
    // the bright ridges and darken the troughs (emphasises the caustic look).
    let combined = (a + b) * 0.5;        // in [-1, 1]
    let remapped = combined * 0.5 + 0.5; // in [ 0, 1]
    return pow(remapped, 3.0);           // sharpened; bright ridges stand out
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src    = textureLoad(input_source, coord, 0);
    let tinted = textureLoad(input_tinted, coord, 0);

    // Normalised UV in [0, 1] for position-independent caustic generation.
    let uv = vec2<f32>(gid.xy) / vec2<f32>(f32(dims.x), f32(dims.y));

    let c = caustic(uv);

    // Caustic overlay: add the procedural pattern onto the tinted base.
    // The pattern is tinted toward white-blue to match sunlight filtering
    // through water (slight blue emphasis to complement the tint pass).
    let caustic_color = vec3<f32>(c * 0.7, c * 0.85, c) * params.caustic_intensity;
    let caustic_rgb   = tinted.rgb + caustic_color;

    // Final blend: mix the original source with the caustic composite based on
    // strength. At strength=0.0 this returns the source exactly (identity).
    let out_rgb = mix(src.rgb, caustic_rgb, params.strength);

    // Lower-bound clamp prevents floating-point noise from producing negative
    // values; the upper bound is left open to preserve linear-light headroom.
    let safe_rgb = max(out_rgb, vec3<f32>(0.0));

    textureStore(output_texture, coord, vec4<f32>(safe_rgb, src.a));
}
