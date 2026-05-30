// Frost Ice — single-pass procedural vignette texture mask effect.
//
// Simulates the appearance of a frost-covered or icy glass window by combining
// three elements:
//
//   1. Radial vignette mask — a smooth radial falloff that grows from the edges
//      inward.  The `coverage` parameter controls how far the frost extends from
//      the edge toward the center.
//
//   2. UV distortion — near the frost region, UV coordinates are displaced by
//      domain-warped noise to simulate the way ice crystals refract and distort
//      the view.  The `distortion` parameter controls the displacement amplitude.
//
//   3. Cold blue tint — an icy blue-white colour is blended over the distorted
//      source wherever frost is present.  The `strength` parameter controls the
//      overall opacity of the entire effect (0.0 = identity, 1.0 = full frost).
//
// Procedural frost texture is produced by two layers of domain-warped fract-sin
// noise, which approximates the dendritic crystal structure of window frost
// without requiring any auxiliary texture.
//
// Identity condition: when `strength` is 0.0, the output is exactly the source
// pixel (mix factor is 0 → all three elements are multiplied out).

struct FrostIceParams {
    coverage:   f32, // frost inward reach: 0.0=edges only, 1.0=full frame
    distortion: f32, // UV warp amplitude: 0.0=none, 1.0=heavy distortion
    strength:   f32, // effect opacity: 0.0=identity, 1.0=full frost
    _padding:   f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: FrostIceParams;

// ---------------------------------------------------------------------------
// Procedural noise helpers
// ---------------------------------------------------------------------------

// Deterministic pseudo-random scalar in [0, 1) from a 2-D seed.
fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// Smooth value noise in [0, 1) on a unit grid.
// Uses bilinear interpolation between the four corner hash values of the
// grid cell containing `p`, smoothed with a quintic ease curve to avoid
// visible grid artifacts.
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // Quintic smoothstep for C2-continuous interpolation.
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Two-octave domain-warped noise.  The first octave warps the input coordinates
// before the second octave samples them, producing the irregular branching
// patterns that characterise frost crystal growth.
fn frost_noise(uv: vec2<f32>) -> f32 {
    // First octave at moderate scale — broad crystal structure.
    let n0 = value_noise(uv * 6.0);

    // Warp the coordinates by the first octave before sampling the second.
    // The warp offsets are orthogonal (90° apart) to avoid a directional bias.
    let warp = vec2<f32>(
        value_noise(uv * 6.0 + vec2<f32>(1.7, 9.2)),
        value_noise(uv * 6.0 + vec2<f32>(8.3, 2.8)),
    );
    let warped_uv = uv + (warp - 0.5) * 0.4;

    // Second octave at higher frequency — fine crystal detail.
    let n1 = value_noise(warped_uv * 12.0);

    // Combine: 60 % coarse structure + 40 % fine detail.
    return n0 * 0.6 + n1 * 0.4;
}

// ---------------------------------------------------------------------------
// Main dispatch
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<u32>(global_id.xy);

    // Normalised UV in [0, 1] with half-pixel offset for pixel centres.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // --- Radial frost mask ---
    //
    // Compute the distance from the nearest edge (0 at edge, 0.5 at center).
    // `edge_dist` increases inward; `coverage` scales how far the frost extends.
    // At coverage=0 the mask is 0 everywhere (no frost).
    // At coverage=1 the mask reaches the very center.
    let edge_dist = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));

    // Map coverage [0,1] to an inward reach of [0, 0.5] in UV space.
    let frost_reach = params.coverage * 0.5;

    // frost_mask is 1 at the edge and falls to 0 at `frost_reach` from the edge.
    // smoothstep produces a soft transition rather than a hard cutoff.
    let frost_mask = 1.0 - smoothstep(0.0, frost_reach + 0.001, edge_dist);

    // --- Procedural frost noise at this UV position ---
    let noise_val = frost_noise(uv);

    // --- UV distortion ---
    //
    // Displace the sampling UV by noise to simulate ice-crystal refraction.
    // The displacement is scaled by both `distortion` and the frost mask so
    // that distortion only appears where frost is present.
    //
    // Maximum displacement is 0.05 UV units (5 % of image width/height) at
    // distortion=1.0, keeping the warp visually plausible rather than chaotic.
    let max_disp: f32 = 0.05;
    let disp_amount = params.distortion * frost_mask * max_disp;

    // Derive two orthogonal noise values for X and Y displacement.
    let disp_x = value_noise(uv * 8.0 + vec2<f32>(3.1, 5.7)) - 0.5;
    let disp_y = value_noise(uv * 8.0 + vec2<f32>(7.4, 1.2)) - 0.5;
    let distorted_uv = uv + vec2<f32>(disp_x, disp_y) * disp_amount * 2.0;

    // Clamp distorted UV to the valid [0, 1] range to avoid out-of-bounds reads.
    let sample_uv = clamp(distorted_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let sample_coord = vec2<i32>(sample_uv * vec2<f32>(dims));
    let clamped_coord = vec2<i32>(
        clamp(sample_coord.x, 0, i32(dims.x) - 1),
        clamp(sample_coord.y, 0, i32(dims.y) - 1),
    );
    let source_pixel = textureLoad(src_texture, clamped_coord, 0);

    // --- Frost colour ---
    //
    // Ice and window frost appear as a translucent, slightly blue-white surface.
    // The base frost colour is a pale blue-white in linear light.
    // Noise modulates the brightness to simulate the uneven thickness of ice
    // crystals: thicker deposits are more opaque and brighter.
    let frost_base = vec3<f32>(0.85, 0.92, 1.0);   // pale icy blue-white
    let frost_colour = frost_base * (0.6 + noise_val * 0.4);

    // --- Composite ---
    //
    // Blend the frost colour over the (distorted) source using the frost mask.
    // `frost_blend` is the mix factor between source and frost: higher where the
    // mask is strong and the noise is high (dense ice crystal coverage).
    let frost_blend = frost_mask * (0.5 + noise_val * 0.5);
    let frosted_rgb = mix(source_pixel.rgb, frost_colour, frost_blend);

    // `strength` is the final global opacity: 0.0 returns the original source
    // unchanged (identity), 1.0 applies the full frost effect.
    let out_rgb = mix(source_pixel.rgb, frosted_rgb, params.strength);

    // Do NOT clamp — preserve headroom above 1.0 for downstream shaders.
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, source_pixel.a));
}
