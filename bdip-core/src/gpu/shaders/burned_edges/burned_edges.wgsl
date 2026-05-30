// Burned Edges shader
//
// Simulates a photograph that has been burned or scorched around the perimeter.
// The effect darkens and tints the edges, with an organic, uneven burn look
// produced by layering procedural noise over the edge-distance falloff.
//
// Design:
//
//   1. Edge distance: each pixel's "burn candidate" value is based on how close
//      it is to any of the four edges, using the minimum of the four edge
//      distances mapped to a normalised [0, 1] range.  This gives a rectangular
//      perimeter shape rather than the circular vignette shape used by the
//      standard Vignette shader.
//
//   2. Organic noise: a two-level procedural hash perturbs the burn boundary
//      so it is irregular rather than a smooth gradient.  The first (coarse)
//      level simulates large charred patches; the second (fine) level adds
//      small flame-edge texture.
//
//   3. Color tint: the burn mixes toward a charred color that blends from pure
//      black (tint = 0.0) to a warm charred brown/amber (tint = 1.0).  Values
//      between 0 and 1 interpolate continuously, allowing the user to dial in
//      the exact char tone they want.
//
//   4. Identity: when `intensity` is 0.0 the shader is a pure passthrough,
//      regardless of the other parameter values.

struct BurnedEdgesParams {
    /// Blend weight of the burn overlay. 0.0 = no effect (identity).
    intensity: f32,
    /// How far the burn extends inward from each edge, in normalised image
    /// coordinates.  0.0 = no burn, 1.0 = burn reaches the center.
    radius:    f32,
    /// Width of the transition zone between unburned and fully burned regions.
    /// 0.0 = hard edge, 1.0 = fully soft.
    softness:  f32,
    /// Warm charred tint amount.  0.0 = pure black char, 1.0 = warm brown char.
    tint:      f32,
}

@group(0) @binding(0) var src_texture:     texture_2d<f32>;
@group(0) @binding(1) var dst_texture:     texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: BurnedEdgesParams;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Integer hash — maps a 2-D cell coordinate to a pseudo-random float in [0, 1).
// Used to build spatially-stable, uncorrelated noise without an aux texture.
fn hash2(a: u32, b: u32) -> f32 {
    var h: u32 = a * 1664525u + b * 22695477u + 1013904223u;
    h ^= h >> 13u;
    h *= 0x9e3779b9u;
    h ^= h >> 15u;
    return f32(h & 0xFFFFu) / 65536.0;
}

// Smooth bilinear interpolation of hash2 over a grid of cells, producing
// continuous value noise rather than a blocky cell pattern.
//
// `p` is a continuous 2-D position in "cell units."
fn value_noise(p: vec2<f32>) -> f32 {
    let i = vec2<u32>(u32(floor(p.x)), u32(floor(p.y)));
    let f = fract(p);

    // Fetch the four corner values.
    let v00 = hash2(i.x,     i.y    );
    let v10 = hash2(i.x + 1u, i.y    );
    let v01 = hash2(i.x,     i.y + 1u);
    let v11 = hash2(i.x + 1u, i.y + 1u);

    // Hermite smoothing for the interpolation weights.
    let u = f * f * (3.0 - 2.0 * f);

    return mix(mix(v00, v10, u.x), mix(v01, v11, u.x), u.y);
}

// Two-octave fractional Brownian motion (fBm) noise for organic variation.
// The two octaves give both large charred-patch structure and fine flame texture.
fn fbm(p: vec2<f32>) -> f32 {
    return 0.6 * value_noise(p) + 0.4 * value_noise(p * 2.5 + vec2<f32>(17.3, 5.7));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<u32>(gid.xy);
    let src   = textureLoad(src_texture, coord, 0);

    // ── 1. Edge distance ──────────────────────────────────────────────────────
    //
    // Normalised UV in [0, 1].
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Distance to each of the four edges.
    let d_left   = uv.x;
    let d_right  = 1.0 - uv.x;
    let d_top    = uv.y;
    let d_bottom = 1.0 - uv.y;

    // Minimum distance to any edge — this drives the rectangular burn shape.
    let edge_dist = min(min(d_left, d_right), min(d_top, d_bottom));

    // ── 2. Organic noise perturbation ─────────────────────────────────────────
    //
    // Sample fBm at two scales aligned to image-space pixel coordinates.
    // The cell size (8 cells across the image) was chosen so the noise patches
    // are large enough to look like char blotches but not so large that the
    // edge becomes a single uniform band.
    let noise_scale_coarse: f32 = 8.0;
    let noise_scale_fine:   f32 = 20.0;

    let noise_pos_c = uv * noise_scale_coarse;
    let noise_pos_f = uv * noise_scale_fine;

    // Combine coarse and fine noise; shift center to [−0.5, +0.5] so the
    // displacement is bidirectional (inward and outward perturbation of the
    // burn boundary).
    let noise_val = fbm(noise_pos_c) * 0.65 + fbm(noise_pos_f) * 0.35;
    let noise_disp = (noise_val - 0.5) * params.radius * 0.55;

    // Perturb the effective edge distance.  Positive displacement pushes the
    // boundary outward (extends the burn inward toward the center).
    let perturbed_dist = edge_dist - noise_disp;

    // ── 3. Burn mask ──────────────────────────────────────────────────────────
    //
    // The burn starts at edge_dist = radius and fades in over `softness`.
    // A pixel is fully burned when perturbed_dist < (radius - softness), and
    // unburned when perturbed_dist > radius.
    //
    // softness is expressed as a fraction of radius (clamped to avoid division
    // artifacts when radius ≈ 0).
    let r = params.radius;
    let soft_width = r * clamp(params.softness, 0.001, 1.0);
    let burn_mask = 1.0 - smoothstep(r - soft_width, r, perturbed_dist);

    // ── 4. Charred color ──────────────────────────────────────────────────────
    //
    // The char color blends from pure black (tint=0) to a warm charred brown
    // (tint=1).  The warm brown uses approximate linear-light values for a
    // dark amber/umber tone (roughly sRGB #2a1200 converted to linear).
    let char_color = mix(vec3<f32>(0.0, 0.0, 0.0),
                         vec3<f32>(0.018, 0.005, 0.0),
                         params.tint);

    // ── 5. Composite ──────────────────────────────────────────────────────────
    //
    // Blend the source color toward the char color according to the burn mask.
    // When intensity = 0.0 the output equals the source unchanged (identity).
    let burned_rgb = mix(src.rgb, char_color, burn_mask);
    let out_rgb    = mix(src.rgb, burned_rgb, params.intensity);

    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
