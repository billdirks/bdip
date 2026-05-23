// The uniform struct must match the Rust LightLeakParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would produce a 32-byte struct and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct LightLeakParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: LightLeakParams;

// ---------------------------------------------------------------------------
// Light leak procedural generation
//
// The effect is composed of two additive layers:
//
//   1. Corner gradients — polynomial falloffs from the four corners using
//      warm warm-toned colors (orange, yellow-red, amber, gold). Each corner
//      emits a different hue so the effect appears varied, not symmetric.
//
//   2. Sine-based streaks — horizontal and vertical sinusoidal banding that
//      simulates light bleeding along frame edges. The band frequencies are
//      chosen to produce two to three visible streaks per axis.
//
// Both layers are combined additively and blended with the original via
// params.strength. At strength=0 the blend weight is 0 — pure identity.
// Do NOT clamp intermediate results so headroom above 1.0 is preserved for
// downstream shaders (e.g. an exposure adjustment after the light leak).
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Normalised UV in [0,1] with half-pixel offset for stable center.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // ── Layer 1: Corner gradients ────────────────────────────────────────────
    //
    // Each corner contributes a warm-toned glow whose intensity is a cubic
    // falloff of the distance from that corner.  The cubic (1-d)^3 gives a
    // smooth, concentrated pool near the corner that fades quickly outward.
    //
    // Corner assignments (chosen to produce an asymmetric, organic look):
    //   top-left     → deep orange  (r=0.95, g=0.40, b=0.05)
    //   top-right    → amber-yellow (r=0.90, g=0.60, b=0.05)
    //   bottom-left  → golden-red   (r=0.85, g=0.25, b=0.02)
    //   bottom-right → warm yellow  (r=0.92, g=0.70, b=0.08)

    let d_tl = distance(uv, vec2<f32>(0.0, 0.0));
    let d_tr = distance(uv, vec2<f32>(1.0, 0.0));
    let d_bl = distance(uv, vec2<f32>(0.0, 1.0));
    let d_br = distance(uv, vec2<f32>(1.0, 1.0));

    // Cubic falloff: saturate so negative values (d > 1) clamp to 0.
    let f_tl = max(0.0, 1.0 - d_tl) * max(0.0, 1.0 - d_tl) * max(0.0, 1.0 - d_tl);
    let f_tr = max(0.0, 1.0 - d_tr) * max(0.0, 1.0 - d_tr) * max(0.0, 1.0 - d_tr);
    let f_bl = max(0.0, 1.0 - d_bl) * max(0.0, 1.0 - d_bl) * max(0.0, 1.0 - d_bl);
    let f_br = max(0.0, 1.0 - d_br) * max(0.0, 1.0 - d_br) * max(0.0, 1.0 - d_br);

    let tl_color = vec3<f32>(0.95, 0.40, 0.05);
    let tr_color = vec3<f32>(0.90, 0.60, 0.05);
    let bl_color = vec3<f32>(0.85, 0.25, 0.02);
    let br_color = vec3<f32>(0.92, 0.70, 0.08);

    let corner_leak =
        tl_color * f_tl * 0.55
      + tr_color * f_tr * 0.40
      + bl_color * f_bl * 0.35
      + br_color * f_br * 0.50;

    // ── Layer 2: Sine-based edge streaks ─────────────────────────────────────
    //
    // Two streaks run horizontally (varying with y) and one runs vertically
    // (varying with x). A squared-sine gives brighter peaks and dark valleys
    // that match the look of a physical light-bleed band.
    //
    // Each streak is multiplied by an edge proximity weight — a linear falloff
    // from the nearest frame edge — so the streak fades to zero at the center
    // and is strongest within ~25% of the edge.
    //
    // Streak colors are warm oranges/reds consistent with the corner palette.

    // Horizontal streak: runs along the top edge, modulates on uv.y.
    let h_wave    = sin(uv.y * 6.2832 * 1.5 + 0.8);    // ~1.5 cycles per frame
    let h_band    = h_wave * h_wave;                    // squared → always ≥ 0
    let h_edge    = 1.0 - uv.y;                         // strong at top, 0 at bottom
    let h_weight  = h_band * h_edge * h_edge;           // squared edge for sharper fade
    let h_streak  = vec3<f32>(0.90, 0.45, 0.05) * h_weight * 0.30;

    // Vertical streak: runs along the right edge, modulates on uv.x.
    let v_wave    = sin(uv.x * 6.2832 * 2.0 + 1.4);    // ~2 cycles per frame
    let v_band    = v_wave * v_wave;
    let v_edge    = uv.x;                               // strong at right, 0 at left
    let v_weight  = v_band * v_edge * v_edge;
    let v_streak  = vec3<f32>(0.88, 0.35, 0.04) * v_weight * 0.25;

    // ── Combine layers ───────────────────────────────────────────────────────
    let leak = corner_leak + h_streak + v_streak;

    // Additive blend scaled by strength. At strength=0 the contribution is 0
    // (identity). The result is not clamped so downstream shaders retain the
    // full dynamic range.
    let out_rgb = pixel.rgb + leak * params.strength;

    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
