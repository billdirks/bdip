// The uniform struct must match the Rust SunFlareParams layout exactly:
// 8 × f32 = 32 bytes. Using individual f32 fields avoids the 16-byte alignment
// that vec3<f32> introduces in WGSL, which would extend the struct to 48 bytes
// and mismatch the 32-byte buffer the engine builds from the Rust side.
struct SunFlareParams {
    flare_x:   f32,  // normalised [0, 1] — horizontal position of the light source
    flare_y:   f32,  // normalised [0, 1] — vertical position of the light source
    intensity: f32,  // overall brightness multiplier; 0.0 → identity (no-op)
    size:      f32,  // scale factor for the entire flare complex
    tint_r:    f32,  // linear-RGB red component of the flare colour tint
    tint_g:    f32,  // linear-RGB green component of the flare colour tint
    tint_b:    f32,  // linear-RGB blue component of the flare colour tint
    _padding:  f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: SunFlareParams;

// ---------------------------------------------------------------------------
// Smooth falloff helper
//
// Returns an exponential falloff curve that peaks at 1.0 when dist == 0 and
// decays toward 0.  The sharpness parameter controls how quickly it falls off.
// ---------------------------------------------------------------------------
fn radial_falloff(dist: f32, sharpness: f32) -> f32 {
    return exp(-dist * dist * sharpness);
}

// ---------------------------------------------------------------------------
// Sun Flare effect
//
// Three additive contributions are composited and blended via params.intensity:
//
//   1. Primary glow — a large, soft radial falloff centred on the flare source.
//      Mimics the overexposed halo that surrounds a bright light source.
//
//   2. Radial streaks — starburst lines radiating from the source.  Generated
//      by computing the angular proximity of each pixel to the set of N_STREAKS
//      evenly-spaced spoke directions.  A smooth angular falloff produces thin,
//      glowing lines.
//
//   3. Secondary lens artifacts — smaller bright discs ("ghost" reflections)
//      placed along the axis from the image centre through the flare source.
//      Real lens ghosts are formed by internal reflections between glass elements
//      and appear on the opposite side of the optical axis from the source.
//      Each ghost is positioned at a fixed fraction of the centre→source vector,
//      where fractions > 1 place the ghost beyond the source (reflective bounce)
//      and fractions < 0 place it on the opposite side.
//
// All contributions are in linear light.  Nothing is clamped so downstream
// shaders retain headroom above 1.0.  At intensity=0.0 the contribution is
// exactly 0 and the shader is a pure pass-through.
// ---------------------------------------------------------------------------

// Number of starburst spokes.  8 produces the classic 8-pointed sun shape.
const N_STREAKS: i32 = 8;

// Number of secondary lens-ghost discs along the flare axis.
const N_GHOSTS: i32 = 5;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Normalised UV in [0, 1] with half-pixel offset for a stable centre.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Account for aspect ratio so distances are measured in screen space rather
    // than normalised UV space (avoids distorted ellipses on non-square images).
    let aspect = f32(dims.x) / f32(dims.y);

    // Aspect-corrected source position.
    let src = vec2<f32>(params.flare_x * aspect, params.flare_y);
    // Aspect-corrected pixel position.
    let pos = vec2<f32>(uv.x * aspect, uv.y);

    // Vector from the flare source to the current pixel, and its length.
    let delta   = pos - src;
    let dist    = length(delta);

    // Effective radius scale; larger size → wider flare complex.
    // Dividing by size makes the falloff numerically larger at a given pixel
    // distance, which spreads the visible contribution across more pixels.
    let inv_size = 1.0 / max(params.size, 0.001);

    // ── 1. Primary glow ──────────────────────────────────────────────────────
    //
    // A broad soft disc centred on the source.  The sharpness value 80 gives a
    // falloff radius of roughly 0.1 in normalised (aspect-corrected) units.
    let glow_dist     = dist * inv_size;
    let primary_glow  = radial_falloff(glow_dist, 80.0);

    // An additional tight highlight for the very centre of the bright spot.
    let core_glow     = radial_falloff(glow_dist, 2000.0);

    // ── 2. Radial streaks (starburst) ────────────────────────────────────────
    //
    // For each spoke direction, compute the shortest angular distance between
    // the delta vector and the spoke.  A narrow angular falloff (sharpness 400)
    // produces thin glowing lines.  The streak brightness is attenuated by a
    // radial envelope that fades at the very centre (to avoid a fully-lit core)
    // and at large distances (finite streak length).
    let angle         = atan2(delta.y, delta.x);
    let spoke_step    = 3.14159265 / f32(N_STREAKS); // π / N → evenly spaced

    var streak_sum = 0.0;
    for (var i = 0; i < N_STREAKS; i++) {
        let spoke_angle = f32(i) * spoke_step;
        // Shortest angular distance (modular, wrapped to [-π/2, π/2]).
        var ang_diff = angle - spoke_angle;
        // Reduce modulo π to compare against both the spoke and its opposite.
        ang_diff = ang_diff - round(ang_diff / 3.14159265) * 3.14159265;
        let angular_weight = exp(-ang_diff * ang_diff * 400.0);
        streak_sum += angular_weight;
    }

    // Radial envelope for streaks: bell-shaped, peaking around dist = 0.05,
    // scaled by size.  The peak distance of 0.05 is in aspect-corrected space.
    let streak_peak   = 0.05 * params.size;
    let streak_dr     = dist - streak_peak;
    let streak_radial = exp(-streak_dr * streak_dr / (2.0 * 0.04 * params.size * params.size));
    let streak        = streak_sum * streak_radial * 0.4;

    // ── 3. Secondary lens artifacts (ghosts) ─────────────────────────────────
    //
    // Ghosts are positioned along the centre→source axis at fractions of the
    // centre-to-source distance.  A fraction of 0 places the ghost at the image
    // centre; a fraction > 1 places it beyond the source; a negative fraction
    // places it on the opposite side of centre from the source.
    //
    // Centre of the image in aspect-corrected space.
    let img_centre = vec2<f32>(0.5 * aspect, 0.5);
    // Vector from image centre to the flare source.
    let centre_to_src = src - img_centre;

    // Ghost fractions along the centre→source axis.
    // These values are chosen to space the ghosts across a range that produces
    // classic lens-ghost patterns visible at typical flare positions.
    let ghost_fractions = array<f32, 5>(-0.3, 0.15, 0.6, 1.3, 1.8);
    // Ghost radii (in size-normalised space).
    let ghost_radii     = array<f32, 5>(0.04, 0.025, 0.035, 0.02, 0.015);

    var ghost_sum = 0.0;
    for (var j = 0; j < N_GHOSTS; j++) {
        let ghost_centre    = img_centre + centre_to_src * ghost_fractions[j];
        let ghost_dist      = length(pos - ghost_centre);
        let ghost_r         = ghost_radii[j] * params.size;
        let ghost_sharpness = 1.0 / max(ghost_r * ghost_r, 0.0001);
        ghost_sum += radial_falloff(ghost_dist, ghost_sharpness) * 0.6;
    }

    // ── Combine contributions ─────────────────────────────────────────────────
    //
    // Sum the three layers.  The weights keep the overall brightness range
    // reasonable at intensity=1.0 without clamping.
    let flare_mono = primary_glow * 1.0 + core_glow * 2.0 + streak + ghost_sum;

    // Apply colour tint: the tint vector scales each channel independently.
    // A white tint (1, 1, 1) leaves the hue unchanged; coloured tints bias
    // the flare toward that hue.
    let tint = vec3<f32>(params.tint_r, params.tint_g, params.tint_b);

    // Additive blend scaled by intensity.  At intensity=0, flare_rgb = 0
    // (identity).  Not clamped — downstream shaders retain full dynamic range.
    let flare_rgb = flare_mono * tint * params.intensity;
    let out_rgb   = pixel.rgb + flare_rgb;

    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
