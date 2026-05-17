// Coffee Stained — single-pass procedural coffee/tea stain simulation.
//
// The effect composites an organic, brownish stain pattern over the source
// image using multiplicative blending.  The stain pattern is generated
// entirely from procedural math — no auxiliary textures are required.
//
// Stain generation (coffee ring effect):
//   Real dried coffee stains exhibit the "coffee ring effect": particles
//   carried outward by evaporating liquid concentrate at the perimeter,
//   leaving a dark ring with a relatively clear center.  The stain mask
//   replicates this by computing exponential falloff from a ring edge (not a
//   blob center).  Seven stain rings are distributed asymmetrically across
//   the frame; each has a fixed ring radius to produce organic size variation.
//
//   ring_width controls how thick (diffuse) each ring edge is.
//   inner_clarity controls whether the center is left clear (1.0 = realistic
//   ring effect) or filled in (0.0 = solid filled stain).
//
// Colour tint:
//   The stain colour is a warm, dark brown: (R=0.45, G=0.25, B=0.10) in
//   linear light.  Multiplying the source by this tint inside the stain areas
//   darkens them and shifts them toward brown, mimicking dried coffee on paper.
//   Outside the stain areas (mask ≈ 0) the source is unmodified.
//
// Blending:
//   tint_factor = mix(1.0, stain_tint, stain_mask * strength)
//   out = pixel.rgb * tint_factor
//
//   At strength=0 the tint_factor is 1.0 everywhere — pure identity.
//   At strength=1 and on a stain ring edge the tint_factor equals the stain_tint
//   colour, producing the darkened brown result.  Outside stain rings the
//   tint_factor remains 1.0 regardless of strength.
//
// Headroom:
//   Intermediate and output values are NOT clamped so that values > 1.0
//   are preserved for downstream shaders (e.g. an exposure slider applied
//   after the stain effect).

// The uniform struct must match the Rust CoffeeStainedParams layout exactly:
// four f32 fields (16 bytes total).
struct CoffeeStainedParams {
    strength:      f32,
    ring_width:    f32,
    inner_clarity: f32,
    _padding:      f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CoffeeStainedParams;

// ---------------------------------------------------------------------------
// Stain mask generation
// ---------------------------------------------------------------------------

// Stain ring centres (UV coordinates in [0, 1]).  Seven asymmetric positions
// chosen to look organic: no obvious grid, no axis symmetry.  The centres are
// fixed constants so the shader is deterministic.
const CENTRE_0: vec2<f32> = vec2<f32>(0.18, 0.22);
const CENTRE_1: vec2<f32> = vec2<f32>(0.72, 0.15);
const CENTRE_2: vec2<f32> = vec2<f32>(0.55, 0.60);
const CENTRE_3: vec2<f32> = vec2<f32>(0.30, 0.75);
const CENTRE_4: vec2<f32> = vec2<f32>(0.85, 0.48);
const CENTRE_5: vec2<f32> = vec2<f32>(0.10, 0.88);
const CENTRE_6: vec2<f32> = vec2<f32>(0.65, 0.92);

// Ring radii (UV units) — varied per stain to give organic size differences.
const RING_RADIUS_0: f32 = 0.15;
const RING_RADIUS_1: f32 = 0.12;
const RING_RADIUS_2: f32 = 0.18;
const RING_RADIUS_3: f32 = 0.13;
const RING_RADIUS_4: f32 = 0.10;
const RING_RADIUS_5: f32 = 0.14;
const RING_RADIUS_6: f32 = 0.11;

// Coffee ring stain mask for a single ring:
//   Maximum at ring_radius distance from centre (the ring perimeter), falling
//   off exponentially toward both the interior and exterior.  params.ring_width
//   controls the spread of the falloff in UV space.
//
//   params.inner_clarity blends between:
//     1.0 — interior uses natural ring falloff (realistic clear centre)
//     0.0 — interior is filled to full ring intensity (solid stain)
fn ring_blob(uv: vec2<f32>, centre: vec2<f32>, ring_radius: f32) -> f32 {
    let d = distance(uv, centre);
    let dist_from_ring = abs(d - ring_radius);
    // Clamp ring_width away from zero to prevent division by zero at the
    // slider minimum.
    let scale = 10.0 / max(params.ring_width, 0.001);
    let ring_intensity = exp(-dist_from_ring * scale);
    // For pixels inside the ring, blend between the natural falloff (clear
    // centre) and full fill intensity based on inner_clarity.
    let is_inside = select(0.0, 1.0, d < ring_radius);
    return mix(ring_intensity, 1.0, (1.0 - params.inner_clarity) * is_inside);
}

// Compute a stain mask in [0, 1] from the sum of all ring falloffs.
// The raw sum is remapped with a power curve (pow(x, 0.6)) to soften the
// transition from stained to clean regions and produce a more organic edge.
fn stain_mask(uv: vec2<f32>) -> f32 {
    let raw = ring_blob(uv, CENTRE_0, RING_RADIUS_0)
            + ring_blob(uv, CENTRE_1, RING_RADIUS_1)
            + ring_blob(uv, CENTRE_2, RING_RADIUS_2)
            + ring_blob(uv, CENTRE_3, RING_RADIUS_3)
            + ring_blob(uv, CENTRE_4, RING_RADIUS_4)
            + ring_blob(uv, CENTRE_5, RING_RADIUS_5)
            + ring_blob(uv, CENTRE_6, RING_RADIUS_6);

    // Clamp to [0, 1] before the power remap.  The sum can exceed 1.0 where
    // rings overlap; saturate(raw) keeps the remap physically meaningful.
    let clamped = min(raw, 1.0);
    // Power curve < 1.0 spreads low values upward, giving wider soft edges.
    return pow(clamped, 0.6);
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Normalised UV in [0, 1] with half-pixel offset for a stable texel centre.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Stain mask: 0 = clean area, 1 = full ring-edge intensity.
    let mask = stain_mask(uv);

    // Stain tint colour in linear light.  A dark warm brown that, when
    // multiplied against the source, darkens and shifts it toward coffee tone.
    // Values chosen so the stain reads clearly on both light and dark sources.
    let stain_tint = vec3<f32>(0.45, 0.25, 0.10);

    // Blend factor per channel: 1.0 in clean areas → stain_tint on ring edges.
    // The mix weight is (mask * strength), so at strength=0 the factor is always
    // 1.0 (identity) and at strength=1 it ranges from 1.0 (clean) to stain_tint
    // (full ring-edge intensity).
    let blend_weight = mask * params.strength;
    let tint_factor  = mix(vec3<f32>(1.0, 1.0, 1.0), stain_tint, blend_weight);

    // Multiplicative tint — darkens and warms the stain area.
    // Do NOT clamp: preserve headroom for downstream shaders.
    let out_rgb = pixel.rgb * tint_factor;

    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
