// Coffee Stained — single-pass procedural coffee/tea stain simulation.
//
// The effect composites an organic, brownish stain pattern over the source
// image using multiplicative blending.  The stain pattern is generated
// entirely from procedural math — no auxiliary textures are required.
//
// Stain generation:
//   The stain mask is built from several Worley-style (distance-to-nearest-
//   point) falloffs, each centred on a hard-coded pseudo-random point.  Each
//   point's falloff is a smooth exponential that produces soft, blob-shaped
//   stains.  The blobs are then summed and remapped to [0, 1] using a
//   power curve that concentrates the darkening in the blob centres.
//
//   Seven stain centres are distributed asymmetrically across the frame so
//   no obvious tiling or symmetry is visible.  The centres are fixed constants
//   so the shader is deterministic and produces no temporal flicker.
//
// Colour tint:
//   The stain colour is a warm, dark brown: (R=0.45, G=0.25, B=0.10) in
//   linear light.  Multiplying the source by this tint inside the stain areas
//   darkens them and shifts them toward brown, mimicking wet coffee dried on
//   paper.  Outside the stain areas (mask ≈ 0) the source is unmodified.
//
// Blending:
//   tint_factor = mix(1.0, stain_tint, stain_mask * strength)
//   out = pixel.rgb * tint_factor
//
//   At strength=0 the tint_factor is 1.0 everywhere — pure identity.
//   At strength=1 and inside a stain blob the tint_factor equals the
//   stain_tint colour, producing the darkened brown result.
//   Outside stain blobs the tint_factor remains 1.0 regardless of strength.
//
// Headroom:
//   Intermediate and output values are NOT clamped so that values > 1.0
//   are preserved for downstream shaders (e.g. an exposure slider applied
//   after the stain effect).

// The uniform struct must match the Rust CoffeeStainedParams layout exactly:
// one f32 (strength) followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte
// alignment vec3 would introduce in WGSL, which would produce a 32-byte
// struct mismatched with the 16-byte buffer allocated from the Rust side.
struct CoffeeStainedParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CoffeeStainedParams;

// ---------------------------------------------------------------------------
// Stain mask generation
// ---------------------------------------------------------------------------

// Stain blob centres (UV coordinates in [0, 1]).  Seven asymmetric positions
// chosen to look organic: no obvious grid, no axis symmetry, blobs of varying
// implied size determined by the BLOB_SCALE constant below.
const CENTRE_0: vec2<f32> = vec2<f32>(0.18, 0.22);
const CENTRE_1: vec2<f32> = vec2<f32>(0.72, 0.15);
const CENTRE_2: vec2<f32> = vec2<f32>(0.55, 0.60);
const CENTRE_3: vec2<f32> = vec2<f32>(0.30, 0.75);
const CENTRE_4: vec2<f32> = vec2<f32>(0.85, 0.48);
const CENTRE_5: vec2<f32> = vec2<f32>(0.10, 0.88);
const CENTRE_6: vec2<f32> = vec2<f32>(0.65, 0.92);

// Controls the spread of each stain blob.  Larger values produce smaller,
// tighter blobs; smaller values spread the blobs wider.  8.0 produces blobs
// that cover roughly 10–20 % of the frame area individually.
const BLOB_SCALE: f32 = 8.0;

// Each blob contributes an exponential falloff from its centre.  The falloff
// is exp(-d * BLOB_SCALE) where d is the Euclidean distance in UV space.
// This is always in [0, 1] (maximum 1.0 at the centre) and decays smoothly.
fn blob(uv: vec2<f32>, centre: vec2<f32>) -> f32 {
    let d = distance(uv, centre);
    return exp(-d * BLOB_SCALE);
}

// Compute a stain mask in [0, 1] from the sum of all blob falloffs.
// The raw sum is remapped with a power curve (pow(x, 0.6)) to soften the
// transition from stained to clean regions and produce a more organic edge.
fn stain_mask(uv: vec2<f32>) -> f32 {
    let raw = blob(uv, CENTRE_0)
            + blob(uv, CENTRE_1)
            + blob(uv, CENTRE_2)
            + blob(uv, CENTRE_3)
            + blob(uv, CENTRE_4)
            + blob(uv, CENTRE_5)
            + blob(uv, CENTRE_6);

    // Clamp to [0, 1] before the power remap.  The sum can exceed 1.0 where
    // blobs overlap; saturate(raw) keeps the remap physically meaningful.
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

    // Stain mask: 0 = clean area, 1 = dense stain centre.
    let mask = stain_mask(uv);

    // Stain tint colour in linear light.  A dark warm brown that, when
    // multiplied against the source, darkens and shifts it toward coffee tone.
    // Values chosen so the stain reads clearly on both light and dark sources.
    let stain_tint = vec3<f32>(0.45, 0.25, 0.10);

    // Blend factor per channel: 1.0 in clean areas → stain_tint in stain
    // centres.  The mix weight is (mask * strength), so at strength=0 the
    // factor is always 1.0 (identity) and at strength=1 it ranges from 1.0
    // (clean) to stain_tint (full stain centre).
    let blend_weight = mask * params.strength;
    let tint_factor  = mix(vec3<f32>(1.0, 1.0, 1.0), stain_tint, blend_weight);

    // Multiplicative tint — darkens and warms the stain area.
    // Do NOT clamp: preserve headroom for downstream shaders.
    let out_rgb = pixel.rgb * tint_factor;

    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
