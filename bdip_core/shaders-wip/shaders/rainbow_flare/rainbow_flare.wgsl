// The uniform struct must match the Rust RainbowFlareParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would produce a 32-byte struct and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct RainbowFlareParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: RainbowFlareParams;

// ---------------------------------------------------------------------------
// Spectral hue → linear-RGB conversion
//
// Maps a hue in [0, 1) to an approximate visible-spectrum color using a smooth
// piecewise approximation of the red→orange→yellow→green→cyan→blue→violet
// progression.  The output components are in linear light, not sRGB.
// ---------------------------------------------------------------------------
fn spectrum_to_rgb(t: f32) -> vec3<f32> {
    // Wrap t into [0, 1).
    let h = t - floor(t);

    // Each primary occupies a 1/6 segment; smooth cubic blending between them.
    // Segment boundaries (in sixths of the full cycle):
    //   0.00 – 0.17  red     → orange/yellow
    //   0.17 – 0.33  yellow  → green
    //   0.33 – 0.50  green   → cyan
    //   0.50 – 0.67  cyan    → blue
    //   0.67 – 0.83  blue    → violet
    //   0.83 – 1.00  violet  → red
    let r = abs(h * 6.0 - 3.0) - 1.0;
    let g = 2.0 - abs(h * 6.0 - 2.0);
    let b = 2.0 - abs(h * 6.0 - 4.0);
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---------------------------------------------------------------------------
// Rainbow Flare procedural overlay
//
// Generates an iridescent prism/lens-flare overlay using polar coordinates
// centred on the image.  Two contributions are summed:
//
//   1. Radial rings — concentric spectral bands whose hue shifts with distance
//      from the centre.  The ring frequency is tuned so that several full
//      spectrum cycles are visible across the image diagonal, producing a
//      rainbow-soap-bubble appearance.
//
//   2. Angular sweep — hue varies with the polar angle, cycling once through
//      the full spectrum per revolution.  This is weighted by a soft radial
//      envelope that is strongest in the mid-field and fades at both the
//      centre and the far edges.
//
// The two contributions are averaged and blended additively with the original
// via params.strength.  At strength=0 the contribution is zero (identity).
// Intermediate results are not clamped so downstream shaders retain headroom.
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Normalised UV in [0, 1] with half-pixel offset for a stable centre.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Centred coordinates in [-1, 1] accounting for aspect ratio.
    let aspect = f32(dims.x) / f32(dims.y);
    let centered = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);

    // Polar decomposition.
    let radius = length(centered);
    let angle  = atan2(centered.y, centered.x); // [-π, +π]

    // ── Contribution 1: Radial spectral rings ────────────────────────────────
    //
    // Hue cycles with radius; 5 full spectrum cycles across the image half-
    // diagonal (~0.71 for a square).  sin² envelope smooths the rings.
    let ring_cycles  = 5.0;
    let ring_phase   = radius * ring_cycles;
    let ring_hue     = fract(ring_phase);
    let ring_wave    = sin(ring_phase * 6.2832);
    let ring_weight  = ring_wave * ring_wave;              // always ≥ 0
    let ring_rgb     = spectrum_to_rgb(ring_hue) * ring_weight;

    // ── Contribution 2: Angular spectral sweep ───────────────────────────────
    //
    // Hue cycles once per revolution (angle/2π maps [−π,+π] → [0,1]).
    // A soft bell envelope peaks at radius ≈ 0.3–0.5 and fades toward both
    // the centre (where angle is undefined) and the far edges.
    let angle_hue    = fract(angle / 6.2832);             // [0, 1)
    let bell_peak    = 0.40;                               // radius at peak
    let bell_width   = 0.30;
    let dr           = radius - bell_peak;
    let bell_weight  = exp(-dr * dr / (2.0 * bell_width * bell_width));
    let angle_rgb    = spectrum_to_rgb(angle_hue) * bell_weight;

    // ── Combine and blend ────────────────────────────────────────────────────
    //
    // Average the two contributions so neither dominates.  Scale by 0.6 to
    // keep the overlay from overpowering a fully lit image at strength=1.
    let flare = (ring_rgb + angle_rgb) * 0.5 * 0.6;

    // Additive blend scaled by strength.  At strength=0 the contribution is 0
    // (identity).  Not clamped — downstream shaders retain full dynamic range.
    let out_rgb = pixel.rgb + flare * params.strength;

    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
