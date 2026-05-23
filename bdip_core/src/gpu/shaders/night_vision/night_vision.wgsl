// Night Vision shader
//
// Simulates night-vision goggle imagery through four layered operations:
//
//   1. Light amplification: the source RGB is multiplied by `amplification`,
//      boosting dark areas in a manner analogous to the photomultiplier tube
//      (PMT) in real NV equipment.  amplification = 1.0 is a no-op.
//
//   2. Green phosphor tint: blends the amplified colour pixel toward a
//      phosphor-green monochrome rendition using `green_tint` as the blend
//      weight.  The NV monochrome path converts to luminance and recolours it
//      with the P31 phosphor colour (0.20, 1.00, 0.20).
//      At tint = 0.0 the original (amplified) colour is preserved; at
//      tint = 1.0 the image is fully green monochrome.
//
//   3. CRT scanlines: attenuates every other row by a smooth cosine-derived
//      modulation scaled by `scanline_intensity`. The period matches typical
//      525-line CRT geometry at normal screen viewing distances.
//
//   4. High-frequency sensor noise: adds a spatially uncorrelated hash-derived
//      value to each pixel, scaled by `noise_amount`. A second independent hash
//      dimension breaks the diagonal artifact patterns that single-axis hashes
//      produce.
//
// Identity contract:
//   green_tint = 0.0, noise_amount = 0.0, scanline_intensity = 0.0,
//   amplification = 1.0 → output equals input pixel-for-pixel.

struct NightVisionParams {
    green_tint:         f32,
    noise_amount:       f32,
    scanline_intensity: f32,
    amplification:      f32,
}

@group(0) @binding(0) var src_texture:     texture_2d<f32>;
@group(0) @binding(1) var dst_texture:     texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: NightVisionParams;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Perceptual luminance weights for linear-light RGB (BT.709).
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Integer hash function mapping a 2-D coordinate to a pseudo-random float in
// [0, 1).  Two independent seeds (a, b) are mixed so that the hash is
// uncorrelated along both spatial axes, avoiding the stripe artefacts that
// simpler 1-D hashes produce on uniform inputs.
fn hash2(a: u32, b: u32) -> f32 {
    var h = a * 1664525u + b * 22695477u + 1013904223u;
    h ^= h >> 13u;
    h *= 0x9e3779b9u;
    h ^= h >> 15u;
    return f32(h & 0xFFFFu) / 65536.0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(src_texture, coord, 0);

    // ── 1. Light amplification ───────────────────────────────────────────────
    //
    // Scale the source RGB by the amplification factor.  This lifts dark
    // areas across all channels uniformly, simulating a photomultiplier tube.
    // Values above 1.0 are kept in the Rgba16Float headroom so a downstream
    // tone-mapper can handle them without clamping here.
    // amplification = 1.0 → amplified == src.rgb (no change).

    let amplified = src.rgb * params.amplification;

    // ── 2. Green phosphor tint ───────────────────────────────────────────────
    //
    // The phosphor colour of a P31 screen (the standard NV tube) is roughly
    // (0.20, 1.00, 0.20) in normalised RGB.  We blend from the amplified
    // colour pixel (green_tint = 0) to the phosphor-green monochrome rendition
    // (green_tint = 1) so the effect is continuously adjustable.
    //
    // green_tint = 0.0 → tinted == amplified (original colours preserved).
    // green_tint = 1.0 → tinted == luminance(src) * amplification * phosphor_green.

    let phosphor_green = vec3<f32>(0.20, 1.00, 0.20);
    let lum            = luminance(src.rgb);
    let nv_mono        = lum * params.amplification * phosphor_green;
    let tinted         = mix(amplified, nv_mono, params.green_tint);

    // ── 3. CRT scanlines ─────────────────────────────────────────────────────
    //
    // A horizontal cosine modulation at a two-pixel period approximates the
    // discrete raster lines of the phosphor screen.  The modulation is:
    //
    //   factor = 1 - scanline_intensity * 0.5 * (1 - cos(pi * y))
    //
    // At even rows (y mod 2 = 0) cos(0) = 1 → factor = 1 (no darkening).
    // At odd rows (y mod 2 = 1) cos(pi) = -1 → factor = 1 - scanline_intensity.
    //
    // scanline_intensity = 0.0 → factor = 1 everywhere (identity).

    let pi          = 3.14159265358979;
    let scanline_t  = (1.0 - cos(pi * f32(gid.y))) * 0.5; // 0 on even rows, 1 on odd rows
    let scan_factor = 1.0 - params.scanline_intensity * scanline_t;

    let scanned = tinted * scan_factor;

    // ── 4. High-frequency noise ──────────────────────────────────────────────
    //
    // NV sensors have a characteristic high-frequency shot noise pattern.
    // A hash-derived value centred on 0.0 is added to the luminance.  The
    // noise is added after colour and scanline processing so it appears as
    // photon-counting randomness on top of the phosphor screen.

    let noise_raw = hash2(gid.x, gid.y);         // [0, 1)
    let noise     = (noise_raw - 0.5) * params.noise_amount;  // centred, scaled

    let out_rgb = scanned + noise;

    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
