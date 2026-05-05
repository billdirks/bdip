// The uniform struct must match the Rust HolographicParams layout exactly:
// four f32 fields (16 bytes total, aligned for WebGPU uniform buffers).
struct HolographicParams {
    intensity:      f32,
    frequency:      f32,
    scale:          f32,
    blend_strength: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: HolographicParams;

// ---------------------------------------------------------------------------
// Spectral hue → linear-RGB conversion
//
// Maps a hue in [0, 1) to an approximate visible-spectrum colour using a
// smooth piecewise approximation of the red→orange→yellow→green→cyan→blue→
// violet progression.  Output components are in linear light, not sRGB.
// ---------------------------------------------------------------------------
fn spectrum_to_rgb(t: f32) -> vec3<f32> {
    let h = t - floor(t); // wrap into [0, 1)
    let r = abs(h * 6.0 - 3.0) - 1.0;
    let g = 2.0 - abs(h * 6.0 - 2.0);
    let b = 2.0 - abs(h * 6.0 - 4.0);
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---------------------------------------------------------------------------
// Holographic foil overlay
//
// Generates an iridescent holographic-foil appearance by layering three
// UV-based spectral contributions entirely from mathematics — no textures:
//
//   1. Horizontal bands — hue cycles along the U axis at `frequency` cycles
//      per image width, modulated by a sine-squared envelope.
//
//   2. Diagonal shimmer — hue cycles along the diagonal (U + V) direction,
//      introducing the angular colour shift characteristic of foil.
//
//   3. Interference fringes — a high-frequency sine pattern oriented at 45°
//      (U - V) creates the thin iridescent bands typical of diffraction
//      gratings.  The fringe frequency is 3× the base frequency.
//
// The three contributions are averaged, then blended with the original image
// using two complementary modes:
//
//   Screen blend  — adds luminosity without exceeding the sum (soft light).
//   Additive mix  — raw spectral colour added on top (harder, glowing foil).
//
// `blend_strength` controls the ratio between screen and additive.  Both modes
// are scaled by `intensity`, so intensity=0 is a strict identity.
//
// UV coordinates are divided by `scale` before colour generation, zooming the
// pattern in (scale > 1) or out (scale < 1).
//
// Intermediate results are not clamped so downstream shaders retain headroom.
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Normalised UV in [0, 1] with half-pixel offset for a stable pattern.
    let uv_raw = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Apply spatial scale: dividing by scale zooms the pattern in.
    // Centred on (0.5, 0.5) so zooming doesn't shift the pattern off-screen.
    let uv = (uv_raw - vec2<f32>(0.5)) / params.scale + vec2<f32>(0.5);

    let freq = params.frequency;

    // ── Contribution 1: Horizontal spectral bands ────────────────────────────
    //
    // Hue cycles along U at the requested frequency.  sin² envelope keeps the
    // colour non-negative and creates smooth light/dark alternation within each
    // spectral band.
    let h_phase  = uv.x * freq;
    let h_hue    = fract(h_phase);
    let h_wave   = sin(h_phase * 6.2832);
    let h_weight = h_wave * h_wave; // always ≥ 0
    let h_rgb    = spectrum_to_rgb(h_hue) * h_weight;

    // ── Contribution 2: Diagonal shimmer ────────────────────────────────────
    //
    // Hue varies along the (U + V) diagonal at half the base frequency,
    // producing the characteristic angular colour sweep of foil stickers.
    let d_phase  = (uv.x + uv.y) * freq * 0.5;
    let d_hue    = fract(d_phase);
    let d_wave   = sin(d_phase * 6.2832);
    let d_weight = d_wave * d_wave;
    let d_rgb    = spectrum_to_rgb(d_hue) * d_weight;

    // ── Contribution 3: Interference fringes ────────────────────────────────
    //
    // High-frequency (3× base) fringes oriented along (U - V) simulate
    // diffraction-grating iridescence.  The narrower spacing produces
    // fine spectral bands resembling physical holographic foil.
    let f_phase  = (uv.x - uv.y) * freq * 3.0;
    let f_hue    = fract(f_phase);
    let f_wave   = sin(f_phase * 6.2832);
    let f_weight = f_wave * f_wave;
    let f_rgb    = spectrum_to_rgb(f_hue) * f_weight;

    // Average the three contributions; scale by 0.55 so the combined foil does
    // not over-saturate a fully lit image at intensity=1, blend_strength=1.
    let foil = (h_rgb + d_rgb + f_rgb) * (1.0 / 3.0) * 0.55;

    // ── Screen blend ─────────────────────────────────────────────────────────
    //
    // Screen formula: 1 - (1-A)*(1-B).  Brightens without hard clipping,
    // preserving highlight detail better than pure additive at high strengths.
    let screen = vec3<f32>(1.0) - (vec3<f32>(1.0) - pixel.rgb) * (vec3<f32>(1.0) - foil);

    // ── Combine screen and additive layers ───────────────────────────────────
    //
    // blend_strength=0 → purely additive (raw foil added on top).
    // blend_strength=1 → purely screen   (soft luminous foil).
    // Intermediate values interpolate between both.
    let blended = mix(pixel.rgb + foil, screen, params.blend_strength);

    // Lerp between original and blended result by intensity.
    // At intensity=0 the output equals the input exactly (identity).
    let out_rgb = mix(pixel.rgb, blended, params.intensity);

    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
