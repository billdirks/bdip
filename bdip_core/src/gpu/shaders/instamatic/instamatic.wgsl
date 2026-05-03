// Instamatic — single-pass color grading simulation of cheap instant cameras.
//
// Cheap consumer instant cameras (e.g. Kodak Instamatic, early Polaroid lines)
// had characteristic color rendering that differed from professional film stock:
//
//   - Slightly faded overall (compressed contrast, lifted blacks)
//   - Warm with a yellow-green cast in the midtones from uneven dye response
//   - Muted shadows that lift toward milky grey rather than pure black
//   - Subtle radial vignette from simple plastic lens systems
//
// All effects are implemented via mathematical color curves — no LUT or external
// file is required.
//
// At strength=0 the transform is a mathematical identity (no change to the
// image). At strength=1 the full Instamatic look is applied. Values between 0
// and 1 linearly blend between the original image and the styled output.
//
// The uniform struct must match the Rust InstamticParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte
// alignment vec3 would introduce in WGSL, which would make the struct 32 bytes
// and mismatch the 16-byte buffer the engine allocates from the Rust side.
struct InstamaticParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: InstamaticParams;

// Rec. 709 luminance coefficients (linear light).
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);
    let rgb   = pixel.rgb;

    // Normalised UV in [0, 1] with half-pixel offset for stable texel centres.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    let lum = luminance(rgb);

    // ── Shadow lift toward milky grey ────────────────────────────────────────
    //
    // Cheap instant film had a raised black floor: shadows settle to a milky,
    // slightly warm grey rather than pure black. The lifted target is a pale
    // warm grey approximating aged Instamatic shadow rendering.
    //
    // Shadow weight peaks at lum=0 and falls to 0 at lum=0.35.
    let lift_target    = vec3<f32>(0.055, 0.048, 0.038);
    let shadow_weight  = clamp(1.0 - lum / 0.35, 0.0, 1.0);
    let lifted         = rgb + params.strength * shadow_weight * lift_target;

    // ── Highlight compression (slight fading) ────────────────────────────────
    //
    // Instamatic prints looked slightly faded — highlights never reached film
    // white. Compress the upper tonal range by scaling toward a ceiling below
    // 1.0. At strength=1.0 the ceiling is ~0.94; at strength=0.0 the scale is
    // 1.0 (identity).
    //
    // Scale factor: 1.0 at strength=0 → 0.94 at strength=1.
    let highlight_scale = 1.0 - params.strength * 0.06;
    let faded           = lifted * highlight_scale;

    // ── Yellow-green midtone cast ────────────────────────────────────────────
    //
    // The yellow-green cast in the midtones comes from uneven dye response in
    // cheap instant film stock. Red and green are boosted slightly while blue is
    // reduced in the midtone range.
    //
    // Midtone weight: peaks at lum=0.45, falls to 0 at lum=0 and lum=0.9.
    // Using a smooth triangular weight keeps the cast concentrated in the mids.
    let midtone_weight = (1.0 - abs(lum - 0.45) / 0.45) * clamp(lum / 0.1, 0.0, 1.0);
    let midtone_cast   = vec3<f32>(0.04, 0.05, -0.06); // warm yellow-green shift
    let cast_rgb       = faded + params.strength * midtone_weight * midtone_cast;

    // ── Global warm channel balance ──────────────────────────────────────────
    //
    // The overall rendering of Instamatic film was warm, with slightly elevated
    // red and reduced blue across the full tonal range.
    let channel_scale = vec3<f32>(
        1.0 + params.strength * 0.04,   // red:   +4% at full strength
        1.0 + params.strength * 0.02,   // green: +2% at full strength
        1.0 - params.strength * 0.10,   // blue:  -10% at full strength
    );
    let balanced = cast_rgb * channel_scale;

    // ── Slight desaturation ──────────────────────────────────────────────────
    //
    // Cheap instant film rendered colors as slightly muted compared to
    // professional stock. Blend toward luminance to reduce saturation.
    let desaturated = mix(balanced, vec3<f32>(lum), params.strength * 0.08);

    // ── Radial vignette ──────────────────────────────────────────────────────
    //
    // Simple plastic lenses darken the frame corners. The vignette is a smooth
    // radial falloff computed from the UV distance to the image centre.
    //
    // The strength factor is reduced (×0.35) to keep the vignette subtle —
    // it should be visible but not overwhelm the color grading.
    let center_dist   = length(uv - vec2<f32>(0.5));
    // Smooth falloff: 0 at the centre, 1 at and beyond radius ~0.71 (corner).
    let vignette_mask = smoothstep(0.25, 0.75, center_dist);
    let vignette_amt  = params.strength * 0.35 * vignette_mask;
    let vignetted     = desaturated * (1.0 - vignette_amt);

    // ── Blend with original ──────────────────────────────────────────────────
    //
    // The strength parameter controls the blend between the original input and
    // the fully processed output. At strength=0 the output is the original
    // (mathematical identity). At strength=1 the full effect is applied.
    //
    // Note: the sub-effects above already incorporate strength individually for
    // correct relative weighting. This final blend provides the user-visible
    // blend-with-original behaviour described by the strength slider.
    let out_rgb = mix(rgb, vignetted, params.strength);

    // Do NOT clamp: preserve headroom above 1.0 for downstream shaders.
    textureStore(output_texture, coord, vec4<f32>(out_rgb, pixel.a));
}
