// Fresco shader
//
// Simulates a Renaissance-style fresco painting by applying three transformations:
//
//   1. Matte desaturation: pulls pixel colors toward their luminance value,
//      reducing vivid saturation toward an earthy, matte palette typical of
//      pigments mixed into wet plaster (e.g., ochre, umber, sienna tones).
//
//   2. Contrast softening: slightly lifts shadows and compresses highlights,
//      reproducing the flat, washed-out look of pigments absorbed into plaster
//      as the medium dries.
//
//   3. Plaster grain overlay: samples the paper_grain_256 texture and multiplies
//      it against the processed color.  This adds the rough, porous micro-texture
//      of a plastered wall.  The grain is tiled via a UV scale parameter so the
//      user can adjust the apparent grain size relative to the image.
//
// All three steps are combined into a single result, then blended back onto the
// original source image using `params.strength`.  At strength = 0.0 the shader
// is a pure passthrough (identity).

struct FrescoParams {
    strength:      f32,
    matte:         f32,
    texture_scale: f32,
    _padding:      f32,
}

@group(0) @binding(0) var src_texture:    texture_2d<f32>;
@group(0) @binding(1) var dst_texture:    texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: FrescoParams;
@group(2) @binding(0) var grain_tex:      texture_2d<f32>;
@group(2) @binding(1) var grain_sampler:  sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<u32>(gid.xy);
    let src   = textureLoad(src_texture, coord, 0);

    // ── 1. Matte desaturation ────────────────────────────────────────────────
    //
    // Compute luminance (linear-light BT.709 coefficients) and lerp toward it.
    // params.matte = 0.0 leaves the original color; 1.0 produces a grayscale
    // image at the original luminance level.
    let luma = dot(src.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let matte_rgb = mix(src.rgb, vec3<f32>(luma), params.matte);

    // ── 2. Contrast softening (shadow lift + highlight compression) ──────────
    //
    // Fresco pigments spread into porous plaster and lose contrast as the
    // medium absorbs moisture.  A gentle S-curve inversion reproduces this:
    // shadows are lifted slightly above black, highlights pulled down from peak
    // white.  The formula maps [0, 1] → [0.04, 0.96] at full strength.
    //
    // The soft_amount below is derived from params.matte so that a fully matte
    // (desaturated) look also carries the expected contrast compression.  Using
    // a fixed fraction (0.5) of the matte parameter gives a moderate effect
    // without requiring a separate control.
    let soft_amount = params.matte * 0.5;
    let shadow_lift = soft_amount * 0.04;
    let highlight_compress = 1.0 - soft_amount * 0.08;
    let soft_rgb = matte_rgb * highlight_compress + shadow_lift;

    // ── 3. Plaster grain overlay ─────────────────────────────────────────────
    //
    // Tile the 256×256 paper grain texture across the image using the UV scale
    // parameter.  The grain values are in [0, 1]; multiplying by them darkens
    // the image where the plaster is recessed or porous.
    let grain_dims = vec2<f32>(textureDimensions(grain_tex));
    let uv = fract(vec2<f32>(gid.xy) / (grain_dims * params.texture_scale));
    let grain = textureSampleLevel(grain_tex, grain_sampler, uv, 0.0).rgb;

    // Soft blend of grain: full multiplication is very dark; lerp toward 1.0
    // at a fixed 0.25 weight so grain reads as subtle texture rather than a
    // heavy overlay.
    let grain_weight = 0.25;
    let textured_rgb = soft_rgb * mix(vec3<f32>(1.0), grain, grain_weight);

    // ── 4. Blend back onto source ─────────────────────────────────────────────
    //
    // When strength = 0.0 this returns the original source unchanged (identity).
    let out_rgb = mix(src.rgb, textured_rgb, params.strength);
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
