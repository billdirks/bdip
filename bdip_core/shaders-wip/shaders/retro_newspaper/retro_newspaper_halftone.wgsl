// Retro Newspaper — Pass 3: Halftone dot overlay + blend with source.
//
// Applies a rotated halftone dot grid over the quantised grayscale image.
// Each cell in the grid is a square whose size scales with `dot_frequency`.
// Within each cell, a dot is rendered whose radius is proportional to the
// local quantised tone: bright areas → small dot (lots of white), dark areas
// → large dot (lots of ink).
//
// The grid is rotated 45° — the classic newspaper halftone screen angle that
// minimises moiré artifacts in single-channel prints.
//
// The halftone result is then blended with the original source image using
// `strength` as the blend factor:
//
//   output = mix(src, halftone, strength)
//
// At strength=0.0 the source passes through unchanged (identity). At
// strength=1.0 the full retro-newspaper effect is applied.
//
// The output is written to the final texture.

struct RetroNewspaperParams {
    dot_frequency: f32,
    strength:      f32,
    _padding:      vec2<f32>,
}

// Bindings: input 0 = source (original colour), input 1 = quantised grayscale.
// Output is the final composited result.
@group(0) @binding(0) var src_texture:        texture_2d<f32>;
@group(0) @binding(1) var quantised_texture:  texture_2d<f32>;
@group(0) @binding(2) var dst_texture:        texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params:    RetroNewspaperParams;

const PI: f32 = 3.14159265358979;

// Rotate a 2D point by 45° clockwise.
fn rotate45(p: vec2<f32>) -> vec2<f32> {
    let c = cos(PI * 0.25); // 0.7071…
    let s = sin(PI * 0.25); // 0.7071…
    return vec2<f32>(c * p.x + s * p.y, -s * p.x + c * p.y);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(src_texture, coord, 0);
    let quant = textureLoad(quantised_texture, coord, 0);

    // UV coordinates in [0, 1].
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    // Scale UV by dot_frequency to create the grid, then rotate 45°.
    // dot_frequency controls how many dot cells appear across the shorter axis.
    let short_axis = f32(min(dims.x, dims.y));
    let cell_uv    = uv * params.dot_frequency;
    let rot_uv     = rotate45(cell_uv);

    // Fractional position within the current cell: [-0.5, 0.5].
    let cell_frac = fract(rot_uv) - 0.5;

    // Distance from cell centre, normalised so the cell half-width = 0.5.
    let dist = length(cell_frac);

    // Dot radius scales with the quantised tone: dark tone → large dot (more ink).
    // Radius 0.0 → no dot (pure white); radius 0.5 → dot fills entire cell (pure black).
    // The tone range [0, 1] maps to dot radius [0.45, 0.0]:
    //   luma=0 (black)  → radius=0.45 (large dot, mostly ink)
    //   luma=1 (white)  → radius=0.0  (no dot, all paper)
    let dot_radius = 0.45 * (1.0 - quant.r);

    // Paper colour: off-white, matching yellowed newsprint.
    let paper = vec3<f32>(0.94, 0.91, 0.82);
    // Ink colour: near-black with a slight warm cast.
    let ink   = vec3<f32>(0.06, 0.05, 0.04);

    // Dot test: inside the circle → ink; outside → paper.
    let dot_value = select(paper, ink, dist < dot_radius);

    // Blend with the original source image.
    // At strength=0.0 output equals src (identity).
    // At strength=1.0 output is the full halftone effect.
    let out_rgb = mix(src.rgb, dot_value, params.strength);
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
