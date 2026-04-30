// Pop Art — combine pass.
//
// Overlays a procedural halftone dot pattern on the colorized image and blends
// the result with the original based on the strength slider. Inside each
// dot_scale-sized grid cell, a circular dot retains the bold color; the gaps
// between dots are lightened to simulate paper showing through a silkscreen print.
//
// Binding layout (2 inputs → inputs at 0–1, output at 2):
//   @binding(0) Source (original)
//   @binding(1) Scratch("colorize")
//   @binding(2) output

struct PopArtParams {
    strength:  f32,
    levels:    f32,
    dot_scale: f32,
    _padding:  f32,
}

@group(0) @binding(0) var input_source:   texture_2d<f32>;
@group(0) @binding(1) var input_colorize: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: PopArtParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord    = vec2<i32>(gid.xy);
    let src      = textureLoad(input_source,   coord, 0);
    let colorize = textureLoad(input_colorize, coord, 0);

    // Halftone dot: circular region centered in each dot_scale-sized grid cell.
    // Radius of 0.35 (out of 0.5 half-cell) keeps dots separated without touching.
    let cell_frac = fract(vec2<f32>(gid.xy) / params.dot_scale) - 0.5;
    let dist = length(cell_frac);
    let dot  = step(dist, 0.35);

    // Outside the dot, lighten toward the midpoint of the color to simulate paper
    // showing through the silkscreen ink gaps.
    let paper  = colorize.rgb * 0.5 + 0.5;
    let dotted = mix(paper, colorize.rgb, dot);

    // Blend between original and pop-art output based on strength.
    let out_rgb = mix(src.rgb, dotted, params.strength);

    textureStore(output_texture, coord, vec4<f32>(out_rgb, src.a));
}
