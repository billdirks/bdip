// Comic Book — combine pass (3-input).
//
// Composites the source, halftone shading, and bold ink edges into the final
// comic book look. The halftone pass provides dot shading and the edges pass
// provides the ink outlines.
//
// Binding layout (3 inputs → inputs at 0–2, output at 3):
//   @binding(0) Source (original)
//   @binding(1) Scratch("halftone")
//   @binding(2) Scratch("edges")
//   @binding(3) output

struct ComicBookParams {
    strength:        f32,
    dot_scale:       f32,
    edge_threshold:  f32,
    edge_thickness:  f32,
}

@group(0) @binding(0) var input_source:   texture_2d<f32>;
@group(0) @binding(1) var input_halftone: texture_2d<f32>;
@group(0) @binding(2) var input_edges:    texture_2d<f32>;
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ComicBookParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src      = textureLoad(input_source,   coord, 0);
    let halftone = textureLoad(input_halftone, coord, 0);
    let edges    = textureLoad(input_edges,    coord, 0);

    // Colorize the halftone: modulate the source hue by the halftone mask.
    // Where the halftone is 1.0 (paper), keep source color; where 0.0 (ink), darken.
    let colorized = src.rgb * halftone.r;

    // Overlay ink outlines: edges.r = 1 means ink edge, darken to near-black.
    let with_edges = colorized * (1.0 - edges.r);

    // Blend between original and comic book output based on strength.
    let out_rgb = mix(src.rgb, with_edges, params.strength);

    textureStore(output_texture, coord, vec4<f32>(out_rgb, src.a));
}
