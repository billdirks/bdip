// Blueprint — combine pass.
//
// Produces the final blueprint look:
//   - The base layer is a deep blue background derived from the inverted source
//     luminance. Dark areas of the original become bright blue-white lines; bright
//     areas become deep blue — mimicking architectural blueprint paper.
//   - Edge pixels (from the edges scratch) are overlaid as bright white lines to
//     reinforce structural detail.
//   - The whole effect is blended against the original image via `strength` so that
//     strength=0.0 is a pixel-perfect identity pass.
//
// All Blueprint WGSL files declare the full BlueprintParams struct to satisfy
// WebGPU's uniform binding-size validation.

struct BlueprintParams {
    strength:        f32,
    edge_threshold:  f32,
    edge_thickness:  f32,
    _padding:        f32,
}

@group(0) @binding(0) var input_source:   texture_2d<f32>;
@group(0) @binding(1) var input_edges:    texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: BlueprintParams;

fn luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let c = vec2<i32>(gid.xy);

    let src   = textureLoad(input_source, c, 0);
    let edge  = textureLoad(input_edges,  c, 0).r;

    // Convert source luminance to a blueprint base layer.
    // Inverted luminance maps bright source areas → near-zero (deep blue) and
    // dark source areas → near-one (blue-white lines), emulating blueprint paper.
    let inv_luma = 1.0 - luma(src.rgb);

    // Blueprint blue: a rich cobalt/Prussian blue (linear light values).
    // The base color scales the inv_luma so structure is visible in the blue field.
    let blue_base = vec3<f32>(0.02, 0.10, 0.55) + inv_luma * vec3<f32>(0.18, 0.30, 0.45);

    // Overlay Sobel edges as bright near-white lines.
    let blueprint = mix(blue_base, vec3<f32>(0.85, 0.92, 1.0), edge);

    // Blend between the original pixel and the blueprint result.
    let out_rgb = mix(src.rgb, blueprint, params.strength);

    textureStore(output_texture, c, vec4<f32>(out_rgb, src.a));
}
