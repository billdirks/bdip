// Cartoon — combine pass (3-input).
//
// Mixes Source and quantized image by `strength`, then darkens by the edge mask.
//
// Binding layout (position-indexed, 3 inputs → inputs at 0–2, output at 3):
//   @binding(0) Source
//   @binding(1) Scratch("quant")  — posterized image (full-res)
//   @binding(2) Scratch("edges")
//   @binding(3) output
//
// All four Cartoon WGSL files declare the full CartoonParams struct to satisfy
// WebGPU's uniform binding-size validation (see specs/multi-pass-plan.md
// § "Bind-group contract (multi-pass passes)").

struct CartoonParams {
    strength:       f32,
    levels:         f32,
    smoothing:      f32,
    edge_threshold: f32,
    edge_softness:  f32,
    edge_darkness:  f32,
    _padding0:      f32,
    _padding1:      f32,
}

@group(0) @binding(0) var input_source:   texture_2d<f32>;
@group(0) @binding(1) var input_quant:    texture_2d<f32>;
@group(0) @binding(2) var input_edges:    texture_2d<f32>;
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CartoonParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(input_source, coord, 0);
    let quant = textureLoad(input_quant,  coord, 0);
    let edges = textureLoad(input_edges,  coord, 0);

    // Blend original with posterized image, then darken by the edge mask.
    let color_base = mix(src.rgb, quant.rgb, params.strength);
    let darken     = 1.0 - params.edge_darkness * edges.r;
    let out_rgb    = clamp(color_base * darken, vec3<f32>(0.0), vec3<f32>(1.0));

    textureStore(output_texture, coord, vec4<f32>(out_rgb, src.a));
}
