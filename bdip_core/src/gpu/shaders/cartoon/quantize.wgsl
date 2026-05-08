// Cartoon — posterize (quantize) pass.
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

// Bindings — position-indexed (1 input → input at 0, output at 1).
@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CartoonParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord   = vec2<i32>(gid.xy);
    let smoothed = textureLoad(input_texture, coord, 0);

    // Quantization runs in linear-light space (consistent with the rest of the pipeline).
    // Bands fall at energy-uniform intervals, which differs visibly from sRGB-gamma
    // quantization (e.g., Photoshop Posterize). An sRGB-space Cartoon variant is tracked
    // in specs/tech_debt.md "Cartoon (sRGB-quantization variant)".
    let L = floor(clamp(params.levels, 2.0, 16.0));
    let quantized = clamp(floor(smoothed.rgb * L) / (L - 1.0), vec3<f32>(0.0), vec3<f32>(1.0));

    textureStore(output_texture, coord, vec4<f32>(quantized, smoothed.a));
}
