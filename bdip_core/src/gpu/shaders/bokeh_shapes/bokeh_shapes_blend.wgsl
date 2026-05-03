// Bokeh Shapes — blend pass.
//
// Blends the original source texture with the polygon-blurred scratch texture using
// `params.strength` as the mix weight:
//
//   out_rgb = mix(source.rgb, blurred.rgb, strength)
//
// At strength=0 the output is identical to the source (identity). At strength=1 the
// output is the fully blurred result. Alpha is always copied from the source.
//
// All BokehShapes WGSL files declare the full BokehShapesParams struct to satisfy
// WebGPU's uniform binding-size validation.

struct BokehShapesParams {
    radius:   f32,
    sides:    f32,
    strength: f32,
    _padding: f32,
}

// Two inputs (source at binding 0, blurred scratch at binding 1), output at binding 2.
@group(0) @binding(0) var input_source:  texture_2d<f32>;
@group(0) @binding(1) var input_blurred: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: BokehShapesParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord   = vec2<i32>(gid.xy);
    let src     = textureLoad(input_source,  coord, 0);
    let blurred = textureLoad(input_blurred, coord, 0);

    let out_rgb = mix(src.rgb, blurred.rgb, params.strength);
    textureStore(output_texture, coord, vec4<f32>(out_rgb, src.a));
}
