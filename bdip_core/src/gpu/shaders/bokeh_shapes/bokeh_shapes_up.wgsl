// Bokeh Shapes — bilinear upsample pass.
//
// Upsamples the blurred downsampled texture back to full resolution using manual
// bilinear interpolation from the 4 nearest input texels. The scale factor is
// derived from texture dimensions so no hardcoded constant is needed.
//
// Declares the full BokehShapesParams struct to satisfy WebGPU's uniform
// binding-size validation.

struct BokehShapesParams {
    radius:   f32,
    sides:    f32,
    strength: f32,
    _padding: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: BokehShapesParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(output_texture);
    if gid.x >= out_dims.x || gid.y >= out_dims.y { return; }

    let in_dims = textureDimensions(input_texture);
    let scale   = vec2<f32>(in_dims) / vec2<f32>(out_dims);

    // Map output pixel centre to fractional input coordinate.
    let src  = (vec2<f32>(gid.xy) + 0.5) * scale - 0.5;
    let p0   = vec2<i32>(floor(src));
    let frac = src - vec2<f32>(p0);

    let max_coord = vec2<i32>(in_dims) - 1;
    let c00 = textureLoad(input_texture, clamp(p0,                    vec2<i32>(0), max_coord), 0);
    let c10 = textureLoad(input_texture, clamp(p0 + vec2<i32>(1, 0), vec2<i32>(0), max_coord), 0);
    let c01 = textureLoad(input_texture, clamp(p0 + vec2<i32>(0, 1), vec2<i32>(0), max_coord), 0);
    let c11 = textureLoad(input_texture, clamp(p0 + vec2<i32>(1, 1), vec2<i32>(0), max_coord), 0);

    let top    = mix(c00, c10, frac.x);
    let bottom = mix(c01, c11, frac.x);
    textureStore(output_texture, vec2<i32>(gid.xy), mix(top, bottom, frac.y));
}
