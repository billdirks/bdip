// Cartoon — box-filter downsample pass.
//
// Reduces the input to output dimensions using a box filter. The scale factor is
// derived from texture dimensions so the Rust-side PassScale::Down(N) controls
// allocation; this shader adapts automatically.
//
// Declares the full CartoonParams struct to satisfy WebGPU's uniform
// binding-size validation (see specs/multi-pass-plan.md § "Bind-group contract").

struct CartoonParams {
    strength:       f32,
    levels:         f32,
    edge_threshold: f32,
    edge_softness:  f32,
    edge_darkness:  f32,
    _padding0:      f32,
    _padding1:      f32,
    _padding2:      f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CartoonParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(output_texture);
    if gid.x >= out_dims.x || gid.y >= out_dims.y { return; }

    let in_dims = textureDimensions(input_texture);
    let scale   = vec2<f32>(in_dims) / vec2<f32>(out_dims);
    let base    = vec2<i32>(vec2<f32>(gid.xy) * scale);
    let block   = vec2<i32>(ceil(scale));

    var accum = vec4<f32>(0.0);
    var count = 0.0;
    for (var dy: i32 = 0; dy < block.y; dy = dy + 1) {
        for (var dx: i32 = 0; dx < block.x; dx = dx + 1) {
            let c = clamp(
                base + vec2<i32>(dx, dy),
                vec2<i32>(0),
                vec2<i32>(in_dims) - 1,
            );
            accum = accum + textureLoad(input_texture, c, 0);
            count = count + 1.0;
        }
    }

    textureStore(output_texture, vec2<i32>(gid.xy), accum / count);
}
