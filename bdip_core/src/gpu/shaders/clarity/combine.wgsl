// Clarity — combine pass.
//
// Implements: C_out = C_in + (C_in - C_blurred) * amount * W_mid
// where W_mid is a midtone luminance weight that peaks at 0.5 and falls to 0 at 0 and 1.
//
// All three Clarity WGSL files declare the full ClarityParams struct to satisfy
// WebGPU's uniform binding-size validation (see specs/multi-pass-plan.md
// § "Bind-group contract (multi-pass passes)").

struct ClarityParams {
    amount:   f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

// Bindings — position-indexed (2 inputs → inputs at 0 and 1, output at 2).
@group(0) @binding(0) var input_source:  texture_2d<f32>;
@group(0) @binding(1) var input_blurred: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ClarityParams;

// Midtone weight peaks at luma=0.5, falls smoothly to 0 at 0 and 1.
// Standard form: 1 - (2*luma - 1)^2.
fn midtone_weight(luma: f32) -> f32 {
    let t = 2.0 * luma - 1.0;
    return clamp(1.0 - t * t, 0.0, 1.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord   = vec2<i32>(gid.xy);
    let src     = textureLoad(input_source,  coord, 0);
    let blurred = textureLoad(input_blurred, coord, 0);

    // High-pass signal: the detail the blur removed.
    let c_hp  = src.rgb - blurred.rgb;
    let luma  = dot(src.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let w_mid = midtone_weight(luma);

    // Clamped to [0, 1]: the formula can legitimately exceed the range on
    // saturated pixels; Rgba16Float preserves any overshoot until readback, but
    // clamping here matches the reference formula's intent.
    let out_rgb = clamp(
        src.rgb + c_hp * params.amount * w_mid,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    textureStore(output_texture, coord, vec4<f32>(out_rgb, src.a));
}
