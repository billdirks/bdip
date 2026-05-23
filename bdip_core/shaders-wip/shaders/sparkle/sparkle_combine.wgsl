// Sparkle — combine pass.
//
// Blends the Gaussian-spread glow layer back onto the original image.
// Formula: out = clamp(src + glow * intensity, 0, ∞)
//
// The upper bound is intentionally left unclamped to preserve >1.0 headroom
// for downstream shaders. Only the lower bound (zero) is clamped because the
// glow is strictly additive and cannot go negative.
//
// At intensity=0.0 the formula reduces to `src + 0 = src`, which is the
// identity. The glow contribution scales linearly with intensity.
//
// All Sparkle WGSL files declare the full SparkleParams struct to satisfy
// WebGPU's uniform binding-size validation (see specs/adding_a_shader.md
// § "Shared-uniform alignment rule").

struct SparkleParams {
    threshold: f32,
    intensity: f32,
    radius:    f32,
    _padding:  f32,
}

// Bindings — position-indexed (2 inputs → inputs at 0 and 1, output at 2).
@group(0) @binding(0) var input_source: texture_2d<f32>;
@group(0) @binding(1) var input_glow:   texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: SparkleParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(input_source, coord, 0);
    let glow  = textureLoad(input_glow,   coord, 0);

    // Additive blend: glow is added on top of the original.
    // Lower-bound clamp prevents negative values from floating-point noise.
    // Upper bound is left open to preserve linear-light headroom.
    let out_rgb = max(src.rgb + glow.rgb * params.intensity, vec3<f32>(0.0));

    // Alpha is always taken from the source; the glow layer is RGB-only.
    textureStore(output_texture, coord, vec4<f32>(out_rgb, src.a));
}
