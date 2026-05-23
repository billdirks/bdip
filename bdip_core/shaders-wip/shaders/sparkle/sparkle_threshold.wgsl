// Sparkle — threshold pass.
//
// Isolates pixels above a luminance threshold, outputting their contribution
// to a scratch texture for the subsequent blur passes. Pixels below the
// threshold are output as black (zero), so only bright regions carry the
// sparkle signal into the blur.
//
// All Sparkle WGSL files declare the full SparkleParams struct to satisfy
// WebGPU's uniform binding-size validation (see specs/adding_a_shader.md
// § "Shared-uniform alignment rule").

struct SparkleParams {
    threshold: f32,  // luminance cutoff ∈ [0.0, 1.0]
    intensity: f32,  // glow blend strength ∈ [0.0, 1.0]
    radius:    f32,  // spread size ∈ [0.0, 1.0] (fraction of image short axis)
    _padding:  f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: SparkleParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // Rec.709 luminance.
    let luma = dot(pixel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Amount by which this pixel exceeds the threshold. At threshold=1.0 no
    // pixel ever qualifies (identity). At threshold=0.0 all pixels pass through
    // at their original brightness.
    let excess = max(luma - params.threshold, 0.0);
    let scale  = select(0.0, excess / max(luma, 0.0001), luma > params.threshold);

    textureStore(output_texture, coord, vec4<f32>(pixel.rgb * scale, pixel.a));
}
