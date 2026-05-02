// Tilt-Shift — horizontal separable Gaussian blur pass.
//
// Operates on the 4× downsampled image from the downsample pass. The sigma is
// computed from the downsampled height so the kernel radius stays proportional
// to the original image after the upsample pass. Because SIGMA_FRACTION acts on
// the downsampled dimension, the apparent blur radius at full resolution is
// blur_strength * SIGMA_FRACTION * full_height — the same visual scale as if the
// blur were performed at full resolution, at 1/16 the pixel count.
//
// All five Tilt-Shift WGSL files declare the full TiltShiftParams struct to satisfy
// WebGPU's uniform binding-size validation (the engine validates the uniform
// binding size against the pipeline layout at creation time).
//
// RADIUS_CAP is paired with the data-dependent radius to bound loop iterations
// (see specs/adding_a_shader.md § "Data-dependent loop bounds (RADIUS_CAP)").

struct TiltShiftParams {
    focus_center:  f32,
    focus_width:   f32,
    blur_strength: f32,
    _padding:      f32,
}

// Bindings — position-indexed (1 input → input at 0, output at 1).
@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: TiltShiftParams;

// Sigma as a fraction of the (downsampled) image height.
const SIGMA_FRACTION: f32 = 0.05;
// Upper bound on kernel radius at 4× downsampled resolution (~1/4 of full-res cap).
const RADIUS_CAP: i32 = 40;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let sigma = params.blur_strength * SIGMA_FRACTION * f32(dims.y);
    let radius = min(i32(ceil(3.0 * sigma)), RADIUS_CAP);

    // Guard against sigma=0 to avoid NaN in Gaussian weight calculations.
    if radius == 0 {
        let centre = textureLoad(input_texture, vec2<i32>(gid.xy), 0);
        textureStore(output_texture, vec2<i32>(gid.xy), centre);
        return;
    }

    let two_sigma_sq = 2.0 * sigma * sigma;
    var accum:      vec4<f32> = vec4<f32>(0.0);
    var weight_sum: f32       = 0.0;
    let coord = vec2<i32>(gid.xy);

    for (var t: i32 = -radius; t <= radius; t = t + 1) {
        let offset = vec2<i32>(t, 0); // horizontal tap
        let s = textureLoad(
            input_texture,
            clamp(coord + offset, vec2<i32>(0), vec2<i32>(dims) - 1),
            0,
        );
        let w = exp(-f32(t * t) / two_sigma_sq);
        accum      = accum + s * w;
        weight_sum = weight_sum + w;
    }

    let out = accum / weight_sum;
    // Alpha is copied from the centre pixel — the blur must not smear alpha.
    let src_alpha = textureLoad(input_texture, coord, 0).a;
    textureStore(output_texture, coord, vec4<f32>(out.rgb, src_alpha));
}
