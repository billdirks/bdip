// Cartoon — edge detection pass (Sobel on Source).
//
// Sobel magnitude is computed on Rec.709 luma of the original Source image, not
// the bilateral-filtered one. The bilateral pass intentionally erases intra-region
// variation; computing Sobel on Source preserves the faithful edge structure of the
// photograph.
//
// Output: single-channel edge mask stored in .r; .gba = (0, 0, 1).
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
@group(0) @binding(0) var input_source:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CartoonParams;

fn luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn sample_luma(coord: vec2<i32>, dims: vec2<u32>) -> f32 {
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(dims) - 1);
    return luma(textureLoad(input_source, clamped, 0).rgb);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let c = vec2<i32>(gid.xy);

    // 3×3 Sobel kernel — sample the 3×3 neighbourhood.
    let tl = sample_luma(c + vec2<i32>(-1, -1), dims);
    let tc = sample_luma(c + vec2<i32>( 0, -1), dims);
    let tr = sample_luma(c + vec2<i32>( 1, -1), dims);
    let ml = sample_luma(c + vec2<i32>(-1,  0), dims);
    let mr = sample_luma(c + vec2<i32>( 1,  0), dims);
    let bl = sample_luma(c + vec2<i32>(-1,  1), dims);
    let bc = sample_luma(c + vec2<i32>( 0,  1), dims);
    let br = sample_luma(c + vec2<i32>( 1,  1), dims);

    let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    let mag = length(vec2<f32>(gx, gy));

    // Smooth the edge mask between edge_threshold and edge_threshold + edge_softness.
    let ramp_end = clamp(params.edge_threshold + params.edge_softness, 0.0, 2.83);
    let edge = smoothstep(params.edge_threshold, ramp_end, mag);

    textureStore(output_texture, c, vec4<f32>(edge, 0.0, 0.0, 1.0));
}
