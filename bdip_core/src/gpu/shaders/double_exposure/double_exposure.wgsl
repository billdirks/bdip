// The uniform struct must match the Rust DoubleExposureParams layout exactly:
// one f32 followed by three f32 padding fields (16 bytes total).
// Using individual f32 fields rather than vec3<f32> avoids the 16-byte alignment
// that vec3 introduces in WGSL, which would produce a 32-byte struct and mismatch
// the 16-byte buffer the engine allocates from the Rust side.
struct DoubleExposureParams {
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: DoubleExposureParams;

// ---------------------------------------------------------------------------
// Double Exposure
//
// Simulates the classic film technique of exposing the same frame twice.
// A "ghost" second exposure is derived from the source image itself by:
//
//   1. Approximating a soft blur via a 3×3 box sample of neighbouring pixels.
//      This gives the ghost a slightly defocused, dreamlike quality that
//      distinguishes it from a sharp copy of the original layer.
//
//   2. Inverting each channel of the blurred sample (channel = 1.0 - channel).
//      Per-channel inversion is the standard photographic negative operation;
//      it maps dark areas to bright and bright areas to dark, producing the
//      characteristic halo-around-highlights look of real double-exposure film.
//      It is well-behaved at all luminance levels, including pure black (→ white)
//      and pure white (→ black).
//
//   3. Shifting the hue of the ghost by rotating the RGB channels — a fast
//      approximation of a 120° hue shift (R→G→B→R) that gives the ghost a
//      complementary color cast without a full HSL conversion.
//
// The ghost is composited onto the original using Screen blend mode:
//
//   screen(a, b) = 1 - (1 - a) * (1 - b)
//
// Screen brightens the image while preserving detail in both layers; it can
// never make the output darker than the original.
//
// At params.strength = 0.0 the scaled ghost is zero-valued, so the screen
// formula collapses to the identity:
//   1 - (1 - a) * (1 - 0) = a
// Intermediate results are not clamped so downstream shaders retain headroom
// above 1.0.
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    // ── Step 1: Approximate soft blur via 3×3 box sample ────────────────────
    //
    // Sample a 3×3 neighbourhood. Pixel coordinates are clamped to [0, dims-1]
    // so edge pixels are handled without artefacts (border pixels replicate
    // their own value for out-of-bounds neighbours).
    var blur_sum = vec3<f32>(0.0);
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let sample_coord = clamp(
                coord + vec2<i32>(dx, dy),
                vec2<i32>(0),
                vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1),
            );
            blur_sum += textureLoad(input_texture, sample_coord, 0).rgb;
        }
    }
    let blurred = blur_sum / 9.0;

    // ── Step 2: Per-channel inversion of the blurred ghost ───────────────────
    //
    // channel_inverted = 1.0 - channel
    // Dark input → bright ghost, bright input → dark ghost.
    // This is the photographic negative of the blurred layer.
    let inverted = 1.0 - blurred;

    // ── Step 3: Shift hue via RGB channel rotation ───────────────────────────
    //
    // Rotating channels (R→G→B→R) is a fast approximation of a 120° hue shift
    // that gives the ghost a complementary color cast, separating it visually
    // from the original exposure.
    let ghost = vec3<f32>(inverted.b, inverted.r, inverted.g);

    // ── Step 4: Screen blend mode ────────────────────────────────────────────
    //
    // screen(a, b) = 1 - (1 - a) * (1 - b)
    // When strength=0 the ghost is multiplied to zero, so the term
    //   1 - (1 - a) * (1 - 0) = a  → identity.
    let scaled_ghost = ghost * params.strength;
    let screened = 1.0 - (1.0 - pixel.rgb) * (1.0 - scaled_ghost);

    textureStore(output_texture, coord, vec4<f32>(screened, pixel.a));
}
