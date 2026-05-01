// Pass 2 of 2: Horizontal-slice glitch — displaces each scanline by a pseudo-random
// horizontal offset derived from the row's Y coordinate and a seed parameter.
//
// The pseudo-random function is a two-step hash (sin + fract) seeded by the Y row
// index combined with `seed`. This produces a different displacement for each row
// and each seed value without requiring an auxiliary noise texture.
//
// At strength=0 the maximum displacement is 0 pixels (identity). At strength=1 the
// maximum displacement is 20% of the image width. Rows near the middle of the image
// are more likely to receive large displacements than rows near the top or bottom
// because the random value is multiplied by a triangular envelope that peaks at y=0.5.
// This concentrates the "damage" toward the vertical centre while keeping the top and
// bottom edges anchored — matching the aesthetic of real display signal corruption.
//
// Pixels that would be sampled from outside the image boundary are clamped to the
// nearest edge pixel (border-clamp).

struct GlitchArtParams {
    strength: f32,
    seed:     f32,
    _pad0:    f32,
    _pad1:    f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: GlitchArtParams;

// A deterministic pseudo-random value in [0, 1) seeded by two floats.
// Uses the standard fract-sin hash; the large multiplier spreads nearby seeds
// across the full [0, 1) range.
fn rand(a: f32, b: f32) -> f32 {
    return fract(sin(a * 127.1 + b * 311.7) * 43758.5453);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Normalised Y position in [0, 1], used as the primary seed component so
    // every row in the same scanline band shares the same displacement.
    let y_norm = f32(coord.y) / f32(dims.y);

    // Triangular envelope peaks at y=0.5 and reaches 0 at y=0 and y=1.
    // This anchors the top/bottom and concentrates displacement mid-frame.
    let envelope = 1.0 - abs(2.0 * y_norm - 1.0);

    // Pseudo-random offset in [0, 1) for this row.
    let r = rand(y_norm, params.seed);

    // Signed displacement in [-0.5, 0.5], scaled by envelope and strength.
    // Maximum displacement is 20% of the image width at full strength.
    let max_disp     = 0.2 * f32(dims.x);
    let displacement = (r - 0.5) * 2.0 * envelope * params.strength * max_disp;

    // Apply horizontal displacement and clamp to valid column range.
    let src_x = clamp(
        i32(coord.x) + i32(displacement),
        0,
        i32(dims.x) - 1,
    );
    let src_coord = vec2<i32>(src_x, i32(coord.y));

    let pixel = textureLoad(src_texture, src_coord, 0);
    textureStore(dst_texture, coord, pixel);
}
