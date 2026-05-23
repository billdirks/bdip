// Pass 1 of 2: Bit-crush — quantise each colour channel to a reduced bit depth.
//
// The bit depth is derived from `strength`: at strength=0 there are 256 levels
// (8-bit, identity); at strength=1 there are 4 levels (2-bit, heavy crushing).
// The shader works in linear-light Rgba16Float space. The source texture stores
// values in [0, 1] (sRGB encoded as linear-light f16), so quantisation is applied
// directly to the normalised [0, 1] range.
//
// "Bit crushing" here means: reduce the number of discrete output levels by
// snapping each channel to the nearest multiple of (1 / levels).

struct GlitchArtParams {
    strength: f32,
    seed:     f32,
    _pad0:    f32,
    _pad1:    f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: GlitchArtParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    let pixel = textureLoad(src_texture, vec2<i32>(coord), 0);

    // Map strength ∈ [0, 1] to a level count.
    //   strength = 0.0 → 256 levels (≈ 8-bit, perceptually identity)
    //   strength = 1.0 →   4 levels (≈ 2-bit, heavy crush)
    // We interpolate logarithmically: levels = 2^(8 - 6*strength).
    let exponent  = 8.0 - 6.0 * params.strength;
    let levels    = pow(2.0, exponent);
    let step      = 1.0 / levels;

    // Snap each channel to the nearest grid point; preserve alpha.
    let crushed = floor(pixel.rgb / step + 0.5) * step;
    textureStore(dst_texture, coord, vec4<f32>(crushed, pixel.a));
}
