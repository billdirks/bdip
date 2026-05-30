use crate::gpu::assets::{AuxAssetFormat, AuxAssetRegistration};
use crate::gpu::shaders::AuxTextureDimension;

// 128×128 greyscale character density atlas.
//
// The texture contains 16 ASCII characters (8×8 px each) arranged in a
// single row, ordered from least dense (space, index 0) to most dense
// (@, index 15). Each character's bitmask was hand-authored so that ink
// pixels are white (f16 ≈ 1.0) and background pixels are black (f16 = 0.0).
//
// The ASCII Art shader samples this texture to determine whether a given
// sub-pixel within a character cell should be rendered as ink or background.
inventory::submit!(AuxAssetRegistration {
    name: "ascii_char_map_16x16",
    raw_bytes: include_bytes!("ascii_char_map_16x16.png"),
    format: AuxAssetFormat::Png,
    dimension: AuxTextureDimension::D2,
});
