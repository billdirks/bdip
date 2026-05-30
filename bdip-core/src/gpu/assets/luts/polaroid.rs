use crate::gpu::assets::{AuxAssetFormat, AuxAssetRegistration};
use crate::gpu::shaders::AuxTextureDimension;

inventory::submit!(AuxAssetRegistration {
    name: "polaroid_lut_64",
    raw_bytes: include_bytes!("polaroid_64.bin"),
    format: AuxAssetFormat::CubeRaw { size: 64 },
    dimension: AuxTextureDimension::D3,
});
