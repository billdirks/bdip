use crate::gpu::assets::{AuxAssetFormat, AuxAssetRegistration};
use crate::gpu::shaders::AuxTextureDimension;

inventory::submit!(AuxAssetRegistration {
    name: "identity_lut_64",
    raw_bytes: include_bytes!("identity_64.bin"),
    format: AuxAssetFormat::CubeRaw { size: 64 },
    dimension: AuxTextureDimension::D3,
});
