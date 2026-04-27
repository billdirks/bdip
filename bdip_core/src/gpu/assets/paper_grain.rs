use crate::gpu::assets::{AuxAssetFormat, AuxAssetRegistration};
use crate::gpu::shaders::AuxTextureDimension;

inventory::submit!(AuxAssetRegistration {
    name: "paper_grain_256",
    raw_bytes: include_bytes!("paper_grain_256.png"),
    format: AuxAssetFormat::Png,
    dimension: AuxTextureDimension::D2,
});
