use crate::gpu::assets::{AuxAssetFormat, AuxAssetRegistration};
use crate::gpu::shaders::AuxTextureDimension;

inventory::submit!(AuxAssetRegistration {
    name: "blue_noise_128",
    raw_bytes: include_bytes!("blue_noise_128.png"),
    format: AuxAssetFormat::Png,
    dimension: AuxTextureDimension::D2,
});
