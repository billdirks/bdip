use crate::gpu::assets::{AuxAssetFormat, AuxAssetRegistration};
use crate::gpu::shaders::AuxTextureDimension;

inventory::submit!(AuxAssetRegistration {
    name: "twilight_cat",
    raw_bytes: include_bytes!("twilight_cat.png"),
    format: AuxAssetFormat::Png,
    dimension: AuxTextureDimension::D2,
});
