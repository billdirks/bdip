use crate::gpu::assets::{AuxAssetFormat, AuxAssetRegistration};
use crate::gpu::shaders::AuxTextureDimension;

inventory::submit!(AuxAssetRegistration {
    name: "thermal_gradient",
    raw_bytes: include_bytes!("thermal_gradient_256x1.png"),
    format: AuxAssetFormat::Png,
    dimension: AuxTextureDimension::D2,
});
