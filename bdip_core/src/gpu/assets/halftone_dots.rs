use crate::gpu::assets::{AuxAssetFormat, AuxAssetRegistration};
use crate::gpu::shaders::AuxTextureDimension;

inventory::submit!(AuxAssetRegistration {
    name: "halftone_dots",
    raw_bytes: include_bytes!("halftone_dots.png"),
    format: AuxAssetFormat::Png,
    dimension: AuxTextureDimension::D2,
});
