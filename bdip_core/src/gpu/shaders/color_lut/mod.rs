use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension, ParamKind, PassDef, PassInput,
    PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorLutParams {
    pub intensity: f32,
    pub _padding1: f32,
    pub _padding2: f32,
    pub _padding3: f32,
}

impl TransformShader for ColorLutParams {
    const ID: &'static str = "color_lut";
    const DISPLAY_NAME: &'static str = "Color LUT";
    const DESCRIPTION: &'static str =
        "Applies a 3D color look-up table (LUT) color grade to the image.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Intensity",
        min: 0.0,
        max: 1.0,
        default: 1.0,
        description: "How strongly the LUT color grade is applied; \
                      0 leaves the image unchanged, 1 applies it fully.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "color_lut",
        wgsl_source: include_str!("color_lut.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[AuxTextureDef {
            name: "identity_lut_64",
            dimension: AuxTextureDimension::D3,
            filter: AuxSamplerFilter::Linear,
        }],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            intensity: values[0],
            _padding1: 0.0,
            _padding2: 0.0,
            _padding3: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ColorLutParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_color_lut_registry_entry_exists() {
        assert!(registry_by_id("color_lut").is_some());
    }

    #[test]
    fn test_color_lut_registry_metadata() {
        let reg = registry_by_id("color_lut").unwrap();
        assert_eq!(reg.meta.display_name, "Color LUT");
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Intensity",
                min: 0.0,
                max: 1.0,
                default: 1.0,
                description: "How strongly the LUT color grade is applied; \
                              0 leaves the image unchanged, 1 applies it fully.",
            }])
        );
        assert_eq!(
            reg.meta.passes[0].aux_textures.len(),
            1,
            "must declare exactly 1 aux texture"
        );
    }

    #[test]
    fn test_color_lut_identity_lut_is_passthrough() {
        // The identity LUT at full intensity is a near-no-op: the only error is
        // from the sRGB↔linear round-trip (pow approximation) and f16 precision.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "color_lut",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: identity LUT must pass through mid-gray within ±64, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 32767).abs() <= 64,
                "G: identity LUT must pass through mid-gray within ±64, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 32767).abs() <= 64,
                "B: identity LUT must pass through mid-gray within ±64, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_color_lut_intensity_zero_is_identity() {
        // At intensity=0.0, mix(..., ..., 0.0) discards the LUT result entirely.
        // The output is the original linear image passed through ingest→present.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "color_lut",
                values: vec![0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 128,
                "R: intensity=0 must return original within ±128, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 32767).abs() <= 128,
                "G: intensity=0 must return original within ±128, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 32767).abs() <= 128,
                "B: intensity=0 must return original within ±128, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_color_lut_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "color_lut",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    #[test]
    fn test_color_lut_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let params = vec![1.0f32];
        let out_1 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "color_lut",
                values: params.clone(),
            }],
        );
        let out_2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "color_lut",
                values: params,
            }],
        );
        for (a, b) in out_1.pixels().zip(out_2.pixels()) {
            assert_eq!(a, b, "identical params must produce identical output");
        }
    }

    #[test]
    fn test_color_lut_aux_texture_declared() {
        let reg = registry_by_id("color_lut").unwrap();
        let aux = &reg.meta.passes[0].aux_textures;
        assert_eq!(aux.len(), 1, "must declare exactly one aux texture");
        assert_eq!(aux[0].dimension, AuxTextureDimension::D3, "aux must be D3");
        assert_eq!(
            aux[0].filter,
            AuxSamplerFilter::Linear,
            "aux must use Linear filter"
        );
    }

    #[test]
    fn test_color_lut_make_uniform_known_value() {
        let reg = registry_by_id("color_lut").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&ColorLutParams {
            intensity: 0.75,
            _padding1: 0.0,
            _padding2: 0.0,
            _padding3: 0.0,
        });
        assert_eq!(bytes, expected);
    }
}
