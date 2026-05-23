use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension, ParamKind, PassDef, PassInput,
    PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ThermalParams {
    pub intensity: f32,
    pub _padding1: f32,
    pub _padding2: f32,
    pub _padding3: f32,
}

impl TransformShader for ThermalParams {
    const ID: &'static str = "thermal";
    const DISPLAY_NAME: &'static str = "Thermal Heat Map";
    const DESCRIPTION: &'static str = "Remaps luminance through a thermal heat-map color gradient, simulating an infrared camera look.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Intensity",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend factor between the original image and the thermal gradient output; \
                      0 is unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "thermal",
        wgsl_source: include_str!("thermal.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[AuxTextureDef {
            name: "thermal_gradient",
            dimension: AuxTextureDimension::D2,
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

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ThermalParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_thermal_registry_entry_exists() {
        assert!(registry_by_id("thermal").is_some());
    }

    #[test]
    fn test_thermal_registry_metadata() {
        let reg = registry_by_id("thermal").unwrap();
        assert_eq!(reg.meta.display_name, "Thermal Heat Map");
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Intensity",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend factor between the original image and the thermal gradient output; \
                              0 is unchanged.",
            }])
        );
        assert_eq!(
            reg.meta.passes[0].aux_textures.len(),
            1,
            "must declare exactly 1 aux texture"
        );
    }

    #[test]
    fn test_thermal_intensity_zero_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "thermal",
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
                (pixel[1] as i32 - 16384).abs() <= 128,
                "G: intensity=0 must return original within ±128, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 8192).abs() <= 128,
                "B: intensity=0 must return original within ±128, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_thermal_full_intensity_remaps_luminance() {
        // White and black pixels must produce visibly different thermal colors
        // (not just grayscale). A white pixel (high luma) maps to the warm end
        // of the gradient; a black pixel (low luma) maps to the cool/dark end.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let white_img = make_solid_image(4, 4, 65535, 65535, 65535);
        let black_img = make_solid_image(4, 4, 0, 0, 0);

        let out_white = roundtrip(
            &mut renderer,
            &engine,
            &white_img,
            &[Transform {
                shader_id: "thermal",
                values: vec![1.0],
            }],
        );
        let out_black = roundtrip(
            &mut renderer,
            &engine,
            &black_img,
            &[Transform {
                shader_id: "thermal",
                values: vec![1.0],
            }],
        );

        // White input → high-luma end of gradient (near white: R high, G high, B high)
        // Black input → low-luma end of gradient (near black: R low, G low, B low)
        // At minimum, the R channel of white output must exceed that of black output.
        let white_r = out_white.pixels().next().unwrap()[0];
        let black_r = out_black.pixels().next().unwrap()[0];
        assert!(
            white_r > black_r,
            "white input must produce brighter thermal output than black input: \
             white_r={white_r}, black_r={black_r}"
        );
    }

    #[test]
    fn test_thermal_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "thermal",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    #[test]
    fn test_thermal_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let params = vec![0.8f32];
        let out_1 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "thermal",
                values: params.clone(),
            }],
        );
        let out_2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "thermal",
                values: params,
            }],
        );
        for (a, b) in out_1.pixels().zip(out_2.pixels()) {
            assert_eq!(a, b, "identical params must produce identical output");
        }
    }

    #[test]
    fn test_thermal_make_uniform_known_value() {
        let reg = registry_by_id("thermal").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&ThermalParams {
            intensity: 0.75,
            _padding1: 0.0,
            _padding2: 0.0,
            _padding3: 0.0,
        });
        assert_eq!(bytes, expected);
    }
}
