use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExposureParams {
    pub exposure: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for ExposureParams {
    const ID: &'static str = "exposure";
    const DISPLAY_NAME: &'static str = "Exposure";
    const DESCRIPTION: &'static str =
        "Adjusts exposure in stops by multiplying linear light values by a power-of-two factor.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Exposure",
        min: -4.0,
        max: 4.0,
        default: 0.0,
        description: "Exposure shift in stops. +1 doubles luminance; -1 halves it.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "exposure",
        wgsl_source: include_str!("exposure.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            exposure: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ExposureParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_exposure_registry_entry_exists() {
        assert!(registry_by_id("exposure").is_some());
    }

    #[test]
    fn test_exposure_registry_metadata() {
        let reg = registry_by_id("exposure").unwrap();
        assert_eq!(reg.meta.display_name, "Exposure");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Exposure",
                min: -4.0,
                max: 4.0,
                default: 0.0,
                description: "Exposure shift in stops. +1 doubles luminance; -1 halves it.",
            }])
        );
    }

    #[test]
    fn test_exposure_make_uniform_known_value() {
        let reg = registry_by_id("exposure").unwrap();
        let bytes = (reg.make_uniform)(&[1.0]);
        let expected = bytemuck::bytes_of(&ExposureParams {
            exposure: 1.0,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_exposure_identity_zero_stops() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 10000, 5000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "exposure",
                values: vec![0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!((pixel[0] as i32 - 20000).abs() <= 64);
            assert!((pixel[1] as i32 - 10000).abs() <= 64);
            assert!((pixel[2] as i32 - 5000).abs() <= 64);
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_exposure_brightens_at_positive_stops() {
        // +1 stop doubles light — output should be significantly brighter.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 10000, 10000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "exposure",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(pixel[0] > 10000, "expected pixel brighter than input");
        }
    }

    #[test]
    fn test_exposure_darkens_at_negative_stops() {
        // -1 stop halves light — output should be significantly darker.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "exposure",
                values: vec![-1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(pixel[0] < 30000, "expected pixel darker than input");
        }
    }

    #[test]
    fn test_exposure_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "exposure",
                values: vec![2.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_exposure_chained_with_brightness() {
        // Exposure identity (0.0) followed by brightness zero should equal brightness zero alone.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 20000, 20000);

        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "exposure",
                    values: vec![0.0],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );
        let brightness_only = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "brightness",
                values: vec![0.0],
            }],
        );

        for (a, b) in chained.pixels().zip(brightness_only.pixels()) {
            assert!((a[0] as i32 - b[0] as i32).abs() <= 64);
            assert!((a[1] as i32 - b[1] as i32).abs() <= 64);
            assert!((a[2] as i32 - b[2] as i32).abs() <= 64);
            assert_eq!(a[3], b[3]);
        }
    }
}
