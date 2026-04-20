use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, SliderDef, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TemperatureParams {
    pub temp: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for TemperatureParams {
    const ID: &'static str = "temperature";
    const DISPLAY_NAME: &'static str = "Temperature";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Temperature",
        min: -1.0,
        max: 1.0,
        default: 0.0,
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "temperature",
        wgsl_source: include_str!("temperature.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            temp: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    TemperatureParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_temperature_registry_entry_exists() {
        assert!(registry_by_id("temperature").is_some());
    }

    #[test]
    fn test_temperature_registry_metadata() {
        let reg = registry_by_id("temperature").unwrap();
        assert_eq!(reg.meta.display_name, "Temperature");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Temperature",
                min: -1.0,
                max: 1.0,
                default: 0.0,
            }])
        );
    }

    #[test]
    fn test_temperature_make_uniform_known_value() {
        let reg = registry_by_id("temperature").unwrap();
        let bytes = (reg.make_uniform)(&[0.5]);
        let expected = bytemuck::bytes_of(&TemperatureParams {
            temp: 0.5,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_temperature_identity_zero() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 10000, 5000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "temperature",
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
    fn test_temperature_warm_increases_red_decreases_blue() {
        // Positive temp: R *= (1 + temp) > 1, B *= (1 - temp) < 1.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "temperature",
                values: vec![0.5],
            }],
        );
        for pixel in out.pixels() {
            assert!(pixel[0] > 30000, "expected red to increase");
            assert!(pixel[2] < 30000, "expected blue to decrease");
        }
    }

    #[test]
    fn test_temperature_cool_decreases_red_increases_blue() {
        // Negative temp: R *= (1 + temp) < 1, B *= (1 - temp) > 1.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "temperature",
                values: vec![-0.5],
            }],
        );
        for pixel in out.pixels() {
            assert!(pixel[0] < 30000, "expected red to decrease");
            assert!(pixel[2] > 30000, "expected blue to increase");
        }
    }

    #[test]
    fn test_temperature_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "temperature",
                values: vec![0.5],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_temperature_chained_with_brightness() {
        // Temperature identity (0.0) followed by brightness zero should equal brightness zero alone.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 20000, 20000);

        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "temperature",
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
