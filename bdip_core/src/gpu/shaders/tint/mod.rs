use crate::gpu::shaders::{ParamKind, ShaderMeta, SliderDef, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TintParams {
    pub tint: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for TintParams {
    const META: ShaderMeta = ShaderMeta {
        id: "tint",
        display_name: "Tint",
        wgsl_source: include_str!("tint.wgsl"),
        param: ParamKind::Sliders(&[SliderDef {
            name: "Tint",
            min: -1.0,
            max: 1.0,
            default: 0.0,
        }]),
    };

    fn from_values(values: &[f32]) -> Self {
        Self {
            tint: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<TintParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_tint_registry_entry_exists() {
        assert!(registry_by_id("tint").is_some());
    }

    #[test]
    fn test_tint_registry_metadata() {
        let reg = registry_by_id("tint").unwrap();
        assert_eq!(reg.meta.display_name, "Tint");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Tint",
                min: -1.0,
                max: 1.0,
                default: 0.0,
            }])
        );
    }

    #[test]
    fn test_tint_make_uniform_known_value() {
        let reg = registry_by_id("tint").unwrap();
        let bytes = (reg.make_uniform)(&[0.5]);
        let expected = bytemuck::bytes_of(&TintParams {
            tint: 0.5,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_tint_identity_zero() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 10000, 5000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tint",
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
    fn test_tint_positive_shifts_toward_magenta() {
        // Positive tint adds to Q in YIQ space. For a neutral gray, this shifts toward magenta,
        // reducing green and shifting red and blue together.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 32768, 32768, 32768);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tint",
                values: vec![0.5],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                pixel[1] < 32768,
                "expected green to decrease with positive tint"
            );
        }
    }

    #[test]
    fn test_tint_negative_shifts_toward_green() {
        // Negative tint subtracts from Q in YIQ space. For a neutral gray, green increases.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 32768, 32768, 32768);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tint",
                values: vec![-0.5],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                pixel[1] > 32768,
                "expected green to increase with negative tint"
            );
        }
    }

    #[test]
    fn test_tint_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tint",
                values: vec![0.5],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_tint_chained_with_brightness() {
        // Tint identity (0.0) followed by brightness zero should equal brightness zero alone.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 20000, 20000);

        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "tint",
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
