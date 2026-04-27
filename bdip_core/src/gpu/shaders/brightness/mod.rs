use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrightnessParams {
    pub value: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for BrightnessParams {
    const ID: &'static str = "brightness";
    const DISPLAY_NAME: &'static str = "Brightness";
    const DESCRIPTION: &'static str =
        "Shifts image brightness on a linear scale in the working color space.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Amount",
        min: -1.0,
        max: 1.0,
        default: 0.0,
        description: "Amount to brighten or darken. Negative values darken; positive values brighten.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "brightness",
        wgsl_source: include_str!("brightness.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            value: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    BrightnessParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_brightness_registry_entry_exists() {
        assert!(registry_by_id("brightness").is_some());
    }

    #[test]
    fn test_brightness_registry_metadata() {
        let reg = registry_by_id("brightness").unwrap();
        assert_eq!(reg.meta.display_name, "Brightness");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Amount",
                min: -1.0,
                max: 1.0,
                default: 0.0,
                description: "Amount to brighten or darken. Negative values darken; positive values brighten.",
            }])
        );
    }

    #[test]
    fn test_brightness_make_uniform_known_value() {
        let reg = registry_by_id("brightness").unwrap();
        let bytes = (reg.make_uniform)(&[0.5]);
        let expected = bytemuck::bytes_of(&BrightnessParams {
            value: 0.5,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_transform_display_slider() {
        let t = Transform {
            shader_id: "brightness",
            values: vec![0.35],
        };
        assert_eq!(t.to_string(), "Brightness: 0.35");
    }

    #[test]
    fn test_brightness_shader_positive() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% gray in sRGB (32767/65535 ≈ 0.500 sRGB → ~0.214 linear)
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "brightness",
                values: vec![0.5],
            }],
        );

        // 0.214 + 0.5 = 0.714 linear → sRGB ≈ 0.862 → u16 ≈ 56500
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 56500).abs() <= 64,
                "R: expected ~56500, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 56500).abs() <= 64,
                "G: expected ~56500, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 56500).abs() <= 64,
                "B: expected ~56500, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_brightness_shader_negative() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% gray in sRGB → ~0.214 linear; 0.214 - 0.6 = -0.386 → clamped to 0
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "brightness",
                values: vec![-0.6],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[0], 0);
            assert_eq!(pixel[1], 0);
            assert_eq!(pixel[2], 0);
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_brightness_shader_zero() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 10794, 25700, 51400);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "brightness",
                values: vec![0.0],
            }],
        );

        // sRGB → linear → sRGB is a mathematical identity; differences are f16 rounding.
        for pixel in out_img.pixels() {
            assert!((pixel[0] as i32 - 10794).abs() <= 64);
            assert!((pixel[1] as i32 - 25700).abs() <= 64);
            assert!((pixel[2] as i32 - 51400).abs() <= 64);
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_brightness_headroom_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32767/65535 ≈ 0.500 sRGB → ~0.214 linear
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "brightness",
                    values: vec![0.8],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![-0.8],
                },
            ],
        );

        // 0.214 + 0.8 = 1.014 (above 1.0, held in f16 headroom); 1.014 - 0.8 = 0.214 linear.
        // linear_to_srgb(0.214) ≈ 0.500 sRGB → u16 ≈ 32767; allow ±64 for f16 precision.
        for pixel in out_img.pixels() {
            assert!((pixel[0] as i32 - 32767).abs() <= 64);
            assert!((pixel[1] as i32 - 32767).abs() <= 64);
            assert!((pixel[2] as i32 - 32767).abs() <= 64);
            assert_eq!(pixel[3], 65535);
        }
    }
}
