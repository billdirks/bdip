use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, PassScale, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrayscaleParams {
    pub _unused: [f32; 4],
}

impl TransformShader for GrayscaleParams {
    const ID: &'static str = "grayscale";
    const DISPLAY_NAME: &'static str = "Grayscale";
    const PARAM: ParamKind = ParamKind::Toggle;
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "grayscale",
        wgsl_source: include_str!("grayscale.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(_: &[f32]) -> Self {
        Self { _unused: [0.0; 4] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    GrayscaleParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_grayscale_registry_entry_exists() {
        assert!(registry_by_id("grayscale").is_some());
    }

    #[test]
    fn test_grayscale_registry_metadata() {
        let reg = registry_by_id("grayscale").unwrap();
        assert_eq!(reg.meta.display_name, "Grayscale");
        assert_eq!(reg.meta.param, ParamKind::Toggle);
    }

    #[test]
    fn test_grayscale_make_uniform_known_value() {
        let reg = registry_by_id("grayscale").unwrap();
        let bytes = (reg.make_uniform)(&[]);
        let expected = bytemuck::bytes_of(&GrayscaleParams { _unused: [0.0; 4] });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_transform_display_toggle() {
        let t = Transform {
            shader_id: "grayscale",
            values: vec![],
        };
        assert_eq!(t.to_string(), "Grayscale");
    }

    #[test]
    fn test_grayscale_produces_equal_rgb_channels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Colored input: channels are distinct, so any non-trivial operation is detectable.
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "grayscale",
                values: vec![],
            }],
        );

        // After grayscale, R, G, and B must all equal the luminance value.
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "R and G should be equal: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[1] as i32 - pixel[2] as i32).abs() <= 64,
                "G and B should be equal: G={}, B={}",
                pixel[1],
                pixel[2]
            );
        }
    }

    #[test]
    fn test_grayscale_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "grayscale",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by grayscale");
        }
    }

    #[test]
    fn test_grayscale_all_black_stays_black() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure black: all channels 0 linear → luminance = 0 → output stays 0.
        let img = make_solid_image(2, 2, 0, 0, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "grayscale",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(
                pixel[0], 0,
                "R: black input should produce 0, got {}",
                pixel[0]
            );
            assert_eq!(
                pixel[1], 0,
                "G: black input should produce 0, got {}",
                pixel[1]
            );
            assert_eq!(
                pixel[2], 0,
                "B: black input should produce 0, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_grayscale_all_white_stays_white() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure white: all channels 1.0 linear → luminance = 0.2126+0.7152+0.0722 = 1.0 → white.
        let img = make_solid_image(2, 2, 65535, 65535, 65535);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "grayscale",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 64,
                "R: white input should stay white, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 65535).abs() <= 64,
                "G: white input should stay white, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 65535).abs() <= 64,
                "B: white input should stay white, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_grayscale_chained_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Apply brightness first to shift the values, then grayscale.
        // The result must still have equal R=G=B channels.
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "brightness",
                    values: vec![0.2],
                },
                Transform {
                    shader_id: "grayscale",
                    values: vec![],
                },
            ],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "R and G should be equal after brightness+grayscale: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[1] as i32 - pixel[2] as i32).abs() <= 64,
                "G and B should be equal after brightness+grayscale: G={}, B={}",
                pixel[1],
                pixel[2]
            );
        }
    }
}
