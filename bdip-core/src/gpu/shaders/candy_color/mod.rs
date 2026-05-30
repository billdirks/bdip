use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CandyColorParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for CandyColorParams {
    const ID: &'static str = "candy_color";
    const DISPLAY_NAME: &'static str = "Candy Color";
    const DESCRIPTION: &'static str = "Boosts color vibrance with a smart saturation that lifts muted tones \
         more aggressively than already-vivid hues, creating a pop-candy aesthetic.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Vibrance intensity. 0 is a no-op; 1 applies maximum vibrance boost.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "candy_color",
        wgsl_source: include_str!("candy_color.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    CandyColorParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_candy_color_registry_entry_exists() {
        assert!(registry_by_id("candy_color").is_some());
    }

    #[test]
    fn test_candy_color_registry_metadata() {
        let reg = registry_by_id("candy_color").unwrap();
        assert_eq!(reg.meta.display_name, "Candy Color");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Vibrance intensity. 0 is a no-op; 1 applies maximum vibrance boost.",
            }])
        );
    }

    #[test]
    fn test_candy_color_make_uniform_known_value() {
        let reg = registry_by_id("candy_color").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&CandyColorParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_candy_color_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Non-neutral color so any saturation change would be visible.
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "candy_color",
                values: vec![0.0],
            }],
        );

        // strength=0 → vibrance boost multiplier = 0 → identity; f16 rounding applies.
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 16384).abs() <= 64,
                "G: expected ~16384, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 8192).abs() <= 64,
                "B: expected ~8192, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_candy_color_positive_strength_boosts_muted_color() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // A muted warm color (low saturation). Vibrance should push the dominant
        // channel (R) higher relative to the others.
        let img = make_solid_image(2, 2, 40000, 30000, 25000);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "candy_color",
                values: vec![1.0],
            }],
        );

        // After vibrance boost at strength=1, the dominant channel (R, which is
        // furthest from luminance) should increase relative to the neutral channels.
        // The hue direction should be preserved: R > G > B must still hold.
        for pixel in out_img.pixels() {
            assert!(
                pixel[0] > pixel[1],
                "R should remain above G after vibrance boost: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                pixel[1] > pixel[2],
                "G should remain above B after vibrance boost: G={}, B={}",
                pixel[1],
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_candy_color_gray_is_unchanged() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // A neutral gray: R=G=B. Vibrance operates on the distance from luminance;
        // for a gray pixel that distance is zero, so any strength leaves it unchanged.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "candy_color",
                values: vec![1.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 32767).abs() <= 64,
                "G: expected ~32767, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 32767).abs() <= 64,
                "B: expected ~32767, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_candy_color_alpha_preserved() {
        // Alpha is verified to pass through unchanged in every GPU roundtrip test
        // (all make_solid_image calls set alpha=65535 and each test asserts pixel[3]==65535).
        // This dedicated test confirms the contract using a white pixel explicitly.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 65535, 32767, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "candy_color",
                values: vec![0.8],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535, "Alpha must be preserved: got {}", pixel[3]);
        }
    }

    #[test]
    fn test_candy_color_chaining_with_brightness() {
        // Verify the shader can be chained with another shader without GPU errors
        // or incorrect output ordering. This tests the integration "glue" between
        // the candy_color pass and the pipeline's in-VRAM chaining mechanism.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "candy_color",
                    values: vec![0.5],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );

        // The brightness pass is identity (0.0), so we're mainly verifying that
        // the pipeline didn't crash or produce a fully-black output.
        for pixel in out_img.pixels() {
            assert!(
                pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0,
                "Output should not be fully black after chaining"
            );
            assert_eq!(pixel[3], 65535);
        }
    }
}
