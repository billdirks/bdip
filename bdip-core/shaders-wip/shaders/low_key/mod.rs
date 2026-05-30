use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LowKeyParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for LowKeyParams {
    const ID: &'static str = "low_key";
    const DISPLAY_NAME: &'static str = "Low Key";
    const DESCRIPTION: &'static str = "Simulates low-key lighting: drops exposure and boosts contrast so shadows \
         crush to black while highlights remain partially visible.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Intensity of the low-key effect. 0 is no change; 1 is fully low-key.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "low_key",
        wgsl_source: include_str!("low_key.wgsl"),
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

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<LowKeyParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    fn make_engine_and_renderer() -> (GpuEngine, Renderer) {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);
        (engine, renderer)
    }

    #[test]
    fn test_low_key_registry_entry_exists() {
        assert!(registry_by_id("low_key").is_some());
    }

    #[test]
    fn test_low_key_registry_metadata() {
        let reg = registry_by_id("low_key").unwrap();
        assert_eq!(reg.meta.display_name, "Low Key");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Intensity of the low-key effect. 0 is no change; 1 is fully low-key.",
            }])
        );
    }

    #[test]
    fn test_low_key_make_uniform_known_value() {
        let reg = registry_by_id("low_key").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&LowKeyParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_low_key_identity_at_zero_strength() {
        // strength=0: exp_scale=1, contrast_scale=1 — the formula reduces to identity.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 10794, 25700, 51400);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "low_key",
                values: vec![0.0],
            }],
        );

        // sRGB→linear→sRGB round-trip with f16 precision; allow ±64 LSB.
        for pixel in out.pixels() {
            assert!((pixel[0] as i32 - 10794).abs() <= 64);
            assert!((pixel[1] as i32 - 25700).abs() <= 64);
            assert!((pixel[2] as i32 - 51400).abs() <= 64);
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_low_key_mid_gray_crushes_to_black_at_full_strength() {
        // 50% sRGB (32767 u16) ≈ 0.214 linear.
        // At strength=1: darkened = 0.214 * 0.25 = 0.054; contrast = (0.054-0.25)*3+0.25 = -0.34
        // → clamped to 0 → pure black output.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "low_key",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(
                pixel[0], 0,
                "mid-gray R should crush to black at full strength"
            );
            assert_eq!(
                pixel[1], 0,
                "mid-gray G should crush to black at full strength"
            );
            assert_eq!(
                pixel[2], 0,
                "mid-gray B should crush to black at full strength"
            );
        }
    }

    #[test]
    fn test_low_key_pure_white_retains_partial_brightness_at_full_strength() {
        // Pure white (1.0 linear) at strength=1:
        //   darkened = 1.0 * 0.25 = 0.25 linear
        //   contrast = (0.25 - 0.25) * 3 + 0.25 = 0.25 linear
        //   sRGB ≈ 0.502 → u16 ≈ 32_920
        // The highlight lands at the contrast midpoint and stays visibly gray —
        // confirming highlights are not crushed to black.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "low_key",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            // Must be significantly above zero (not crushed to black).
            assert!(
                pixel[0] > 20000,
                "pure white should retain partial brightness, got R={}",
                pixel[0]
            );
            // Must also be darker than the original white (effect was applied).
            assert!(
                pixel[0] < 65535,
                "pure white must be darkened by low-key, got R={}",
                pixel[0]
            );
        }
    }

    #[test]
    fn test_low_key_pure_black_stays_black() {
        // Pure black (0 linear) at any strength:
        //   darkened = 0 * exp_scale = 0
        //   contrast = (0 - midpoint) * scale + midpoint
        // The contrast formula shifts black below zero → clamped back to 0.
        // So pure black input always produces pure black output.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "low_key",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[0], 0, "pure black should remain black");
            assert_eq!(pixel[1], 0, "pure black should remain black");
            assert_eq!(pixel[2], 0, "pure black should remain black");
        }
    }

    #[test]
    fn test_low_key_alpha_preserved() {
        // Alpha channel must pass through unmodified regardless of strength.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "low_key",
                values: vec![0.5],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    #[test]
    fn test_low_key_darkens_image_at_full_strength() {
        // Overall image should be darker at strength=1 vs strength=0 for a bright input.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 50000, 50000, 50000);

        let out_identity = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "low_key",
                values: vec![0.0],
            }],
        );
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "low_key",
                values: vec![1.0],
            }],
        );

        for (identity_pixel, full_pixel) in out_identity.pixels().zip(out_full.pixels()) {
            assert!(
                full_pixel[0] < identity_pixel[0],
                "low_key at strength=1 must darken the image; identity R={}, full R={}",
                identity_pixel[0],
                full_pixel[0]
            );
        }
    }

    #[test]
    fn test_low_key_chaining_with_brightness() {
        // Chain low_key into a zero-offset brightness to verify in-VRAM handoff.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 50000, 50000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "low_key",
                    values: vec![0.5],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );

        // brightness at 0 is identity — output must still be darker than the original input.
        for pixel in out.pixels() {
            assert!(
                pixel[0] < 50000,
                "chained output should be darker than original input, got R={}",
                pixel[0]
            );
        }
    }
}
