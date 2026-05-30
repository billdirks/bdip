use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PastelDreamsParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for PastelDreamsParams {
    const ID: &'static str = "pastel_dreams";
    const DISPLAY_NAME: &'static str = "Pastel Dreams";
    const DESCRIPTION: &'static str = "Creates a soft pastel aesthetic by lifting brightness toward white \
         while simultaneously reducing saturation.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Effect intensity. 0 leaves the image unchanged; 1 fully applies the \
                      high-brightness, low-saturation pastel look.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "pastel_dreams",
        wgsl_source: include_str!("pastel_dreams.wgsl"),
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
    PastelDreamsParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_pastel_dreams_registry_entry_exists() {
        assert!(registry_by_id("pastel_dreams").is_some());
    }

    #[test]
    fn test_pastel_dreams_registry_metadata() {
        let reg = registry_by_id("pastel_dreams").unwrap();
        assert_eq!(reg.meta.display_name, "Pastel Dreams");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Effect intensity. 0 leaves the image unchanged; 1 fully applies \
                              the high-brightness, low-saturation pastel look.",
            }])
        );
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_pastel_dreams_make_uniform_known_value() {
        let reg = registry_by_id("pastel_dreams").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&PastelDreamsParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    /// Identity: strength = 0.0 must leave the image unchanged.
    #[test]
    fn test_pastel_dreams_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_dreams",
                values: vec![0.0],
            }],
        );

        // sRGB → linear → sRGB round-trip introduces ≤64/65535 error from f16 rounding.
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
        }
    }

    /// Alpha channel must not be affected by the pastel effect at any strength.
    #[test]
    fn test_pastel_dreams_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_dreams",
                values: vec![1.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by pastel_dreams");
        }
    }

    /// At full strength, a mid-gray input must be lifted toward white (output > input).
    #[test]
    fn test_pastel_dreams_brightens_midtones_at_full_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% gray in sRGB (32767/65535 ≈ 0.500 sRGB → ~0.214 linear).
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_dreams",
                values: vec![1.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[0] > 32767,
                "R: midtone should be lifted toward white at full strength, got {}",
                pixel[0]
            );
            assert!(
                pixel[1] > 32767,
                "G: midtone should be lifted toward white at full strength, got {}",
                pixel[1]
            );
            assert!(
                pixel[2] > 32767,
                "B: midtone should be lifted toward white at full strength, got {}",
                pixel[2]
            );
        }
    }

    /// A fully saturated color must have reduced channel spread (lower saturation) after
    /// the effect. The R-G channel difference for a red input should decrease.
    #[test]
    fn test_pastel_dreams_reduces_saturation_at_full_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure red: R=65535, G=0, B=0 in sRGB → linear (1.0, 0.0, 0.0).
        let img = make_solid_image(2, 2, 65535, 0, 0);

        let out_no_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_dreams",
                values: vec![0.0],
            }],
        );
        let out_full_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_dreams",
                values: vec![1.0],
            }],
        );

        let pixel_before = out_no_effect.pixels().next().unwrap();
        let pixel_after = out_full_effect.pixels().next().unwrap();

        let spread_before = pixel_before[0] as i32 - pixel_before[1] as i32;
        let spread_after = pixel_after[0] as i32 - pixel_after[1] as i32;

        assert!(
            spread_after < spread_before,
            "R-G spread should decrease with pastel_dreams: before={spread_before}, \
             after={spread_after}"
        );
    }

    /// Pure white input must remain at or near white at any strength — lifting toward
    /// white is a no-op for an already-white pixel.
    #[test]
    fn test_pastel_dreams_white_stays_near_white() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 65535, 65535, 65535);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_dreams",
                values: vec![1.0],
            }],
        );

        // Brightness lift pushes values above 1.0 (clamped on readback) — white stays white.
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 256,
                "R: white should stay near white, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 65535).abs() <= 256,
                "G: white should stay near white, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 65535).abs() <= 256,
                "B: white should stay near white, got {}",
                pixel[2]
            );
        }
    }

    /// Chaining pastel_dreams with brightness must produce valid output with alpha intact.
    #[test]
    fn test_pastel_dreams_chained_with_brightness_preserves_alpha() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "brightness",
                    values: vec![0.1],
                },
                Transform {
                    shader_id: "pastel_dreams",
                    values: vec![0.5],
                },
            ],
        );

        for pixel in out_img.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be preserved through brightness+pastel_dreams chain"
            );
        }
    }
}
