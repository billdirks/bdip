use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PastelPunchParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for PastelPunchParams {
    const ID: &'static str = "pastel_punch";
    const DISPLAY_NAME: &'static str = "Pastel Punch";
    const DESCRIPTION: &'static str = "Pushes colors toward white based on luminance, reducing saturation while \
         lifting brightness for a soft pastel look.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend strength toward white. 0 leaves the image unchanged; 1 applies the \
                      full luminance-driven pastel effect.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "pastel_punch",
        wgsl_source: include_str!("pastel_punch.wgsl"),
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
    PastelPunchParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_pastel_punch_registry_entry_exists() {
        assert!(registry_by_id("pastel_punch").is_some());
    }

    #[test]
    fn test_pastel_punch_registry_metadata() {
        let reg = registry_by_id("pastel_punch").unwrap();
        assert_eq!(reg.meta.display_name, "Pastel Punch");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend strength toward white. 0 leaves the image unchanged; 1 \
                              applies the full luminance-driven pastel effect.",
            }])
        );
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_pastel_punch_make_uniform_known_value() {
        let reg = registry_by_id("pastel_punch").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&PastelPunchParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    /// Identity: strength = 0.0 must leave the image unchanged.
    #[test]
    fn test_pastel_punch_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_punch",
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

    /// Alpha channel must not be affected by the pastel effect.
    #[test]
    fn test_pastel_punch_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_punch",
                values: vec![1.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by pastel_punch");
        }
    }

    /// Pure white input must remain white at any strength — blend toward white is a no-op.
    #[test]
    fn test_pastel_punch_white_stays_white() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 65535, 65535, 65535);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_punch",
                values: vec![1.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 64,
                "R: white should stay white, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 65535).abs() <= 64,
                "G: white should stay white, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 65535).abs() <= 64,
                "B: white should stay white, got {}",
                pixel[2]
            );
        }
    }

    /// Pure black input: luminance = 0, so blend factor = 0, output stays black.
    #[test]
    fn test_pastel_punch_black_stays_black() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 0, 0, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_punch",
                values: vec![1.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[0], 0, "R: black should stay black, got {}", pixel[0]);
            assert_eq!(pixel[1], 0, "G: black should stay black, got {}", pixel[1]);
            assert_eq!(pixel[2], 0, "B: black should stay black, got {}", pixel[2]);
        }
    }

    /// At full strength, a mid-gray input must be lifted toward white (output > input).
    #[test]
    fn test_pastel_punch_brightens_midgray() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% gray in sRGB (32767/65535): linear ≈ 0.214.
        // Luminance of a neutral gray equals that linear value (≈ 0.214).
        // blend = 0.214 * 1.0 = 0.214  →  output = mix(0.214, 1.0, 0.214) ≈ 0.377 linear.
        // 0.377 linear → sRGB ≈ 0.647 → u16 ≈ 42400.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_punch",
                values: vec![1.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[0] > 32767,
                "R: mid-gray should be lifted, got {}",
                pixel[0]
            );
            assert!(
                pixel[1] > 32767,
                "G: mid-gray should be lifted, got {}",
                pixel[1]
            );
            assert!(
                pixel[2] > 32767,
                "B: mid-gray should be lifted, got {}",
                pixel[2]
            );
        }
    }

    /// A fully saturated color must have its saturation reduced after the pastel effect.
    /// Red input (1, 0, 0 linear): luminance = 0.2126, blend = 0.2126.
    /// R: mix(1.0, 1.0, 0.2126) = 1.0; G: mix(0, 1.0, 0.2126) ≈ 0.2126; B same as G.
    /// G and B channels are lifted from 0, so channel spread (R - G) decreases.
    #[test]
    fn test_pastel_punch_desaturates_color() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure red in sRGB: 65535, 0, 0 → linear (1.0, 0.0, 0.0).
        let img = make_solid_image(2, 2, 65535, 0, 0);
        let out_no_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_punch",
                values: vec![0.0],
            }],
        );
        let out_full_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pastel_punch",
                values: vec![1.0],
            }],
        );

        let before_spread = out_no_effect.pixels().next().unwrap();
        let after_spread = out_full_effect.pixels().next().unwrap();

        let spread_before = before_spread[0] as i32 - before_spread[1] as i32;
        let spread_after = after_spread[0] as i32 - after_spread[1] as i32;

        assert!(
            spread_after < spread_before,
            "pastel effect should reduce R-G channel spread: before={}, after={}",
            spread_before,
            spread_after
        );
    }

    /// Chaining pastel_punch with brightness must preserve the alpha channel throughout.
    #[test]
    fn test_pastel_punch_chained_with_brightness() {
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
                    shader_id: "pastel_punch",
                    values: vec![0.5],
                },
            ],
        );

        for pixel in out_img.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be preserved through brightness+pastel_punch chain"
            );
        }
    }
}
