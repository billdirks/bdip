use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Monochrome Green shader.
///
/// The effect computes Rec.709 luminance then maps it to the green channel only,
/// producing the characteristic phosphor-green appearance of old monochrome CRT
/// monitors or a stylised night-vision display. `strength` is a linear blend between
/// the original image and the fully-converted result, so 0.0 is a mathematical identity.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MonochromeGreenParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full green-monochrome effect.
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for MonochromeGreenParams {
    const ID: &'static str = "monochrome_green";
    const DISPLAY_NAME: &'static str = "Monochrome Green";
    const DESCRIPTION: &'static str = "Converts the image to a green-tinted monochrome using Rec.709 luminance, \
         evoking a phosphor CRT or night-vision display.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend between the original image (0.0) and the full green-monochrome \
                      effect (1.0). The identity value is 0.0.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "monochrome_green",
        wgsl_source: include_str!("monochrome_green.wgsl"),
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
    MonochromeGreenParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_monochrome_green_registry_entry_exists() {
        assert!(registry_by_id("monochrome_green").is_some());
    }

    #[test]
    fn test_monochrome_green_registry_metadata() {
        let reg = registry_by_id("monochrome_green").unwrap();
        assert_eq!(reg.meta.display_name, "Monochrome Green");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend between the original image (0.0) and the full green-monochrome \
                              effect (1.0). The identity value is 0.0.",
            }])
        );
    }

    #[test]
    fn test_monochrome_green_passes_count() {
        let reg = registry_by_id("monochrome_green").unwrap();
        assert_eq!(
            reg.meta.passes.len(),
            1,
            "monochrome_green must have exactly 1 pass"
        );
    }

    #[test]
    fn test_monochrome_green_make_uniform_known_value() {
        let reg = registry_by_id("monochrome_green").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&MonochromeGreenParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 (the identity default) the output must equal the source image.
    #[test]
    fn test_monochrome_green_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 20000, 40000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "monochrome_green",
                values: vec![0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 20000).abs() <= 64,
                "R: expected ~20000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 40000).abs() <= 64,
                "G: expected ~40000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 50000).abs() <= 64,
                "B: expected ~50000, got {}",
                pixel[2]
            );
        }
    }

    /// At strength=1.0 the red and blue channels must be zero (green-only output).
    #[test]
    fn test_monochrome_green_full_strength_red_and_blue_are_zero() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a non-grey input so that the green channel is non-trivially derived.
        let img = make_solid_image(4, 4, 40000, 20000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "monochrome_green",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(
                pixel[0], 0,
                "R must be 0 at full strength, got {}",
                pixel[0]
            );
            assert_eq!(
                pixel[2], 0,
                "B must be 0 at full strength, got {}",
                pixel[2]
            );
        }
    }

    /// At strength=1.0 the green channel must carry the Rec.709 luminance of the source.
    ///
    /// Input: R=32767 (≈0.500 sRGB → ~0.214 linear), G=32767, B=32767 (neutral grey).
    /// Rec.709 luminance of a neutral grey equals each channel value in linear light.
    /// The expected green output in sRGB (u16) is approximately the same as the input.
    #[test]
    fn test_monochrome_green_full_strength_green_carries_luminance() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Neutral grey input: all channels equal → luminance = any channel value.
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "monochrome_green",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            // The green channel should closely match the input grey value (identity
            // luminance for a neutral grey), within f16 rounding tolerance.
            assert!(
                (pixel[1] as i32 - 32767).abs() <= 128,
                "G: expected ~32767 (luminance of neutral grey), got {}",
                pixel[1]
            );
        }
    }

    /// Alpha channel must pass through unchanged at any strength value.
    #[test]
    fn test_monochrome_green_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "monochrome_green",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved at full strength");
        }
    }

    /// At strength=0.5 each channel must lie strictly between the strength=0.0 (identity)
    /// and strength=1.0 (fully converted) outputs.
    #[test]
    fn test_monochrome_green_half_strength_blends() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a strongly-coloured input so the R channel moves a large distance.
        let img = make_solid_image(4, 4, 60000, 10000, 10000);
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "monochrome_green",
                values: vec![1.0],
            }],
        );
        let out_half = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "monochrome_green",
                values: vec![0.5],
            }],
        );
        // Source R=60000; full-strength R=0. At half strength R must lie strictly between.
        for (half_px, full_px) in out_half.pixels().zip(out_full.pixels()) {
            let src_r = 60000i32;
            let full_r = full_px[0] as i32;
            let half_r = half_px[0] as i32;
            let lo = src_r.min(full_r);
            let hi = src_r.max(full_r);
            assert!(
                half_r > lo && half_r < hi,
                "R at half strength ({half_r}) must lie strictly between \
                 source ({src_r}) and full-effect ({full_r})"
            );
        }
    }

    /// Pure black input must produce pure black output at any strength (luminance of
    /// black is 0, and 0 * any_vector = 0).
    #[test]
    fn test_monochrome_green_black_stays_black() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "monochrome_green",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[0], 0, "R: black input must stay 0");
            assert_eq!(pixel[1], 0, "G: black input luminance is 0");
            assert_eq!(pixel[2], 0, "B: black input must stay 0");
        }
    }

    /// Chaining monochrome_green with brightness must not panic and must preserve alpha.
    #[test]
    fn test_monochrome_green_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 20000, 40000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "monochrome_green",
                    values: vec![1.0],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 after chaining");
        }
    }
}
