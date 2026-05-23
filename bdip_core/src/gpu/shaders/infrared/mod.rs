use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Infrared shader.
///
/// The effect swaps the red and green channels, approximating the false-colour
/// appearance of infrared film where vegetation (a strong IR reflector) appears
/// bright and clear sky (an IR absorber) appears very dark. `strength` is a
/// linear blend between the unmodified source and the fully-swapped result,
/// so 0.0 is a mathematical identity.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InfraredParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full channel-swap effect.
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for InfraredParams {
    const ID: &'static str = "infrared";
    const DISPLAY_NAME: &'static str = "Infrared";
    const DESCRIPTION: &'static str = "Simulates false-colour infrared film by swapping the red and green channels, \
         making foliage appear bright and sky appear dark.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend between the original image (0.0) and the full infrared \
                      channel-swap effect (1.0). The identity value is 0.0.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "infrared",
        wgsl_source: include_str!("infrared.wgsl"),
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

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<InfraredParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_infrared_registry_entry_exists() {
        assert!(registry_by_id("infrared").is_some());
    }

    #[test]
    fn test_infrared_registry_metadata() {
        let reg = registry_by_id("infrared").unwrap();
        assert_eq!(reg.meta.display_name, "Infrared");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend between the original image (0.0) and the full infrared \
                              channel-swap effect (1.0). The identity value is 0.0.",
            }])
        );
        assert_eq!(
            reg.meta.passes.len(),
            1,
            "Infrared must have exactly 1 pass"
        );
    }

    #[test]
    fn test_infrared_make_uniform_known_value() {
        let reg = registry_by_id("infrared").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&InfraredParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 (the identity default) the output must equal the source image.
    /// mix(pixel, infrared, 0.0) = pixel, so no change is expected.
    #[test]
    fn test_infrared_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 20000, 40000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "infrared",
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

    /// At strength=1.0 red and green channels must be fully swapped.
    #[test]
    fn test_infrared_full_strength_swaps_red_and_green() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use distinct R/G values so the swap is clearly detectable.
        let img = make_solid_image(4, 4, 10000, 50000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "infrared",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            // After swap: R←G=50000, G←R=10000, B unchanged=30000.
            assert!(
                (pixel[0] as i32 - 50000).abs() <= 64,
                "R: expected ~50000 (was G), got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 10000).abs() <= 64,
                "G: expected ~10000 (was R), got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 30000).abs() <= 64,
                "B: expected ~30000 (unchanged), got {}",
                pixel[2]
            );
        }
    }

    /// Blue channel must not be affected by the red/green swap at any strength.
    #[test]
    fn test_infrared_blue_channel_unaffected() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 20000, 40000, 55000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "infrared",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[2] as i32 - 55000).abs() <= 64,
                "B: expected ~55000 (unchanged), got {}",
                pixel[2]
            );
        }
    }

    /// Alpha channel must pass through unchanged at any strength value.
    #[test]
    fn test_infrared_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "infrared",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// At strength=0.5 each channel must be strictly between the strength=0.0
    /// (identity) and strength=1.0 (fully-swapped) outputs.
    ///
    /// The blend is performed in linear light, so the sRGB output value is not the
    /// naive average of the sRGB input values. Testing that the result lies between
    /// the two extreme outputs is a robust way to verify the blend without requiring
    /// an exact linear-to-sRGB computation.
    #[test]
    fn test_infrared_half_strength_blends_channels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use distinct R and G values so the swap produces a measurably different result.
        let img = make_solid_image(4, 4, 10000, 50000, 20000);
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "infrared",
                values: vec![1.0],
            }],
        );
        let out_half = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "infrared",
                values: vec![0.5],
            }],
        );
        // The source R=10000 and the full-swap R=50000 (the original G).
        // At half strength the output R must be between the source and the full-swap output.
        for (half_px, full_px) in out_half.pixels().zip(out_full.pixels()) {
            let src_r = 10000i32;
            let full_r = full_px[0] as i32;
            let half_r = half_px[0] as i32;
            let lo = src_r.min(full_r);
            let hi = src_r.max(full_r);
            assert!(
                half_r > lo && half_r < hi,
                "R at half strength ({half_r}) must lie strictly between \
                 source ({src_r}) and full-swap ({full_r})"
            );
        }
    }

    /// Chaining infrared with brightness must not panic and must preserve alpha.
    #[test]
    fn test_infrared_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 20000, 40000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "infrared",
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
