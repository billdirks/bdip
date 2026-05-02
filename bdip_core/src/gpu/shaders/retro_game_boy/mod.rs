use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RetroGameBoyParams {
    pub palette_intensity: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for RetroGameBoyParams {
    const ID: &'static str = "retro_game_boy";
    const DISPLAY_NAME: &'static str = "Retro Game Boy";
    const DESCRIPTION: &'static str = "Simulates the original DMG Game Boy LCD: converts to grayscale, quantizes to 4 \
         brightness levels, and tints with the classic pea-green/olive palette.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Palette Intensity",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend strength of the Game Boy palette. 0.0 leaves the image unchanged; \
                      1.0 applies the full 4-shade pea-green tint.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "retro_game_boy",
        wgsl_source: include_str!("retro_game_boy.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            palette_intensity: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    RetroGameBoyParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_retro_game_boy_registry_entry_exists() {
        assert!(registry_by_id("retro_game_boy").is_some());
    }

    #[test]
    fn test_retro_game_boy_registry_metadata() {
        let reg = registry_by_id("retro_game_boy").unwrap();
        assert_eq!(reg.meta.display_name, "Retro Game Boy");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Palette Intensity",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend strength of the Game Boy palette. 0.0 leaves the image \
                              unchanged; 1.0 applies the full 4-shade pea-green tint.",
            }])
        );
    }

    #[test]
    fn test_retro_game_boy_passes_count() {
        let reg = registry_by_id("retro_game_boy").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_retro_game_boy_make_uniform_known_value() {
        let reg = registry_by_id("retro_game_boy").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&RetroGameBoyParams {
            palette_intensity: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // Identity: palette_intensity=0.0 must leave the image unchanged.
    // The default value (0.0) results in the gray = mix(gray, palette, 0.0) = gray
    // passthrough, which for a neutral gray input reproduces the original pixel values.
    #[test]
    fn test_retro_game_boy_identity_at_zero_intensity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Neutral gray image; sRGB/linear round-trip is the identity up to f16 error.
        let img = make_solid_image(2, 2, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_game_boy",
                values: vec![0.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 20000).abs() <= 64,
                "R: identity expected ~20000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 20000).abs() <= 64,
                "G: identity expected ~20000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 20000).abs() <= 64,
                "B: identity expected ~20000, got {}",
                pixel[2]
            );
        }
    }

    // Alpha must not be modified by any palette_intensity value.
    #[test]
    fn test_retro_game_boy_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_game_boy",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    // Full palette (intensity=1.0): mid-bucket luma 0.125 must map to Level 0 palette colour.
    // Level 0 palette (linear): (0.0048, 0.0395, 0.0048)
    // Re-encoded to sRGB u16: R≈3855, G≈14392, B≈3855. Tolerance 512 for f16 quantization.
    //
    // To hit luma=0.125 reliably, we use a neutral gray input whose linear value equals 0.125.
    // sRGB gray for linear=0.125: sRGB≈0.3886 → u16≈25465.
    #[test]
    fn test_retro_game_boy_level0_palette_at_full_intensity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 25465, 25465, 25465);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_game_boy",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 3855).abs() <= 512,
                "R: level 0 expected ~3855, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 14392).abs() <= 512,
                "G: level 0 expected ~14392, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 3855).abs() <= 512,
                "B: level 0 expected ~3855, got {}",
                pixel[2]
            );
        }
    }

    // Full palette (intensity=1.0): mid-bucket luma 0.375 must map to Level 1 palette colour.
    // Level 1 palette (linear): (0.0296, 0.1221, 0.0296)
    // Re-encoded to sRGB u16: R≈12336, G≈25186, B≈12336. Tolerance 512 for f16 quantization.
    //
    // sRGB gray for linear=0.375: sRGB≈0.6461 → u16≈42341.
    #[test]
    fn test_retro_game_boy_level1_palette_at_full_intensity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 42341, 42341, 42341);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_game_boy",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 12336).abs() <= 512,
                "R: level 1 expected ~12336, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 25186).abs() <= 512,
                "G: level 1 expected ~25186, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 12336).abs() <= 512,
                "B: level 1 expected ~12336, got {}",
                pixel[2]
            );
        }
    }

    // Full palette (intensity=1.0): mid-bucket luma 0.625 must map to Level 2 palette colour.
    // Level 2 palette (linear): (0.2582, 0.4125, 0.0048)
    // Re-encoded to sRGB u16: R≈35723, G≈44204, B≈3855. Tolerance 512 for f16 quantization.
    //
    // sRGB gray for linear=0.625: sRGB≈0.8124 → u16≈53238.
    #[test]
    fn test_retro_game_boy_level2_palette_at_full_intensity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 53238, 53238, 53238);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_game_boy",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 35723).abs() <= 512,
                "R: level 2 expected ~35723, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 44204).abs() <= 512,
                "G: level 2 expected ~44204, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 3855).abs() <= 512,
                "B: level 2 expected ~3855, got {}",
                pixel[2]
            );
        }
    }

    // Full palette (intensity=1.0): mid-bucket luma 0.875 must map to Level 3 palette colour.
    // Level 3 palette (linear): (0.3278, 0.5029, 0.0048)
    // Re-encoded to sRGB u16: R≈39835, G≈48316, B≈3855. Tolerance 512 for f16 quantization.
    //
    // sRGB gray for linear=0.875: sRGB≈0.9429 → u16≈61793.
    #[test]
    fn test_retro_game_boy_level3_palette_at_full_intensity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 61793, 61793, 61793);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_game_boy",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 39835).abs() <= 512,
                "R: level 3 expected ~39835, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 48316).abs() <= 512,
                "G: level 3 expected ~48316, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 3855).abs() <= 512,
                "B: level 3 expected ~3855, got {}",
                pixel[2]
            );
        }
    }

    // Black input (all zeros) must map to Level 0 at full intensity.
    #[test]
    fn test_retro_game_boy_black_maps_to_level0() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_game_boy",
                values: vec![1.0],
            }],
        );

        // Level 0: R≈3855, G≈14392, B≈3855
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 3855).abs() <= 512,
                "R: black should map to level-0 R (~3855), got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 14392).abs() <= 512,
                "G: black should map to level-0 G (~14392), got {}",
                pixel[1]
            );
        }
    }

    // The green channel must exceed the red channel for all palette levels, reflecting
    // the characteristic pea-green bias of the DMG screen.
    #[test]
    fn test_retro_game_boy_green_dominant_across_all_levels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Test one representative input per bucket: ~12.5%, ~37.5%, ~62.5%, ~87.5% luma.
        for u16_val in [25465u16, 42341, 53238, 61793] {
            let img = make_solid_image(2, 2, u16_val, u16_val, u16_val);
            let out = roundtrip(
                &mut renderer,
                &engine,
                &img,
                &[Transform {
                    shader_id: "retro_game_boy",
                    values: vec![1.0],
                }],
            );

            for pixel in out.pixels() {
                assert!(
                    pixel[1] > pixel[0],
                    "G should exceed R at u16_val={}: G={}, R={}",
                    u16_val,
                    pixel[1],
                    pixel[0]
                );
            }
        }
    }

    // Chaining with brightness must not break the green-dominant property of the palette.
    #[test]
    fn test_retro_game_boy_chained_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "brightness",
                    values: vec![0.1],
                },
                Transform {
                    shader_id: "retro_game_boy",
                    values: vec![1.0],
                },
            ],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[1] > pixel[0],
                "G should exceed R after brightness+retro_game_boy: G={}, R={}",
                pixel[1],
                pixel[0]
            );
        }
    }
}
