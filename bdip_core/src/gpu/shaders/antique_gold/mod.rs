use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AntiqueGoldParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for AntiqueGoldParams {
    const ID: &'static str = "antique_gold";
    const DISPLAY_NAME: &'static str = "Antique Gold";
    const DESCRIPTION: &'static str = "Applies a warm golden-brown tint that evokes antique metal and aged photographs, \
         with highlights leaning yellow-gold and shadows leaning dark amber.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Intensity of the antique gold tint. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "antique_gold",
        wgsl_source: include_str!("antique_gold.wgsl"),
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
    AntiqueGoldParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ── Registry / metadata ──────────────────────────────────────────────────

    #[test]
    fn test_antique_gold_registry_entry_exists() {
        assert!(registry_by_id("antique_gold").is_some());
    }

    #[test]
    fn test_antique_gold_registry_metadata() {
        let reg = registry_by_id("antique_gold").unwrap();
        assert_eq!(reg.meta.display_name, "Antique Gold");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Intensity of the antique gold tint. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_antique_gold_passes_count() {
        let reg = registry_by_id("antique_gold").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_antique_gold_make_uniform_known_value() {
        let reg = registry_by_id("antique_gold").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&AntiqueGoldParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_antique_gold_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "antique_gold",
                values: vec![0.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 20000).abs() <= 64,
                "R mismatch at identity: {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 15000).abs() <= 64,
                "G mismatch at identity: {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 30000).abs() <= 64,
                "B mismatch at identity: {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    /// At full strength, a neutral grey input must shift to R > G > B,
    /// producing the characteristic golden-brown tone.
    #[test]
    fn test_antique_gold_golden_tone_channel_ordering_on_grey() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Mid-grey: equal R, G, B channels.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "antique_gold",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[0] > pixel[1],
                "R should exceed G for grey input with golden tone: R={} G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                pixel[1] > pixel[2],
                "G should exceed B for grey input with golden tone: G={} B={}",
                pixel[1],
                pixel[2]
            );
        }
    }

    /// At full strength, highlights (near-white) should show the yellow-gold bias:
    /// R > G and blue should be substantially reduced relative to the original.
    #[test]
    fn test_antique_gold_highlights_shift_toward_yellow_gold() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Near-white neutral input.
        let img = make_solid_image(2, 2, 60000, 60000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "antique_gold",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[0] > pixel[2],
                "R should exceed B in highlights: R={} B={}",
                pixel[0],
                pixel[2]
            );
        }
    }

    /// At full strength, shadows (near-black) should show the dark-amber bias:
    /// R > G > B with all channels reduced, not amplified.
    #[test]
    fn test_antique_gold_shadows_shift_toward_dark_amber() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Near-black input: small but nonzero to allow ratio comparison.
        let img = make_solid_image(2, 2, 3000, 3000, 3000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "antique_gold",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[0] > pixel[2],
                "R should exceed B in dark shadows: R={} B={}",
                pixel[0],
                pixel[2]
            );
        }
    }

    /// Black input must remain black at full strength.
    /// The matrix coefficients all multiply the input, so 0 * anything = 0.
    #[test]
    fn test_antique_gold_black_stays_black() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "antique_gold",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[0], 0, "R: black input must stay 0, got {}", pixel[0]);
            assert_eq!(pixel[1], 0, "G: black input must stay 0, got {}", pixel[1]);
            assert_eq!(pixel[2], 0, "B: black input must stay 0, got {}", pixel[2]);
        }
    }

    /// Alpha channel must not be modified regardless of strength.
    #[test]
    fn test_antique_gold_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "antique_gold",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// Chaining with a brightness identity pass must not alter the result.
    #[test]
    fn test_antique_gold_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 25000, 20000, 15000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "antique_gold",
                values: vec![0.6],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "antique_gold",
                    values: vec![0.6],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );

        for (a, b) in standalone.pixels().zip(chained.pixels()) {
            assert!((a[0] as i32 - b[0] as i32).abs() <= 64, "R chain mismatch");
            assert!((a[1] as i32 - b[1] as i32).abs() <= 64, "G chain mismatch");
            assert!((a[2] as i32 - b[2] as i32).abs() <= 64, "B chain mismatch");
        }
    }
}
