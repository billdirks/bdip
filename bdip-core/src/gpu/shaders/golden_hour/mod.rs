use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GoldenHourParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for GoldenHourParams {
    const ID: &'static str = "golden_hour";
    const DISPLAY_NAME: &'static str = "Golden Hour";
    const DESCRIPTION: &'static str = "Simulates golden-hour lighting by applying a warm color temperature shift \
         that boosts reds and oranges, slightly boosts greens, reduces blues, and \
         adds a warm amber tint to shadows and midtones.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // Identity: no warm shift applied.
        description: "Intensity of the golden-hour effect. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "golden_hour",
        wgsl_source: include_str!("golden_hour.wgsl"),
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
    GoldenHourParams,
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
    fn test_golden_hour_registry_entry_exists() {
        assert!(registry_by_id("golden_hour").is_some());
    }

    #[test]
    fn test_golden_hour_registry_metadata() {
        let reg = registry_by_id("golden_hour").unwrap();
        assert_eq!(reg.meta.display_name, "Golden Hour");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Intensity of the golden-hour effect. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_golden_hour_passes_count() {
        let reg = registry_by_id("golden_hour").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_golden_hour_make_uniform_known_value() {
        let reg = registry_by_id("golden_hour").unwrap();
        let bytes = (reg.make_uniform)(&[0.6]);
        let expected = bytemuck::bytes_of(&GoldenHourParams {
            strength: 0.6,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_golden_hour_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "golden_hour",
                values: vec![0.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 20000).abs() <= 64,
                "R mismatch: {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 15000).abs() <= 64,
                "G mismatch: {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 30000).abs() <= 64,
                "B mismatch: {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    /// The warm channel scaling must increase R and decrease B relative to the input
    /// when strength > 0, on a neutral (grey) midtone input.
    #[test]
    fn test_golden_hour_warm_shift_increases_red_decreases_blue() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Mid-grey neutral input: equal R, G, B channels.
        let img = make_solid_image(2, 2, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "golden_hour",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[0] > 30000,
                "R should increase with warm shift: got {}",
                pixel[0]
            );
            assert!(
                pixel[2] < 30000,
                "B should decrease with warm shift: got {}",
                pixel[2]
            );
        }
    }

    /// Shadows (dark pixels) should shift toward warm amber: R and G should dominate
    /// over B after applying full strength, reflecting the warm tint blend.
    #[test]
    fn test_golden_hour_shadows_shift_toward_warm_amber() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Near-black input: maximum shadow weight for the warm tint.
        let img = make_solid_image(2, 2, 800, 800, 800);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "golden_hour",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            // Amber: R > B, indicating warm color shift in shadows.
            assert!(
                pixel[0] > pixel[2],
                "R should exceed B in warm-shifted shadow: R={} B={}",
                pixel[0],
                pixel[2]
            );
        }
    }

    /// Bright highlights (near-white) sit above the warm-tint zone (lum > 0.7) and
    /// should not receive the tint blend. Only the global channel scaling applies.
    #[test]
    fn test_golden_hour_highlights_channel_scale_only() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Near-white neutral input: lum ≈ 0.9 linear, above the tint threshold.
        let img = make_solid_image(2, 2, 62000, 62000, 62000);
        let out_with = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "golden_hour",
                values: vec![1.0],
            }],
        );
        let out_without = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "golden_hour",
                values: vec![0.0],
            }],
        );

        // Highlights should be affected (channel scale), not identical to input.
        let r_diff = out_with.pixels().next().unwrap()[0] as i32
            - out_without.pixels().next().unwrap()[0] as i32;
        let b_diff = out_with.pixels().next().unwrap()[2] as i32
            - out_without.pixels().next().unwrap()[2] as i32;
        // R should have increased and B should have decreased due to channel scaling.
        assert!(
            r_diff > 0,
            "R should be higher at full strength in highlights"
        );
        assert!(
            b_diff < 0,
            "B should be lower at full strength in highlights"
        );
    }

    /// Alpha channel must not be modified regardless of strength.
    #[test]
    fn test_golden_hour_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "golden_hour",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_golden_hour_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 15000, 10000, 25000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "golden_hour",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "golden_hour",
                    values: vec![0.5],
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
