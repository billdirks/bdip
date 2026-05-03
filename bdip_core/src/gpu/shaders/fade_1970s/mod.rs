use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Fade1970sParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for Fade1970sParams {
    const ID: &'static str = "fade_1970s";
    const DISPLAY_NAME: &'static str = "1970s Fade";
    const DESCRIPTION: &'static str = "Simulates the faded, warm color cast of 1970s film photography: lifted blacks, \
         warm orange-brown midtones, faint yellow-green highlight tint, and muted saturation.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // Identity: no fading applied.
        description: "Intensity of the 1970s fade effect. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "fade_1970s",
        wgsl_source: include_str!("fade_1970s.wgsl"),
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
    Fade1970sParams,
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
    fn test_fade_1970s_registry_entry_exists() {
        assert!(registry_by_id("fade_1970s").is_some());
    }

    #[test]
    fn test_fade_1970s_registry_metadata() {
        let reg = registry_by_id("fade_1970s").unwrap();
        assert_eq!(reg.meta.display_name, "1970s Fade");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Intensity of the 1970s fade effect. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_fade_1970s_passes_count() {
        let reg = registry_by_id("fade_1970s").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_fade_1970s_make_uniform_known_value() {
        let reg = registry_by_id("fade_1970s").unwrap();
        let bytes = (reg.make_uniform)(&[0.7]);
        let expected = bytemuck::bytes_of(&Fade1970sParams {
            strength: 0.7,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_fade_1970s_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fade_1970s",
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

    /// Pure black input must be lifted above zero at full strength (raised black floor).
    #[test]
    fn test_fade_1970s_black_point_lifted() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fade_1970s",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[0] > 0,
                "R black floor should be lifted: got {}",
                pixel[0]
            );
            assert!(
                pixel[1] > 0,
                "G black floor should be lifted: got {}",
                pixel[1]
            );
            assert!(
                pixel[2] > 0,
                "B black floor should be lifted: got {}",
                pixel[2]
            );
        }
    }

    /// Lifted blacks must be warm: R and G should be higher than B in the lifted shadow.
    #[test]
    fn test_fade_1970s_black_lift_is_warm() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fade_1970s",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[0] > pixel[2],
                "lifted shadow R should exceed B (warm lift): R={} B={}",
                pixel[0],
                pixel[2]
            );
        }
    }

    /// Warm channel scale: neutral midtone input should have R increased and B decreased
    /// relative to the original at full strength.
    #[test]
    fn test_fade_1970s_warm_shift_increases_red_decreases_blue() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Neutral mid-grey: equal R, G, B channels.
        let img = make_solid_image(2, 2, 32000, 32000, 32000);
        let out_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fade_1970s",
                values: vec![1.0],
            }],
        );
        let out_identity = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fade_1970s",
                values: vec![0.0],
            }],
        );

        for (e, i) in out_effect.pixels().zip(out_identity.pixels()) {
            assert!(
                e[0] > i[0],
                "R should increase with warm shift: effect={} identity={}",
                e[0],
                i[0]
            );
            assert!(
                e[2] < i[2],
                "B should decrease with warm shift: effect={} identity={}",
                e[2],
                i[2]
            );
        }
    }

    /// Bright highlight input should exhibit a yellow-green tint: at full strength,
    /// green should become elevated relative to blue (G > B), reflecting the
    /// highlight tint toward pale yellow-green.
    #[test]
    fn test_fade_1970s_highlights_have_yellow_green_tint() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Near-white neutral: lum ≈ 0.9, above the highlight tint threshold (0.6).
        let img = make_solid_image(2, 2, 62000, 62000, 62000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fade_1970s",
                values: vec![1.0],
            }],
        );

        // The highlight tint target (0.88, 0.92, 0.72) has G > B.
        // After blending, G should be higher than B in the output.
        for pixel in out.pixels() {
            assert!(
                pixel[1] > pixel[2],
                "highlights should show yellow-green tint (G > B): G={} B={}",
                pixel[1],
                pixel[2]
            );
        }
    }

    /// Saturation should be reduced at full strength: a saturated input should have
    /// its channels drawn closer together (toward grey) compared to identity.
    #[test]
    fn test_fade_1970s_saturation_reduced() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Saturated red-heavy input with strong channel separation.
        let img = make_solid_image(2, 2, 60000, 10000, 10000);
        let out_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fade_1970s",
                values: vec![1.0],
            }],
        );
        let out_identity = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fade_1970s",
                values: vec![0.0],
            }],
        );

        // Channel spread (max - min) should be smaller at full strength.
        let spread_effect = |px: &image::Rgba<u16>| {
            let vals = [px[0], px[1], px[2]];
            *vals.iter().max().unwrap() as i32 - *vals.iter().min().unwrap() as i32
        };

        for (e, i) in out_effect.pixels().zip(out_identity.pixels()) {
            assert!(
                spread_effect(e) < spread_effect(i),
                "channel spread should decrease (desaturation): effect={} identity={}",
                spread_effect(e),
                spread_effect(i)
            );
        }
    }

    /// Alpha channel must not be modified regardless of strength.
    #[test]
    fn test_fade_1970s_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fade_1970s",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_fade_1970s_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 15000, 10000, 25000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fade_1970s",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "fade_1970s",
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
