use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DoubleExposureParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for DoubleExposureParams {
    const ID: &'static str = "double_exposure";
    const DISPLAY_NAME: &'static str = "Double Exposure";
    const DESCRIPTION: &'static str = "Simulates the classic film technique of exposing the same frame twice by blending \
         a procedurally derived ghost image — built from the source via per-channel inversion \
         and a 3×3 blur approximation — using Screen blend mode.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // Identity: strength=0 → ghost contributes nothing → pure pass-through.
        description: "Blend intensity of the double-exposure ghost. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "double_exposure",
        wgsl_source: include_str!("double_exposure.wgsl"),
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
    DoubleExposureParams,
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
    fn test_double_exposure_registry_entry_exists() {
        assert!(registry_by_id("double_exposure").is_some());
    }

    #[test]
    fn test_double_exposure_registry_metadata() {
        let reg = registry_by_id("double_exposure").unwrap();
        assert_eq!(reg.meta.display_name, "Double Exposure");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend intensity of the double-exposure ghost. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_double_exposure_passes_count() {
        let reg = registry_by_id("double_exposure").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_double_exposure_make_uniform_known_value() {
        let reg = registry_by_id("double_exposure").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&DoubleExposureParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_double_exposure_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 15000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "double_exposure",
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
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// At full strength the screen blend must produce output brighter than (or equal to)
    /// the original, because screen can only add light, never subtract it.
    #[test]
    fn test_double_exposure_full_strength_does_not_darken() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Mid-gray input: enough signal for the ghost to be visible.
        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "double_exposure",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            // Screen(a, b) >= a always, so every channel must be >= the original.
            assert!(
                pixel[0] >= 20000 - 64,
                "R should not darken: original=20000, got={}",
                pixel[0]
            );
            assert!(
                pixel[1] >= 20000 - 64,
                "G should not darken: original=20000, got={}",
                pixel[1]
            );
            assert!(
                pixel[2] >= 20000 - 64,
                "B should not darken: original=20000, got={}",
                pixel[2]
            );
        }
    }

    /// A very dark image with full strength must produce visible output due to
    /// the luminance-inverted ghost (dark → bright ghost via inversion).
    #[test]
    fn test_double_exposure_full_strength_brightens_dark_image() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Very dark input so the ghost (which inverts luminance → bright) is dominant.
        let img = make_solid_image(4, 4, 500, 500, 500);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "double_exposure",
                values: vec![1.0],
            }],
        );

        let any_brighter = out
            .pixels()
            .any(|p| p[0] > 10000 || p[1] > 10000 || p[2] > 10000);
        assert!(
            any_brighter,
            "Full-strength double exposure must produce a visible ghost on a dark image"
        );
    }

    /// A fully white (maximum) image with full strength should produce maximum output
    /// because screen(1, anything) = 1.
    #[test]
    fn test_double_exposure_full_strength_white_stays_white() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "double_exposure",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            // White input: screen(1, b) = 1 for any b, so output should stay near 65535.
            assert!(
                pixel[0] >= 65000,
                "R should remain near-white: {}",
                pixel[0]
            );
            assert!(
                pixel[1] >= 65000,
                "G should remain near-white: {}",
                pixel[1]
            );
            assert!(
                pixel[2] >= 65000,
                "B should remain near-white: {}",
                pixel[2]
            );
        }
    }

    /// Alpha channel must not be modified regardless of strength.
    #[test]
    fn test_double_exposure_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "double_exposure",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_double_exposure_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 15000, 10000, 25000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "double_exposure",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "double_exposure",
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
