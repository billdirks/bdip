use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MoodyBlueParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for MoodyBlueParams {
    const ID: &'static str = "moody_blue";
    const DISPLAY_NAME: &'static str = "Moody Blue";
    const DESCRIPTION: &'static str =
        "Tints shadows with a cool blue tone while leaving highlights relatively neutral.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // Identity: no blue tint applied.
        description: "Intensity of the blue shadow tint. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "moody_blue",
        wgsl_source: include_str!("moody_blue.wgsl"),
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
    MoodyBlueParams,
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
    fn test_moody_blue_registry_entry_exists() {
        assert!(registry_by_id("moody_blue").is_some());
    }

    #[test]
    fn test_moody_blue_registry_metadata() {
        let reg = registry_by_id("moody_blue").unwrap();
        assert_eq!(reg.meta.display_name, "Moody Blue");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Intensity of the blue shadow tint. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_moody_blue_passes_count() {
        let reg = registry_by_id("moody_blue").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_moody_blue_make_uniform_known_value() {
        let reg = registry_by_id("moody_blue").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&MoodyBlueParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_moody_blue_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "moody_blue",
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

    /// Dark (shadow) pixels should gain a blue tint when strength > 0.
    /// For a near-black input the blue target (0.02, 0.05, 0.18 linear) dominates,
    /// so B should clearly exceed R.
    #[test]
    fn test_moody_blue_shadows_shift_toward_blue() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Near-black input: lum ≈ 0, maximum shadow weight.
        let img = make_solid_image(2, 2, 800, 800, 800);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "moody_blue",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            // Blue tint: B should exceed R for dark inputs.
            assert!(
                pixel[2] > pixel[0],
                "B should exceed R in blue-tinted shadow: R={} B={}",
                pixel[0],
                pixel[2]
            );
        }
    }

    /// Bright (highlight) pixels have near-zero shadow weight, so they should
    /// remain very close to the original values even at full strength.
    #[test]
    fn test_moody_blue_highlights_minimally_affected() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Near-white neutral grey: lum ≈ 0.9 linear, shadow_w ≈ 0.
        let img = make_solid_image(2, 2, 62000, 62000, 62000);
        let out_with = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "moody_blue",
                values: vec![1.0],
            }],
        );
        let out_without = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "moody_blue",
                values: vec![0.0],
            }],
        );

        // Highlights should be nearly unchanged (within 10% of u16 range = 6553).
        for (a, b) in out_with.pixels().zip(out_without.pixels()) {
            assert!(
                (a[0] as i32 - b[0] as i32).abs() <= 6553,
                "R: highlights changed too much: {} vs {}",
                a[0],
                b[0]
            );
            assert!(
                (a[1] as i32 - b[1] as i32).abs() <= 6553,
                "G: highlights changed too much: {} vs {}",
                a[1],
                b[1]
            );
            assert!(
                (a[2] as i32 - b[2] as i32).abs() <= 6553,
                "B: highlights changed too much: {} vs {}",
                a[2],
                b[2]
            );
        }
    }

    /// Alpha channel must not be modified regardless of strength.
    #[test]
    fn test_moody_blue_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "moody_blue",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_moody_blue_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 10000, 8000, 15000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "moody_blue",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "moody_blue",
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
