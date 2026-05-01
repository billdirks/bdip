use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TechnicolorParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for TechnicolorParams {
    const ID: &'static str = "technicolor";
    const DISPLAY_NAME: &'static str = "Technicolor";
    const DESCRIPTION: &'static str = "Simulates the classic Technicolor 3-strip dye-transfer process with \
         boosted reds and greens, desaturated blues, and warm cross-channel bleeding.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // Identity: no color shift applied.
        description: "Blend strength of the Technicolor grade. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "technicolor",
        wgsl_source: include_str!("technicolor.wgsl"),
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
    TechnicolorParams,
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
    fn test_technicolor_registry_entry_exists() {
        assert!(registry_by_id("technicolor").is_some());
    }

    #[test]
    fn test_technicolor_registry_metadata() {
        let reg = registry_by_id("technicolor").unwrap();
        assert_eq!(reg.meta.display_name, "Technicolor");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend strength of the Technicolor grade. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_technicolor_passes_count() {
        let reg = registry_by_id("technicolor").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_technicolor_make_uniform_known_value() {
        let reg = registry_by_id("technicolor").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&TechnicolorParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_technicolor_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 8000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "technicolor",
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
                (pixel[2] as i32 - 8000).abs() <= 64,
                "B mismatch: {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    /// Alpha channel must not be modified regardless of strength.
    #[test]
    fn test_technicolor_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 30000, 20000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "technicolor",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// At full strength, the red channel should be boosted relative to input for a
    /// neutral grey, reflecting the matrix's 1.3× red gain minus bleed losses.
    /// For a near-neutral grey input, the Technicolor matrix produces warmer output:
    /// r_out = 1.3r - 0.1g - 0.05b > r when r ≈ g ≈ b (net gain ≈ +0.15r).
    #[test]
    fn test_technicolor_red_boosted_for_neutral_grey() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Neutral grey: equal R, G, B in linear space.
        // 32767/65535 ≈ 0.500 sRGB → ~0.214 linear.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "technicolor",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            // Red boosted: r_out = 1.3r - 0.1g - 0.05b = 1.15r for equal channels.
            // Green boosted: g_out = -0.05r + 1.2g + 0.05b = 1.2g for equal channels.
            // Blue desaturated: b_out = 0.05r - 0.15g + 0.9b = 0.8b for equal channels.
            // So output order (descending) should be G > R > B.
            assert!(
                pixel[0] > pixel[2],
                "Red should exceed Blue for neutral grey at full strength: R={} B={}",
                pixel[0],
                pixel[2]
            );
            assert!(
                pixel[1] > pixel[2],
                "Green should exceed Blue for neutral grey at full strength: G={} B={}",
                pixel[1],
                pixel[2]
            );
        }
    }

    /// At full strength, blue is desaturated: the blue output for a neutral grey
    /// should be lower than the input blue due to the 0.9× blue diagonal coefficient
    /// and negative green contribution in the matrix.
    #[test]
    fn test_technicolor_blue_desaturated_for_neutral_grey() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "technicolor",
                values: vec![1.0],
            }],
        );
        let out_zero = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "technicolor",
                values: vec![0.0],
            }],
        );

        for (graded, original) in out_full.pixels().zip(out_zero.pixels()) {
            // b_out = 0.05r - 0.15g + 0.9b = 0.8b for equal channels → blue is lower.
            assert!(
                graded[2] < original[2],
                "Blue should be reduced for neutral grey at full strength: graded={} original={}",
                graded[2],
                original[2]
            );
        }
    }

    /// A pure red input (R=max, G=0, B=0) should produce a boosted red output
    /// since r_out = 1.3r for a red-only source.
    #[test]
    fn test_technicolor_pure_red_is_amplified() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure red in linear: use a mid-level value so the 1.3× boost doesn't saturate.
        // 32767/65535 ≈ 0.214 linear; 1.3 × 0.214 = 0.278 linear — safely below 1.0.
        let img = make_solid_image(2, 2, 32767, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "technicolor",
                values: vec![1.0],
            }],
        );
        let out_identity = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "technicolor",
                values: vec![0.0],
            }],
        );

        for (graded, identity) in out.pixels().zip(out_identity.pixels()) {
            assert!(
                graded[0] > identity[0],
                "Red channel should be amplified for pure-red input: graded={} identity={}",
                graded[0],
                identity[0]
            );
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_technicolor_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 15000, 10000, 5000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "technicolor",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "technicolor",
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
