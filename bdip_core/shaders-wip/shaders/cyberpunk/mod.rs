use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CyberpunkParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for CyberpunkParams {
    const ID: &'static str = "cyberpunk";
    const DISPLAY_NAME: &'static str = "Cyberpunk";
    const DESCRIPTION: &'static str = "Neon-lit color grade: boosts cyans and magentas, deepens shadows, \
         adds a teal-to-orange split tone, and pushes neon saturation.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // Identity: no color grade applied.
        description: "Blend strength of the cyberpunk grade. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "cyberpunk",
        wgsl_source: include_str!("cyberpunk.wgsl"),
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
    CyberpunkParams,
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
    fn test_cyberpunk_registry_entry_exists() {
        assert!(registry_by_id("cyberpunk").is_some());
    }

    #[test]
    fn test_cyberpunk_registry_metadata() {
        let reg = registry_by_id("cyberpunk").unwrap();
        assert_eq!(reg.meta.display_name, "Cyberpunk");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend strength of the cyberpunk grade. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_cyberpunk_passes_count() {
        let reg = registry_by_id("cyberpunk").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_cyberpunk_make_uniform_known_value() {
        let reg = registry_by_id("cyberpunk").unwrap();
        let bytes = (reg.make_uniform)(&[0.6]);
        let expected = bytemuck::bytes_of(&CyberpunkParams {
            strength: 0.6,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_cyberpunk_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 8000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cyberpunk",
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
    fn test_cyberpunk_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 30000, 20000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cyberpunk",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// At full strength, a highlight-range pure-red input should gain blue channel
    /// value. In the highlight zone the orange split slightly suppresses blue, but
    /// the cm_b coefficient (+14%) and neon saturation boost together should push
    /// the B channel higher than for a zero-strength run on the same input.
    /// We test with a bright red pixel so the blue channel starts near zero and
    /// any positive blue contribution is unambiguous.
    #[test]
    fn test_cyberpunk_full_strength_boosts_blue_on_bright_red() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Bright red: high R, no G or B. At full strength the cyberpunk grade
        // introduces blue via the cm_b term and teal split for the shadow-region
        // components that survive the highlight calculation.
        let img = make_solid_image(2, 2, 55000, 0, 0);
        let graded = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cyberpunk",
                values: vec![1.0],
            }],
        );
        let identity = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cyberpunk",
                values: vec![0.0],
            }],
        );

        for (g, id) in graded.pixels().zip(identity.pixels()) {
            assert!(
                g[2] > id[2],
                "B should be boosted on bright-red input at full strength: graded={} identity={}",
                g[2],
                id[2]
            );
        }
    }

    /// At full strength, a mid-tone neutral grey should produce lower luminance
    /// than the identity run, because the shadow-deepening power curve (exponent>1)
    /// pulls mid-tone channels downward before the split-tone blend is applied.
    /// We measure luminance rather than a single channel to account for the
    /// channel-specific effects of the cyan/magenta matrix and split tone.
    #[test]
    fn test_cyberpunk_full_strength_darkens_midtone_luminance() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Mid-grey: ~50% sRGB → ~0.214 linear. The shadow curve exponent=1.6
        // reduces this to ~0.214^1.6 ≈ 0.088 linear before the split-tone blend.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let graded = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cyberpunk",
                values: vec![1.0],
            }],
        );
        let identity = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cyberpunk",
                values: vec![0.0],
            }],
        );

        // Compare approximate luminance (Rec. 709 coefficients on u16 values).
        let lum = |p: &image::Rgba<u16>| {
            0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64
        };

        for (g, id) in graded.pixels().zip(identity.pixels()) {
            assert!(
                lum(g) < lum(id),
                "graded luminance should be lower than identity for mid-grey: \
                 graded={:.0} identity={:.0}",
                lum(g),
                lum(id)
            );
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_cyberpunk_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 15000, 10000, 5000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cyberpunk",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "cyberpunk",
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
