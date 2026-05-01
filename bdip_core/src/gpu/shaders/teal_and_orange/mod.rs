use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TealAndOrangeParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for TealAndOrangeParams {
    const ID: &'static str = "teal_and_orange";
    const DISPLAY_NAME: &'static str = "Teal & Orange";
    const DESCRIPTION: &'static str =
        "Classic cinematic color grade: pushes shadows toward teal and highlights toward orange.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // Identity: no color shift applied.
        description: "Blend strength of the teal/orange grade. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "teal_and_orange",
        wgsl_source: include_str!("teal_and_orange.wgsl"),
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
    TealAndOrangeParams,
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
    fn test_teal_and_orange_registry_entry_exists() {
        assert!(registry_by_id("teal_and_orange").is_some());
    }

    #[test]
    fn test_teal_and_orange_registry_metadata() {
        let reg = registry_by_id("teal_and_orange").unwrap();
        assert_eq!(reg.meta.display_name, "Teal & Orange");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend strength of the teal/orange grade. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_teal_and_orange_passes_count() {
        let reg = registry_by_id("teal_and_orange").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_teal_and_orange_make_uniform_known_value() {
        let reg = registry_by_id("teal_and_orange").unwrap();
        let bytes = (reg.make_uniform)(&[0.7]);
        let expected = bytemuck::bytes_of(&TealAndOrangeParams {
            strength: 0.7,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_teal_and_orange_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 8000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "teal_and_orange",
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
    fn test_teal_and_orange_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 30000, 20000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "teal_and_orange",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// Dark (shadow) pixels should gain blue-green (teal) character when strength > 0.
    /// For a very dark input, the teal target (0, 0.25, 0.25 linear) means B and G
    /// should increase while R should stay very low.
    #[test]
    fn test_teal_and_orange_shadows_shift_toward_teal() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Near-black input: lum ≈ 0, maximum shadow weight.
        let img = make_solid_image(2, 2, 800, 800, 800);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "teal_and_orange",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            // Teal: G and B should exceed R for dark inputs pushed toward teal.
            assert!(
                pixel[1] > pixel[0],
                "G should exceed R in teal-shifted shadow: R={} G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                pixel[2] > pixel[0],
                "B should exceed R in teal-shifted shadow: R={} B={}",
                pixel[0],
                pixel[2]
            );
        }
    }

    /// Bright (highlight) pixels should gain orange character when strength > 0.
    /// For a near-white input, the orange target (0.37, 0.18, 0.0 linear) means
    /// R should be boosted relative to B.
    #[test]
    fn test_teal_and_orange_highlights_shift_toward_orange() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Near-white, neutral grey: lum ≈ 1, maximum highlight weight.
        let img = make_solid_image(2, 2, 62000, 62000, 62000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "teal_and_orange",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            // Orange: R should exceed B for bright inputs pushed toward orange.
            assert!(
                pixel[0] > pixel[2],
                "R should exceed B in orange-shifted highlight: R={} B={}",
                pixel[0],
                pixel[2]
            );
        }
    }

    /// Mid-grey (lum ≈ 0.5) sits at the boundary between shadow and highlight
    /// weight functions (both evaluate to 0 there), so the output should be
    /// nearly identical to the input regardless of strength.
    #[test]
    fn test_teal_and_orange_midtones_minimally_affected() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% grey: lum ≈ 0.214 linear (sRGB 0.5 → ~0.214 linear via sRGB decode).
        // The crossover of both smoothstep functions is at lum=0.5, so lum=0.214
        // still has moderate shadow weight — use a brighter mid-grey closer to
        // linear 0.5 to sit near the true crossover.
        // 55000/65535 ≈ 0.839 sRGB → ~0.668 linear — slightly above the 0.5 crossover.
        // Both weights are near zero here, so the output should be close to input.
        let img = make_solid_image(2, 2, 50000, 50000, 50000);
        let out_with = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "teal_and_orange",
                values: vec![1.0],
            }],
        );
        let out_without = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "teal_and_orange",
                values: vec![0.0],
            }],
        );

        for (a, b) in out_with.pixels().zip(out_without.pixels()) {
            // Mid-grey sits at the shadow/highlight boundary where both weights
            // are near zero, so the effect is substantially attenuated but not
            // necessarily zero. Allow a tolerance of 10% of max range (6553 u16).
            assert!(
                (a[0] as i32 - b[0] as i32).abs() <= 6553,
                "R: mid-grey should be only slightly affected: {} vs {}",
                a[0],
                b[0]
            );
            assert!(
                (a[1] as i32 - b[1] as i32).abs() <= 6553,
                "G: mid-grey should be only slightly affected: {} vs {}",
                a[1],
                b[1]
            );
            assert!(
                (a[2] as i32 - b[2] as i32).abs() <= 6553,
                "B: mid-grey should be only slightly affected: {} vs {}",
                a[2],
                b[2]
            );
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_teal_and_orange_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 15000, 10000, 5000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "teal_and_orange",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "teal_and_orange",
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
