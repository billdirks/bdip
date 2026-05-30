use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RainbowFlareParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for RainbowFlareParams {
    const ID: &'static str = "rainbow_flare";
    const DISPLAY_NAME: &'static str = "Rainbow Flare";
    const DESCRIPTION: &'static str = "Overlays an iridescent spectral rainbow as if light is diffracting through a prism \
         or lens coating, using polar coordinates and radial distance from the image centre.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // Identity: no spectral overlay applied.
        description: "Blend intensity of the rainbow-flare overlay. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "rainbow_flare",
        wgsl_source: include_str!("rainbow_flare.wgsl"),
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
    RainbowFlareParams,
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
    fn test_rainbow_flare_registry_entry_exists() {
        assert!(registry_by_id("rainbow_flare").is_some());
    }

    #[test]
    fn test_rainbow_flare_registry_metadata() {
        let reg = registry_by_id("rainbow_flare").unwrap();
        assert_eq!(reg.meta.display_name, "Rainbow Flare");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend intensity of the rainbow-flare overlay. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_rainbow_flare_passes_count() {
        let reg = registry_by_id("rainbow_flare").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_rainbow_flare_make_uniform_known_value() {
        let reg = registry_by_id("rainbow_flare").unwrap();
        let bytes = (reg.make_uniform)(&[0.7]);
        let expected = bytemuck::bytes_of(&RainbowFlareParams {
            strength: 0.7,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_rainbow_flare_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "rainbow_flare",
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

    /// At full strength the overlay must brighten a dark image, since the
    /// spectral contribution is always non-negative.
    #[test]
    fn test_rainbow_flare_full_strength_brightens_dark_image() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Very dark input so any additive spectral contribution is clearly visible.
        // Use a larger image so polar-coordinate variation produces non-zero flare
        // across multiple pixels at different radii and angles.
        let img = make_solid_image(8, 8, 200, 200, 200);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "rainbow_flare",
                values: vec![1.0],
            }],
        );

        let any_brighter = out.pixels().any(|p| p[0] > 500 || p[1] > 500 || p[2] > 500);
        assert!(
            any_brighter,
            "Full-strength rainbow flare must brighten a dark image"
        );
    }

    /// The overlay must produce colour variation across pixels (not a flat tint),
    /// since hue varies with both radius and angle.
    #[test]
    fn test_rainbow_flare_full_strength_produces_color_variation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Neutral grey input so any colour variation comes from the overlay alone.
        let img = make_solid_image(8, 8, 100, 100, 100);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "rainbow_flare",
                values: vec![1.0],
            }],
        );

        // Collect all unique (R, G, B) triples.
        let distinct: std::collections::HashSet<(u16, u16, u16)> =
            out.pixels().map(|p| (p[0], p[1], p[2])).collect();

        assert!(
            distinct.len() > 1,
            "Rainbow flare must produce colour variation across pixels; got {} distinct colours",
            distinct.len()
        );
    }

    /// Alpha channel must not be modified regardless of strength.
    #[test]
    fn test_rainbow_flare_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "rainbow_flare",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_rainbow_flare_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 15000, 10000, 25000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "rainbow_flare",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "rainbow_flare",
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
