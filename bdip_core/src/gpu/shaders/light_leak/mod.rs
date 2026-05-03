use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightLeakParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for LightLeakParams {
    const ID: &'static str = "light_leak";
    const DISPLAY_NAME: &'static str = "Light Leak";
    const DESCRIPTION: &'static str = "Simulates light bleeding into the film frame with warm procedural streaks \
         of orange, yellow, and red from corners and edges.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // Identity: no light leak applied.
        description: "Blend intensity of the light-leak effect. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "light_leak",
        wgsl_source: include_str!("light_leak.wgsl"),
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
    LightLeakParams,
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
    fn test_light_leak_registry_entry_exists() {
        assert!(registry_by_id("light_leak").is_some());
    }

    #[test]
    fn test_light_leak_registry_metadata() {
        let reg = registry_by_id("light_leak").unwrap();
        assert_eq!(reg.meta.display_name, "Light Leak");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend intensity of the light-leak effect. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_light_leak_passes_count() {
        let reg = registry_by_id("light_leak").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_light_leak_make_uniform_known_value() {
        let reg = registry_by_id("light_leak").unwrap();
        let bytes = (reg.make_uniform)(&[0.7]);
        let expected = bytemuck::bytes_of(&LightLeakParams {
            strength: 0.7,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_light_leak_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "light_leak",
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

    /// At full strength the output must be brighter than the input on a dark image,
    /// since the additive light-leak contribution is always non-negative.
    #[test]
    fn test_light_leak_full_strength_brightens_dark_image() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Very dark input so any additive light contribution is clearly visible.
        let img = make_solid_image(4, 4, 500, 500, 500);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "light_leak",
                values: vec![1.0],
            }],
        );

        // At least one pixel should be noticeably brighter than the original input.
        let any_brighter = out
            .pixels()
            .any(|p| p[0] > 1000 || p[1] > 1000 || p[2] > 1000);
        assert!(
            any_brighter,
            "Full-strength light leak must brighten a dark image"
        );
    }

    /// The effect has a warm color bias: at full strength, red should be the dominant
    /// leaked channel (warm oranges/yellows), not blue.
    #[test]
    fn test_light_leak_full_strength_warm_dominant_channel() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Near-black input to isolate the leaked light contribution from the original.
        let img = make_solid_image(4, 4, 200, 200, 200);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "light_leak",
                values: vec![1.0],
            }],
        );

        // Sum each channel's total output across all pixels.
        let (sum_r, sum_b) = out
            .pixels()
            .fold((0u64, 0u64), |(r, b), p| (r + p[0] as u64, b + p[2] as u64));

        assert!(
            sum_r > sum_b,
            "Red channel should exceed blue for a warm light leak: R_sum={sum_r} B_sum={sum_b}"
        );
    }

    /// Alpha channel must not be modified regardless of strength.
    #[test]
    fn test_light_leak_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "light_leak",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_light_leak_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 15000, 10000, 25000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "light_leak",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "light_leak",
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
