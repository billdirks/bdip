use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KodachromeParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for KodachromeParams {
    const ID: &'static str = "kodachrome";
    const DISPLAY_NAME: &'static str = "Kodachrome";
    const DESCRIPTION: &'static str = "Simulates the iconic Kodachrome film stock with deep saturated reds, \
         rich blues, slightly muted greens, and warm shadows.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // Identity: no color shift applied.
        description: "Blend strength of the Kodachrome grade. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "kodachrome",
        wgsl_source: include_str!("kodachrome.wgsl"),
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
    KodachromeParams,
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
    fn test_kodachrome_registry_entry_exists() {
        assert!(registry_by_id("kodachrome").is_some());
    }

    #[test]
    fn test_kodachrome_registry_metadata() {
        let reg = registry_by_id("kodachrome").unwrap();
        assert_eq!(reg.meta.display_name, "Kodachrome");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend strength of the Kodachrome grade. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_kodachrome_passes_count() {
        let reg = registry_by_id("kodachrome").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_kodachrome_make_uniform_known_value() {
        let reg = registry_by_id("kodachrome").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&KodachromeParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_kodachrome_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 8000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kodachrome",
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
    fn test_kodachrome_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 30000, 20000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kodachrome",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// At full strength, the red channel should be boosted for a neutral grey input.
    /// The matrix row for red is (1.20, 0.05, -0.05): for equal r=g=b, net gain = 1.2r.
    /// Red output should exceed the identity output.
    #[test]
    fn test_kodachrome_red_boosted_for_neutral_grey() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Neutral grey: equal R, G, B in linear space.
        // 32767/65535 ≈ 0.500 sRGB → ~0.214 linear.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kodachrome",
                values: vec![1.0],
            }],
        );
        let out_zero = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kodachrome",
                values: vec![0.0],
            }],
        );

        for (graded, original) in out_full.pixels().zip(out_zero.pixels()) {
            assert!(
                graded[0] > original[0],
                "Red should be boosted for neutral grey at full strength: graded={} original={}",
                graded[0],
                original[0]
            );
        }
    }

    /// At full strength, green should be desaturated for a neutral grey input.
    /// The matrix row for green is (-0.10, 0.90, 0.00): for equal r=g=b, net = 0.8g.
    /// Green output should be lower than the identity output.
    #[test]
    fn test_kodachrome_green_desaturated_for_neutral_grey() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kodachrome",
                values: vec![1.0],
            }],
        );
        let out_zero = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kodachrome",
                values: vec![0.0],
            }],
        );

        for (graded, original) in out_full.pixels().zip(out_zero.pixels()) {
            assert!(
                graded[1] < original[1],
                "Green should be desaturated for neutral grey at full strength: \
                 graded={} original={}",
                graded[1],
                original[1]
            );
        }
    }

    /// At full strength, blue should be boosted for a neutral grey input.
    /// The matrix row for blue is (0.05, -0.10, 1.15): for equal r=g=b, net = 1.10b.
    /// Blue output should exceed the identity output.
    #[test]
    fn test_kodachrome_blue_boosted_for_neutral_grey() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kodachrome",
                values: vec![1.0],
            }],
        );
        let out_zero = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kodachrome",
                values: vec![0.0],
            }],
        );

        for (graded, original) in out_full.pixels().zip(out_zero.pixels()) {
            assert!(
                graded[2] > original[2],
                "Blue should be boosted for neutral grey at full strength: graded={} original={}",
                graded[2],
                original[2]
            );
        }
    }

    /// A pure red input should produce amplified red output (matrix diagonal is 1.20).
    /// Mid-level red chosen so the 1.20× boost does not saturate above 1.0 in linear.
    #[test]
    fn test_kodachrome_pure_red_is_amplified() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32767/65535 ≈ 0.500 sRGB → ~0.214 linear; 1.20 × 0.214 = 0.257 — below 1.0.
        let img = make_solid_image(2, 2, 32767, 0, 0);
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kodachrome",
                values: vec![1.0],
            }],
        );
        let out_zero = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kodachrome",
                values: vec![0.0],
            }],
        );

        for (graded, identity) in out_full.pixels().zip(out_zero.pixels()) {
            assert!(
                graded[0] > identity[0],
                "Red channel should be amplified for pure-red input: graded={} identity={}",
                graded[0],
                identity[0]
            );
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the Kodachrome result.
    #[test]
    fn test_kodachrome_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 15000, 10000, 5000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kodachrome",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "kodachrome",
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
