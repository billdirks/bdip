use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Frost Ice effect.
///
/// The shader simulates frost-covered or icy glass in a single pass by
/// combining a radial vignette mask, domain-warped procedural noise, and a
/// cold blue-white tint.  At `strength = 0.0` the output is identical to the
/// source (identity), regardless of the other parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrostIceParams {
    /// How far the frost extends inward from the edges (0.0 = no frost, 1.0 = full frame).
    pub coverage: f32,
    /// Amplitude of the UV warp that simulates ice-crystal refraction (0.0 = none).
    pub distortion: f32,
    /// Overall effect opacity: 0.0 = identity (source unchanged), 1.0 = full frost.
    pub strength: f32,
    pub _padding: f32,
}

impl TransformShader for FrostIceParams {
    const ID: &'static str = "frost_ice";
    const DISPLAY_NAME: &'static str = "Frost Ice";
    const DESCRIPTION: &'static str = "Simulates a frost-covered or icy glass window using a radial vignette mask, \
         procedural domain-warped noise, and a cold blue tint.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Coverage",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "How far the frost extends inward from the frame edges. \
                          At 0.0 no frost is visible (identity). \
                          At 1.0 the frost reaches the center of the frame.",
        },
        SliderDef {
            name: "Distortion",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Amplitude of the UV warp applied inside the frost region, \
                          simulating how ice crystals refract and distort the view behind them. \
                          At 0.0 no distortion is applied.",
        },
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Overall opacity of the frost effect. At 0.0 the output is \
                          identical to the source (identity). At 1.0 the full frost \
                          colour and texture are applied.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "frost_ice",
        wgsl_source: include_str!("frost_ice.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            coverage: values[0],
            distortion: values[1],
            strength: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<FrostIceParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // Convenience identity transform: zero strength means no visible change.
    fn identity_transform() -> Transform {
        Transform {
            shader_id: "frost_ice",
            values: vec![0.5, 0.5, 0.0],
        }
    }

    #[test]
    fn test_frost_ice_registry_entry_exists() {
        assert!(registry_by_id("frost_ice").is_some());
    }

    #[test]
    fn test_frost_ice_registry_metadata() {
        let reg = registry_by_id("frost_ice").unwrap();
        assert_eq!(reg.meta.display_name, "Frost Ice");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Coverage",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "How far the frost extends inward from the frame edges. \
                                  At 0.0 no frost is visible (identity). \
                                  At 1.0 the frost reaches the center of the frame.",
                },
                SliderDef {
                    name: "Distortion",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Amplitude of the UV warp applied inside the frost region, \
                                  simulating how ice crystals refract and distort the view behind them. \
                                  At 0.0 no distortion is applied.",
                },
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Overall opacity of the frost effect. At 0.0 the output is \
                                  identical to the source (identity). At 1.0 the full frost \
                                  colour and texture are applied.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            1,
            "Frost Ice must have exactly 1 pass"
        );
    }

    #[test]
    fn test_frost_ice_make_uniform_known_value() {
        let reg = registry_by_id("frost_ice").unwrap();
        let bytes = (reg.make_uniform)(&[0.4, 0.6, 0.8]);
        let expected = bytemuck::bytes_of(&FrostIceParams {
            coverage: 0.4,
            distortion: 0.6,
            strength: 0.8,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_frost_ice_zero_strength_is_identity() {
        // When strength=0.0 the shader mixes 0% frost into the source,
        // producing an output identical to the input regardless of coverage
        // or distortion values.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 16384, 8192);
        let out = roundtrip(&mut renderer, &engine, &img, &[identity_transform()]);
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 16384).abs() <= 64,
                "G: expected ~16384, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 8192).abs() <= 64,
                "B: expected ~8192, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_frost_ice_default_values_are_identity() {
        // All three defaults are 0.0. With strength=0.0, no frost is applied.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 20000, 40000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "frost_ice",
                values: vec![0.0, 0.0, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 20000).abs() <= 64,
                "R: expected ~20000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 40000).abs() <= 64,
                "G: expected ~40000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 60000).abs() <= 64,
                "B: expected ~60000, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_frost_ice_alpha_preserved_at_identity() {
        // Alpha channel must pass through unchanged when strength=0.0.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(&mut renderer, &engine, &img, &[identity_transform()]);
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved when strength=0.0");
        }
    }

    #[test]
    fn test_frost_ice_alpha_preserved_at_full_strength() {
        // Alpha must be preserved even when the frost effect is fully applied.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "frost_ice",
                values: vec![1.0, 0.5, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be preserved at full frost strength"
            );
        }
    }

    #[test]
    fn test_frost_ice_full_coverage_shifts_toward_blue() {
        // With coverage=1.0 and strength=1.0 all pixels are inside the frost
        // region.  The frost colour is a blue-white (B > R), so the blue
        // channel of the output must exceed the red channel on a neutral-gray
        // input where source R ≈ source B.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "frost_ice",
                values: vec![1.0, 0.0, 1.0],
            }],
        );
        // The average blue channel must exceed the average red channel across
        // all pixels because the frost colour biases toward blue.
        let sum_r: i64 = out.pixels().map(|p| p[0] as i64).sum();
        let sum_b: i64 = out.pixels().map(|p| p[2] as i64).sum();
        assert!(
            sum_b > sum_r,
            "frost must shift output toward blue: sum_r={sum_r}, sum_b={sum_b}"
        );
    }

    #[test]
    fn test_frost_ice_higher_strength_lightens_output() {
        // On a dark image, increasing strength blends in a bright frost colour,
        // which should raise the mean brightness of the output.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 5000, 5000, 5000);

        let out_low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "frost_ice",
                values: vec![1.0, 0.0, 0.3],
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "frost_ice",
                values: vec![1.0, 0.0, 0.9],
            }],
        );

        let sum_low: i64 = out_low
            .pixels()
            .map(|p| p[0] as i64 + p[1] as i64 + p[2] as i64)
            .sum();
        let sum_high: i64 = out_high
            .pixels()
            .map(|p| p[0] as i64 + p[1] as i64 + p[2] as i64)
            .sum();
        assert!(
            sum_high > sum_low,
            "higher strength on a dark image must increase brightness: \
             sum_low={sum_low}, sum_high={sum_high}"
        );
    }

    #[test]
    fn test_frost_ice_zero_coverage_leaves_center_unchanged() {
        // With coverage=0.0 the frost reach is zero, so the frost_mask is 0
        // everywhere (edge_dist >= 0 for all pixels). The output must equal the
        // source regardless of strength or distortion.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 25000, 35000, 45000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "frost_ice",
                values: vec![0.0, 1.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 25000).abs() <= 64,
                "R: expected ~25000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 35000).abs() <= 64,
                "G: expected ~35000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 45000).abs() <= 64,
                "B: expected ~45000, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_frost_ice_chaining_with_brightness() {
        // Frost Ice chained after Brightness must not panic and must produce
        // correct output dimensions and valid alpha. Verifies the shader works
        // correctly in a multi-shader pipeline.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "brightness",
                    values: vec![0.1],
                },
                Transform {
                    shader_id: "frost_ice",
                    values: vec![0.5, 0.3, 0.6],
                },
            ],
        );
        assert_eq!(out.width(), 16);
        assert_eq!(out.height(), 16);
        for pixel in out.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be preserved through Brightness→Frost Ice"
            );
        }
    }
}
