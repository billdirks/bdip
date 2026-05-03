use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Underwater multi-pass effect.
///
/// The shader runs two passes:
/// 1. Tint — applies a blue/teal color shift with slight desaturation.
/// 2. Caustic — overlays a procedurally generated caustic light pattern (additive).
///
/// Identity is achieved when `strength=0.0`. At zero strength the tint pass
/// outputs the source unchanged and the caustic overlay is scaled to zero,
/// making the combined effect a strict no-op.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UnderwaterParams {
    /// Controls the blue/teal tint intensity and blue channel shift. Range [0.0, 1.0].
    pub depth: f32,
    /// Brightness of the caustic light overlay. Range [0.0, 1.0].
    pub caustic_intensity: f32,
    /// Overall blend strength of the entire effect. At 0.0 the image is unchanged.
    pub strength: f32,
    pub _padding: f32,
}

impl TransformShader for UnderwaterParams {
    const ID: &'static str = "underwater";
    const DISPLAY_NAME: &'static str = "Underwater";
    const DESCRIPTION: &'static str = "Simulates a submerged underwater look with a blue/teal tint and \
         procedural caustic light patterns generated via sine interference.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Depth",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Controls the intensity of the blue/teal tint and desaturation. \
                          Higher values simulate deeper water with a stronger color shift.",
        },
        SliderDef {
            name: "Caustic Intensity",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Brightness of the caustic light pattern overlay. \
                          Higher values produce more prominent wavy light streaks.",
        },
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Overall blend strength of the underwater effect. \
                          At 0.0 the image is unchanged (identity).",
        },
    ]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "tint",
            wgsl_source: include_str!("underwater_tint.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("tinted"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "caustic",
            wgsl_source: include_str!("underwater_caustic.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("tinted")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            depth: values[0],
            caustic_intensity: values[1],
            strength: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    UnderwaterParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    fn identity_transform() -> Transform {
        Transform {
            shader_id: "underwater",
            values: vec![0.5, 0.5, 0.0],
        }
    }

    // ── Registry / metadata ──────────────────────────────────────────────────

    #[test]
    fn test_underwater_registry_entry_exists() {
        assert!(registry_by_id("underwater").is_some());
    }

    #[test]
    fn test_underwater_registry_metadata() {
        let reg = registry_by_id("underwater").unwrap();
        assert_eq!(reg.meta.display_name, "Underwater");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Depth",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Controls the intensity of the blue/teal tint and desaturation. \
                                  Higher values simulate deeper water with a stronger color shift.",
                },
                SliderDef {
                    name: "Caustic Intensity",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Brightness of the caustic light pattern overlay. \
                                  Higher values produce more prominent wavy light streaks.",
                },
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Overall blend strength of the underwater effect. \
                                  At 0.0 the image is unchanged (identity).",
                },
            ])
        );
    }

    #[test]
    fn test_underwater_passes_count() {
        let reg = registry_by_id("underwater").unwrap();
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "Underwater must have exactly 2 passes"
        );
    }

    #[test]
    fn test_underwater_make_uniform_known_value() {
        let reg = registry_by_id("underwater").unwrap();
        let bytes = (reg.make_uniform)(&[0.6, 0.4, 0.8]);
        let expected = bytemuck::bytes_of(&UnderwaterParams {
            depth: 0.6,
            caustic_intensity: 0.4,
            strength: 0.8,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// At strength=0.0 the final pass blends source and the tinted+caustic result
    /// with weight 0, so the output must match the source.
    #[test]
    fn test_underwater_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(&mut renderer, &engine, &img, &[identity_transform()]);
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 32767).abs() <= 64,
                "G: expected ~32767, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 32767).abs() <= 64,
                "B: expected ~32767, got {}",
                pixel[2]
            );
        }
    }

    /// At full strength and depth the output blue channel should exceed the red
    /// channel — the blue/teal tint must shift neutral grey toward blue.
    #[test]
    fn test_underwater_full_depth_shifts_neutral_gray_toward_blue() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Mid-gray: R = G = B = 32767.
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "underwater",
                values: vec![1.0, 0.0, 1.0],
            }],
        );
        let p = out.get_pixel(0, 0);
        assert!(
            p[2] > p[0],
            "full depth must push B above R on neutral gray: R={} B={}",
            p[0],
            p[2]
        );
    }

    /// Higher strength must result in a more pronounced blue shift. The blue-red
    /// difference at strength=1.0 should exceed that at strength=0.5.
    #[test]
    fn test_underwater_higher_strength_increases_blue_shift() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);

        let out_low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "underwater",
                values: vec![1.0, 0.0, 0.5],
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "underwater",
                values: vec![1.0, 0.0, 1.0],
            }],
        );

        let p_low = out_low.get_pixel(0, 0);
        let p_high = out_high.get_pixel(0, 0);
        let diff_low = p_low[2] as i32 - p_low[0] as i32;
        let diff_high = p_high[2] as i32 - p_high[0] as i32;
        assert!(
            diff_high > diff_low,
            "higher strength must produce stronger blue shift: \
             diff_low={diff_low}, diff_high={diff_high}"
        );
    }

    /// With caustic_intensity > 0 and strength > 0, the mean brightness of the
    /// output should exceed the tint-only version (caustics are additive).
    #[test]
    fn test_underwater_caustic_overlay_adds_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);

        let out_no_caustic = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "underwater",
                values: vec![0.5, 0.0, 1.0],
            }],
        );
        let out_caustic = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "underwater",
                values: vec![0.5, 1.0, 1.0],
            }],
        );

        let sum_no_caustic: i64 = out_no_caustic
            .pixels()
            .map(|p| p[0] as i64 + p[1] as i64 + p[2] as i64)
            .sum();
        let sum_caustic: i64 = out_caustic
            .pixels()
            .map(|p| p[0] as i64 + p[1] as i64 + p[2] as i64)
            .sum();
        assert!(
            sum_caustic > sum_no_caustic,
            "caustic overlay must add brightness: sum_no_caustic={sum_no_caustic}, \
             sum_caustic={sum_caustic}"
        );
    }

    /// Alpha must pass through both passes unchanged.
    #[test]
    fn test_underwater_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "underwater",
                values: vec![1.0, 1.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be preserved through all Underwater passes"
            );
        }
    }

    /// The caustic pass is position-dependent; different pixels should not all be
    /// identical when the effect is active. Check that spatial variation exists in
    /// the caustic layer by confirming not all pixels in the output are equal.
    #[test]
    fn test_underwater_caustic_produces_spatial_variation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a large enough canvas so caustic frequency produces variation.
        let img = make_solid_image(64, 64, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "underwater",
                values: vec![0.5, 1.0, 1.0],
            }],
        );
        let first = out.get_pixel(0, 0)[0];
        let all_same = out.pixels().all(|p| p[0] == first);
        assert!(
            !all_same,
            "caustic pattern must produce spatial variation across the image"
        );
    }

    /// Chaining Underwater after brightness must not panic and must preserve
    /// image dimensions and alpha (integration glue test).
    #[test]
    fn test_underwater_chained_with_brightness() {
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
                    shader_id: "underwater",
                    values: vec![0.5, 0.5, 0.5],
                },
            ],
        );
        assert_eq!(out.width(), 16);
        assert_eq!(out.height(), 16);
        for pixel in out.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be preserved through Brightness→Underwater"
            );
        }
    }
}
