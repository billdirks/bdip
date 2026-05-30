use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Vortex radial UV twist.
///
/// Unlike Swirl (which peaks at the image centre with a linear falloff),
/// Vortex applies the strongest rotation at a configurable ring distance using
/// a Gaussian-shaped radial envelope. The centre and the far edges therefore
/// receive less twist than the ring, producing a whirlpool/spinning-disk look.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VortexParams {
    /// Total rotation at the peak ring, in full turns (1.0 = 360°). 0.0 = identity.
    pub twist: f32,
    /// Distance from centre (in normalised half-diagonal units) at which the
    /// twist is strongest. Effective range (0.0, 1.5]. 0.0 collapses to identity.
    pub radius_scale: f32,
    /// Blend factor [0.0, 1.0] scaling the full twist envelope.
    /// 0.0 = identity; 1.0 = full effect.
    pub strength: f32,
    pub _padding: f32,
}

impl TransformShader for VortexParams {
    const ID: &'static str = "vortex";
    const DISPLAY_NAME: &'static str = "Vortex";
    const DESCRIPTION: &'static str = "Applies a radial twist that peaks at a ring distance from the centre, \
         creating a whirlpool appearance distinct from the centre-peaked Swirl effect.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Twist",
            min: -3.0,
            max: 3.0,
            default: 0.0,
            description: "Rotation at the peak ring in full turns (1.0 = 360°). \
                          0.0 = identity; positive = counter-clockwise; \
                          negative = clockwise.",
        },
        SliderDef {
            name: "Radius",
            min: 0.05,
            max: 1.5,
            default: 0.5,
            description: "Distance from the image centre (in normalised \
                          half-diagonal units) at which the twist is strongest.",
        },
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend factor: 0.0 = identity (no effect); \
                          1.0 = full twist envelope.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "vortex",
        wgsl_source: include_str!("vortex.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            twist: values[0],
            radius_scale: values[1],
            strength: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<VortexParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ── registry ─────────────────────────────────────────────────────────────

    #[test]
    fn test_vortex_registry_entry_exists() {
        assert!(registry_by_id("vortex").is_some());
    }

    #[test]
    fn test_vortex_registry_metadata() {
        let reg = registry_by_id("vortex").unwrap();
        assert_eq!(reg.meta.display_name, "Vortex");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Twist",
                    min: -3.0,
                    max: 3.0,
                    default: 0.0,
                    description: "Rotation at the peak ring in full turns (1.0 = 360°). \
                                  0.0 = identity; positive = counter-clockwise; \
                                  negative = clockwise.",
                },
                SliderDef {
                    name: "Radius",
                    min: 0.05,
                    max: 1.5,
                    default: 0.5,
                    description: "Distance from the image centre (in normalised \
                                  half-diagonal units) at which the twist is strongest.",
                },
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend factor: 0.0 = identity (no effect); \
                                  1.0 = full twist envelope.",
                },
            ])
        );
    }

    #[test]
    fn test_vortex_make_uniform_known_value() {
        let reg = registry_by_id("vortex").unwrap();
        let bytes = (reg.make_uniform)(&[1.5, 0.6, 0.8]);
        let expected = bytemuck::bytes_of(&VortexParams {
            twist: 1.5,
            radius_scale: 0.6,
            strength: 0.8,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip helpers ─────────────────────────────────────────────────

    fn create_test_image() -> crate::Rgba16Image {
        make_solid_image(4, 4, 20000, 40000, 60000)
    }

    // ── identity conditions ───────────────────────────────────────────────────

    /// When twist=0.0 (default) the shader must be a no-op regardless of other
    /// parameters, because the rotation angle is zero.
    #[test]
    fn test_vortex_identity_at_zero_twist() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = create_test_image();
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "vortex",
                values: vec![0.0, 0.5, 1.0],
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
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// When strength=0.0 the shader must be a no-op regardless of other parameters,
    /// because the blend factor zeroes the rotation envelope.
    #[test]
    fn test_vortex_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = create_test_image();
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "vortex",
                values: vec![2.0, 0.5, 0.0],
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
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// Default parameter values (twist=0, radius=0.5, strength=0) must produce
    /// a no-op, verifying the registered defaults are an identity transformation.
    #[test]
    fn test_vortex_default_values_are_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let reg = registry_by_id("vortex").unwrap();
        let defaults: Vec<f32> = match reg.meta.param {
            ParamKind::Sliders(sliders) => sliders.iter().map(|s| s.default).collect(),
            ParamKind::Toggle => vec![],
        };

        let img = create_test_image();
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "vortex",
                values: defaults,
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
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    // ── alpha preservation ───────────────────────────────────────────────────

    /// The alpha channel must pass through unchanged under any active vortex effect.
    #[test]
    fn test_vortex_alpha_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "vortex",
                values: vec![1.5, 0.5, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    // ── solid-image stability ────────────────────────────────────────────────

    /// A solid-colour image under any vortex setting must remain the same solid
    /// colour everywhere, because all source pixels are identical so any UV
    /// remapping still samples the same value.
    #[test]
    fn test_vortex_solid_image_unchanged_under_twist() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "vortex",
                values: vec![2.0, 0.5, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 30000).abs() <= 64,
                "R: expected ~30000, got {}",
                pixel[0]
            );
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    // ── edge / extreme parameters ────────────────────────────────────────────

    /// A negative twist (clockwise) must not panic and must preserve alpha for
    /// every output pixel, including any out-of-bounds fills.
    #[test]
    fn test_vortex_negative_twist_alpha_intact() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 40000, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "vortex",
                values: vec![-2.5, 0.5, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    /// Maximum twist and maximum strength must not panic.
    #[test]
    fn test_vortex_maximum_twist_does_not_panic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 10000, 20000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "vortex",
                values: vec![3.0, 1.5, 1.0],
            }],
        );

        // Only assert no crash + alpha; colour is distortion-dependent.
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 after maximum twist");
        }
    }

    /// A large radius_scale (ring beyond the image corners) must not panic and
    /// must preserve alpha. At this scale the Gaussian envelope is very wide,
    /// so every pixel may receive significant rotation.
    #[test]
    fn test_vortex_large_radius_scale_alpha_intact() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 50000, 25000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "vortex",
                values: vec![1.0, 1.5, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    // ── chaining ─────────────────────────────────────────────────────────────

    /// Chaining vortex with brightness must not panic and must preserve alpha
    /// across the full pipeline.
    #[test]
    fn test_vortex_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "vortex",
                    values: vec![1.0, 0.5, 1.0],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 after chaining");
        }
    }
}
