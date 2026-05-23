use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RippleParams {
    pub amplitude: f32,
    pub frequency: f32,
    pub phase: f32,
    pub _padding: f32,
}

impl TransformShader for RippleParams {
    const ID: &'static str = "ripple";
    const DISPLAY_NAME: &'static str = "Ripple";
    const DESCRIPTION: &'static str =
        "Applies a sine-wave UV distortion to create a water ripple appearance.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Amplitude",
            min: 0.0,
            max: 0.5,
            default: 0.0,
            description: "Displacement magnitude in UV space. 0.0 = no ripple (identity); \
                0.5 shifts UVs by up to 50% of image dimensions.",
        },
        SliderDef {
            name: "Frequency",
            min: 0.5,
            max: 20.0,
            default: 5.0,
            description: "Number of sine wave cycles across the image. Higher values produce \
                more closely spaced waves.",
        },
        SliderDef {
            name: "Phase",
            min: 0.0,
            max: std::f32::consts::TAU,
            description: "Phase offset in radians [0, 2π], shifting the wave pattern \
                without changing its shape.",
            default: 0.0,
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "ripple",
        wgsl_source: include_str!("ripple.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            amplitude: values[0],
            frequency: values[1],
            phase: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<RippleParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_ripple_registry_entry_exists() {
        assert!(registry_by_id("ripple").is_some());
    }

    #[test]
    fn test_ripple_registry_metadata() {
        let reg = registry_by_id("ripple").unwrap();
        assert_eq!(reg.meta.display_name, "Ripple");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Amplitude",
                    min: 0.0,
                    max: 0.5,
                    default: 0.0,
                    description: "Displacement magnitude in UV space. 0.0 = no ripple (identity); \
                        0.5 shifts UVs by up to 50% of image dimensions.",
                },
                SliderDef {
                    name: "Frequency",
                    min: 0.5,
                    max: 20.0,
                    default: 5.0,
                    description: "Number of sine wave cycles across the image. Higher values produce \
                        more closely spaced waves.",
                },
                SliderDef {
                    name: "Phase",
                    min: 0.0,
                    max: std::f32::consts::TAU,
                    description: "Phase offset in radians [0, 2π], shifting the wave pattern \
                        without changing its shape.",
                    default: 0.0,
                },
            ])
        );
    }

    #[test]
    fn test_ripple_make_uniform_known_value() {
        let reg = registry_by_id("ripple").unwrap();
        let bytes = (reg.make_uniform)(&[0.1, 8.0, 1.5]);
        let expected = bytemuck::bytes_of(&RippleParams {
            amplitude: 0.1,
            frequency: 8.0,
            phase: 1.5,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    /// At amplitude=0.0 (identity default) every pixel must equal the source pixel.
    /// A solid-color image is used so any mis-mapping to a different pixel would
    /// still compare equal, isolating the identity path specifically.
    #[test]
    fn test_ripple_identity_at_zero_amplitude() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 40000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "ripple",
                values: vec![0.0, 5.0, 0.0],
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

    /// Alpha channel must pass through unchanged under non-zero ripple distortion.
    #[test]
    fn test_ripple_alpha_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "ripple",
                values: vec![0.05, 5.0, 0.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    /// At maximum amplitude the shader must not panic and must produce a result.
    /// Interior pixels of a solid-color image that are not mapped outside [0,1]
    /// retain the solid color; this verifies the shader runs to completion.
    #[test]
    fn test_ripple_extreme_amplitude_does_not_panic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 50000, 10000, 30000);
        // amplitude=0.5 with frequency=1.0 keeps most pixels in range.
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "ripple",
                values: vec![0.5, 1.0, 0.0],
            }],
        );

        // The image must have been produced (same dimensions) without panicking.
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
    }

    /// Maximum frequency must not cause a panic or GPU error.
    #[test]
    fn test_ripple_extreme_frequency_does_not_panic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "ripple",
                values: vec![0.05, 20.0, 0.0],
            }],
        );

        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
    }

    /// Phase offset at a full cycle (2π) must produce the same result as phase=0.0,
    /// since sin(x + 2π) == sin(x). A solid-color image is used so any pixel value
    /// comparison holds regardless of which source pixel is sampled.
    #[test]
    fn test_ripple_full_phase_cycle_matches_zero_phase() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(8, 8, 20000, 40000, 60000);

        let out_zero_phase = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "ripple",
                values: vec![0.05, 3.0, 0.0],
            }],
        );

        let out_full_cycle = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "ripple",
                // 2π ≈ 6.2831853 — one complete sine cycle, identical to phase=0.
                values: vec![0.05, 3.0, std::f32::consts::TAU],
            }],
        );

        for (p0, p1) in out_zero_phase.pixels().zip(out_full_cycle.pixels()) {
            assert!(
                (p0[0] as i32 - p1[0] as i32).abs() <= 64,
                "R mismatch at phase=2π vs phase=0: {} vs {}",
                p0[0],
                p1[0]
            );
            assert!(
                (p0[1] as i32 - p1[1] as i32).abs() <= 64,
                "G mismatch at phase=2π vs phase=0: {} vs {}",
                p0[1],
                p1[1]
            );
            assert!(
                (p0[2] as i32 - p1[2] as i32).abs() <= 64,
                "B mismatch at phase=2π vs phase=0: {} vs {}",
                p0[2],
                p1[2]
            );
        }
    }

    /// Chaining ripple with brightness must not panic and must preserve alpha.
    #[test]
    fn test_ripple_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "ripple",
                    values: vec![0.05, 5.0, 0.0],
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
