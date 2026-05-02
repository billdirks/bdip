use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Halftone Dots shader.
///
/// Two meaningful fields plus two padding floats to reach the 16-byte WebGPU
/// uniform alignment requirement.
///
/// # Identity design
///
/// Halftone always replaces continuous tone with a binary black/white dot pattern,
/// so no non-trivial parameter combination produces an identity transformation.
/// The `strength` blend factor defaulting to `0.0` passes the source image
/// through unchanged regardless of `frequency`, matching the blend-based identity
/// pattern used by Pointillism and Pencil Sketch.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HalftoneDotParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full halftone effect.
    pub strength: f32,
    /// Dot grid frequency in cycles per pixel. Higher values produce a finer,
    /// denser grid. Range [0.01, 0.5].
    pub frequency: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for HalftoneDotParams {
    const ID: &'static str = "halftone_dots";
    const DISPLAY_NAME: &'static str = "Halftone Dots";
    const DESCRIPTION: &'static str = "Simulates the halftone printing process using a sine-wave grid mask: bright \
         areas produce small dots while dark areas produce large dots.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full halftone \
                          effect (1.0). The identity value is 0.0.",
        },
        SliderDef {
            name: "Frequency",
            min: 0.01,
            max: 0.5,
            default: 0.1,
            description: "Dot grid frequency in cycles per pixel. Higher values produce \
                          a finer, denser halftone grid.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "halftone_dots",
        wgsl_source: include_str!("halftone_dots.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            frequency: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    HalftoneDotParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_halftone_dots_registry_entry_exists() {
        assert!(registry_by_id("halftone_dots").is_some());
    }

    #[test]
    fn test_halftone_dots_registry_metadata() {
        let reg = registry_by_id("halftone_dots").unwrap();
        assert_eq!(reg.meta.display_name, "Halftone Dots");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full halftone \
                                  effect (1.0). The identity value is 0.0.",
                },
                SliderDef {
                    name: "Frequency",
                    min: 0.01,
                    max: 0.5,
                    default: 0.1,
                    description: "Dot grid frequency in cycles per pixel. Higher values produce \
                                  a finer, denser halftone grid.",
                },
            ])
        );
    }

    #[test]
    fn test_halftone_dots_make_uniform_known_value() {
        let reg = registry_by_id("halftone_dots").unwrap();
        let bytes = (reg.make_uniform)(&[0.8, 0.15]);
        let expected = bytemuck::bytes_of(&HalftoneDotParams {
            strength: 0.8,
            frequency: 0.15,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 the blend reduces to the source image regardless of frequency.
    #[test]
    fn test_halftone_dots_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "halftone_dots",
                values: vec![0.0, 0.1],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 20000).abs() <= 64,
                "G: expected ~20000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 50000).abs() <= 64,
                "B: expected ~50000, got {}",
                pixel[2]
            );
        }
    }

    /// Alpha must pass through unchanged regardless of strength or frequency.
    #[test]
    fn test_halftone_dots_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "halftone_dots",
                values: vec![1.0, 0.1],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// At full strength (1.0) the output must be purely black or white pixels,
    /// because the sine-grid threshold produces a binary black/white halftone.
    #[test]
    fn test_halftone_dots_full_strength_produces_binary_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% gray: some pixels should be black, some white at the halftone threshold.
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "halftone_dots",
                values: vec![1.0, 0.1],
            }],
        );

        // Every output pixel must be near-black (< 1000) or near-white (> 64000).
        for pixel in out.pixels() {
            let r = pixel[0];
            assert!(
                !(1000..=64000).contains(&r),
                "expected binary black or white, got R={}",
                r
            );
        }
    }

    /// A pure white input at full strength must produce an all-white output, because
    /// luminance=1.0 means "mostly white" in halftone printing (smallest dots).
    #[test]
    fn test_halftone_dots_white_input_produces_white_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(32, 32, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "halftone_dots",
                values: vec![1.0, 0.1],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[0] > 64000,
                "white input: expected near-white output, got R={}",
                pixel[0]
            );
        }
    }

    /// A pure black input at full strength must produce an all-black output, because
    /// luminance=0.0 means "all black" in halftone printing (largest dots fill the cell).
    #[test]
    fn test_halftone_dots_black_input_produces_black_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(32, 32, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "halftone_dots",
                values: vec![1.0, 0.1],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[0] < 1000,
                "black input: expected near-black output, got R={}",
                pixel[0]
            );
        }
    }

    /// Different frequency values must produce different output on a non-uniform image.
    #[test]
    fn test_halftone_dots_different_frequencies_produce_different_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Gradient image so the halftone grid falls differently at different scales.
        let mut img = crate::Rgba16Image::new(64, 64);
        for y in 0..64u32 {
            for x in 0..64u32 {
                let v = (x * 1000) as u16;
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_coarse = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "halftone_dots",
                values: vec![1.0, 0.05],
            }],
        );
        let out_fine = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "halftone_dots",
                values: vec![1.0, 0.2],
            }],
        );

        let any_different = out_coarse
            .pixels()
            .zip(out_fine.pixels())
            .any(|(c, f)| (c[0] as i32 - f[0] as i32).abs() > 64);
        assert!(
            any_different,
            "frequency=0.05 and frequency=0.2 must produce different outputs on a gradient image"
        );
    }

    /// Chaining halftone_dots with brightness must not panic and must preserve alpha.
    #[test]
    fn test_halftone_dots_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(16, 16, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "halftone_dots",
                    values: vec![0.5, 0.1],
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
