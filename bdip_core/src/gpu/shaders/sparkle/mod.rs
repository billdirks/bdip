use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Sparkle bloom/glow effect.
///
/// The shader runs four passes:
/// 1. Threshold — isolate pixels above `threshold` luminance.
/// 2. Horizontal blur — Gaussian spread of the bright-pixel mask.
/// 3. Vertical blur — second half of the separable 2D Gaussian.
/// 4. Combine — additive blend of the spread glow onto the original.
///
/// At default values (`threshold=1.0`, `intensity=0.0`, `radius=0.5`) no glow
/// is produced: the threshold passes nothing and the intensity multiplies
/// any residual glow by zero, resulting in a strict identity transformation.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SparkleParams {
    pub threshold: f32, // luminance cutoff ∈ [0.0, 1.0]
    pub intensity: f32, // glow blend strength ∈ [0.0, 1.0]
    pub radius: f32,    // spread size ∈ [0.0, 1.0]
    pub _padding: f32,
}

impl TransformShader for SparkleParams {
    const ID: &'static str = "sparkle";
    const DISPLAY_NAME: &'static str = "Sparkle";
    const DESCRIPTION: &'static str = "Adds a bright glow to high-luminance pixels, simulating lens sparkle \
         or star-burst effects on bright light sources.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Threshold",
            min: 0.0,
            max: 1.0,
            default: 1.0,
            description: "Luminance cutoff above which pixels contribute to the glow. \
                          At 1.0 no pixels qualify and no glow is produced (identity). \
                          Lower values pull more of the image into the glow.",
        },
        SliderDef {
            name: "Intensity",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Strength of the glow blended back onto the original image. \
                          At 0.0 the glow contribution is zero (identity).",
        },
        SliderDef {
            name: "Radius",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Spread size of the glow as a fraction of the image's short \
                          axis. Larger values create a wider, softer bloom.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "threshold",
            wgsl_source: include_str!("sparkle_threshold.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("bright"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "blur_h",
            wgsl_source: include_str!("sparkle_blur_h.wgsl"),
            inputs: &[PassInput::Scratch("bright")],
            output: PassOutput::Scratch("glow_h"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "blur_v",
            wgsl_source: include_str!("sparkle_blur_v.wgsl"),
            inputs: &[PassInput::Scratch("glow_h")],
            output: PassOutput::Scratch("glow"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "combine",
            wgsl_source: include_str!("sparkle_combine.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("glow")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            threshold: values[0],
            intensity: values[1],
            radius: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<SparkleParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // Default parameter values for tests that need no visual effect.
    const IDENTITY_THRESHOLD: f32 = 1.0;
    const IDENTITY_INTENSITY: f32 = 0.0;
    const DEFAULT_RADIUS: f32 = 0.5;

    #[test]
    fn test_sparkle_registry_entry_exists() {
        assert!(registry_by_id("sparkle").is_some());
    }

    #[test]
    fn test_sparkle_registry_metadata() {
        let reg = registry_by_id("sparkle").unwrap();
        assert_eq!(reg.meta.display_name, "Sparkle");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Threshold",
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                    description: "Luminance cutoff above which pixels contribute to the glow. \
                                  At 1.0 no pixels qualify and no glow is produced (identity). \
                                  Lower values pull more of the image into the glow.",
                },
                SliderDef {
                    name: "Intensity",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Strength of the glow blended back onto the original image. \
                                  At 0.0 the glow contribution is zero (identity).",
                },
                SliderDef {
                    name: "Radius",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Spread size of the glow as a fraction of the image's short \
                                  axis. Larger values create a wider, softer bloom.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            4,
            "Sparkle must have exactly 4 passes"
        );
    }

    #[test]
    fn test_sparkle_make_uniform_known_value() {
        let reg = registry_by_id("sparkle").unwrap();
        let bytes = (reg.make_uniform)(&[0.7, 0.5, 0.3]);
        let expected = bytemuck::bytes_of(&SparkleParams {
            threshold: 0.7,
            intensity: 0.5,
            radius: 0.3,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_sparkle_default_values_are_identity() {
        // At threshold=1.0 no pixel qualifies (luma is always < 1.0 for normal images).
        // At intensity=0.0 any residual glow is scaled to zero. Both conditions
        // combine to make the default params a strict no-op.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sparkle",
                values: vec![IDENTITY_THRESHOLD, IDENTITY_INTENSITY, DEFAULT_RADIUS],
            }],
        );
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

    #[test]
    fn test_sparkle_zero_intensity_is_identity() {
        // With intensity=0.0 the combine pass outputs src + glow * 0 = src,
        // regardless of threshold or radius. Any mid-gray pixel must be unchanged.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sparkle",
                values: vec![0.0, 0.0, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767, got {}",
                pixel[0]
            );
        }
    }

    #[test]
    fn test_sparkle_high_threshold_produces_no_glow() {
        // At threshold=1.0 no pixel in a mid-gray image has luma >= 1.0, so the
        // threshold pass outputs black and no glow reaches the combine pass.
        // The output must be indistinguishable from the input even at high intensity.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sparkle",
                values: vec![1.0, 1.0, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767, got {}",
                pixel[0]
            );
        }
    }

    #[test]
    fn test_sparkle_low_threshold_brightens_output() {
        // A bright white image (65535) with a low threshold should accumulate glow
        // and produce output brighter than the source when intensity > 0.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a bright image to ensure pixels exceed any reasonable threshold.
        let img = make_solid_image(16, 16, 60000, 60000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sparkle",
                values: vec![0.1, 1.0, 0.5],
            }],
        );
        // At least some pixels must be brighter than the source (glow is additive).
        let any_brighter = out.pixels().any(|p| p[0] > 60000);
        assert!(
            any_brighter,
            "low threshold with intensity=1.0 must add measurable glow to bright pixels"
        );
    }

    #[test]
    fn test_sparkle_alpha_preserved() {
        // The combine pass copies alpha from the source; neither blur nor threshold
        // must alter the alpha channel.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 60000, 60000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sparkle",
                values: vec![0.1, 1.0, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be preserved through all Sparkle passes"
            );
        }
    }

    #[test]
    fn test_sparkle_higher_intensity_produces_more_glow() {
        // Holding threshold and radius constant, increasing intensity must produce
        // brighter output. The mean channel value of intensity=0.8 must exceed
        // that of intensity=0.4 on a uniformly bright image.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 55000, 55000, 55000);

        let out_low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sparkle",
                values: vec![0.2, 0.4, 0.5],
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sparkle",
                values: vec![0.2, 0.8, 0.5],
            }],
        );

        let sum_low: i64 = out_low.pixels().map(|p| p[0] as i64).sum();
        let sum_high: i64 = out_high.pixels().map(|p| p[0] as i64).sum();
        assert!(
            sum_high > sum_low,
            "higher intensity must produce brighter output: sum_low={sum_low}, sum_high={sum_high}"
        );
    }

    #[test]
    fn test_sparkle_larger_radius_widens_glow() {
        // On an image with a single bright spot surrounded by dark pixels, a larger
        // radius must spread the glow further, resulting in more pixels being above
        // the source value than with a smaller radius.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32×32 image: one bright pixel in the centre, rest dark.
        let mut img = crate::Rgba16Image::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v: u16 = if x == 15 && y == 15 { 65535 } else { 1000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_small = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sparkle",
                values: vec![0.1, 1.0, 0.1],
            }],
        );
        let out_large = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sparkle",
                values: vec![0.1, 1.0, 0.5],
            }],
        );

        // Count pixels that are meaningfully brighter than the dark background.
        let threshold_u16: u16 = 2000;
        let bright_small = out_small.pixels().filter(|p| p[0] > threshold_u16).count();
        let bright_large = out_large.pixels().filter(|p| p[0] > threshold_u16).count();

        assert!(
            bright_large > bright_small,
            "larger radius must spread glow to more pixels: \
             small={bright_small}, large={bright_large}"
        );
    }

    #[test]
    fn test_sparkle_dark_image_identity() {
        // A nearly black image should not be brightened regardless of threshold
        // and intensity, because dark pixels have no excess above any positive threshold.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 100, 100, 100);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sparkle",
                values: vec![0.5, 1.0, 0.5],
            }],
        );
        for pixel in out.pixels() {
            // Dark pixels are below threshold=0.5 (which is ~32768 in u16).
            // The threshold pass produces zero for them; the combine pass outputs src+0=src.
            assert!(
                (pixel[0] as i32 - 100).abs() <= 64,
                "dark pixel must be unchanged: got {}",
                pixel[0]
            );
        }
    }

    #[test]
    fn test_sparkle_chaining_with_brightness() {
        // Sparkle chained after Brightness must not panic and must produce correct
        // dimensions and valid alpha. This verifies multi-pass scratch-pool handoff.
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
                    values: vec![0.3],
                },
                Transform {
                    shader_id: "sparkle",
                    values: vec![0.5, 0.5, 0.3],
                },
            ],
        );
        assert_eq!(out.width(), 16);
        assert_eq!(out.height(), 16);
        for pixel in out.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be preserved through Brightness→Sparkle"
            );
        }
    }
}
