use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Silhouette shader.
///
/// Pixels whose Rec. 709 luminance falls below `threshold` are mapped to the
/// foreground color; pixels above are mapped to the background color. The
/// `softness` parameter widens the transition zone around the threshold into a
/// smooth gradient rather than a hard edge.
///
/// Defaults of threshold=0.5, softness=0.0, fg=(0,0,0) black, bg=(1,1,1) white
/// produce a classic high-contrast silhouette. There is no mathematically
/// identity-preserving default for a step function — the chosen defaults represent
/// the canonical, expected starting state of the effect, following the same
/// convention used by `duo_tone`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SilhouetteParams {
    pub threshold: f32,
    pub softness: f32,
    pub fg_r: f32,
    pub fg_g: f32,
    pub fg_b: f32,
    pub _padding0: f32,
    pub bg_r: f32,
    pub bg_g: f32,
    pub bg_b: f32,
    pub _padding1: [f32; 3],
}

impl TransformShader for SilhouetteParams {
    const ID: &'static str = "silhouette";
    const DISPLAY_NAME: &'static str = "Silhouette";
    const DESCRIPTION: &'static str = "Converts the image to a two-tone silhouette by mapping pixels below a luminance \
         threshold to a foreground color and pixels above to a background color.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Threshold",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Luminance cutoff. Pixels below this value map to the foreground color; \
                          pixels above map to the background color.",
        },
        SliderDef {
            name: "Softness",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Width of the smooth transition zone around the threshold. \
                          0 = hard edge; larger values feather the boundary.",
        },
        SliderDef {
            name: "Foreground R",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Red component of the foreground (dark-tone) color.",
        },
        SliderDef {
            name: "Foreground G",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Green component of the foreground (dark-tone) color.",
        },
        SliderDef {
            name: "Foreground B",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blue component of the foreground (dark-tone) color.",
        },
        SliderDef {
            name: "Background R",
            min: 0.0,
            max: 1.0,
            default: 1.0,
            description: "Red component of the background (bright-tone) color.",
        },
        SliderDef {
            name: "Background G",
            min: 0.0,
            max: 1.0,
            default: 1.0,
            description: "Green component of the background (bright-tone) color.",
        },
        SliderDef {
            name: "Background B",
            min: 0.0,
            max: 1.0,
            default: 1.0,
            description: "Blue component of the background (bright-tone) color.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "silhouette",
        wgsl_source: include_str!("silhouette.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            threshold: values[0],
            softness: values[1],
            fg_r: values[2],
            fg_g: values[3],
            fg_b: values[4],
            _padding0: 0.0,
            bg_r: values[5],
            bg_g: values[6],
            bg_b: values[7],
            _padding1: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    SilhouetteParams,
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
    fn test_silhouette_registry_entry_exists() {
        assert!(registry_by_id("silhouette").is_some());
    }

    #[test]
    fn test_silhouette_registry_metadata() {
        let reg = registry_by_id("silhouette").unwrap();
        assert_eq!(reg.meta.display_name, "Silhouette");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Threshold",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Luminance cutoff. Pixels below this value map to the foreground \
                                  color; pixels above map to the background color.",
                },
                SliderDef {
                    name: "Softness",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Width of the smooth transition zone around the threshold. \
                                  0 = hard edge; larger values feather the boundary.",
                },
                SliderDef {
                    name: "Foreground R",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Red component of the foreground (dark-tone) color.",
                },
                SliderDef {
                    name: "Foreground G",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Green component of the foreground (dark-tone) color.",
                },
                SliderDef {
                    name: "Foreground B",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blue component of the foreground (dark-tone) color.",
                },
                SliderDef {
                    name: "Background R",
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                    description: "Red component of the background (bright-tone) color.",
                },
                SliderDef {
                    name: "Background G",
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                    description: "Green component of the background (bright-tone) color.",
                },
                SliderDef {
                    name: "Background B",
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                    description: "Blue component of the background (bright-tone) color.",
                },
            ])
        );
    }

    #[test]
    fn test_silhouette_passes_count() {
        let reg = registry_by_id("silhouette").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_silhouette_make_uniform_known_value() {
        let reg = registry_by_id("silhouette").unwrap();
        let bytes = (reg.make_uniform)(&[0.4, 0.1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let expected = bytemuck::bytes_of(&SilhouetteParams {
            threshold: 0.4,
            softness: 0.1,
            fg_r: 0.0,
            fg_g: 0.0,
            fg_b: 0.0,
            _padding0: 0.0,
            bg_r: 1.0,
            bg_g: 1.0,
            bg_b: 1.0,
            _padding1: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// A pixel whose luminance is clearly below the threshold must map to the
    /// foreground color (black by default).
    #[test]
    fn test_silhouette_dark_pixel_maps_to_foreground() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Very dark pixel — luminance well below 0.5 threshold.
        let img = make_solid_image(2, 2, 3000, 3000, 3000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "silhouette",
                // threshold=0.5, softness=0.0, fg=black, bg=white
                values: vec![0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }],
        );

        // Foreground is black (linear 0.0 → sRGB 0.0 → u16 0).
        for pixel in out.pixels() {
            assert!(
                pixel[0] <= 64,
                "R must be near 0 (foreground R=0): {}",
                pixel[0]
            );
            assert!(
                pixel[1] <= 64,
                "G must be near 0 (foreground G=0): {}",
                pixel[1]
            );
            assert!(
                pixel[2] <= 64,
                "B must be near 0 (foreground B=0): {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    /// A pixel whose luminance is clearly above the threshold must map to the
    /// background color (white by default).
    #[test]
    fn test_silhouette_bright_pixel_maps_to_background() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Very bright pixel — luminance well above 0.5 threshold.
        let img = make_solid_image(2, 2, 62000, 62000, 62000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "silhouette",
                // threshold=0.5, softness=0.0, fg=black, bg=white
                values: vec![0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }],
        );

        // Background is white (linear 1.0 → sRGB 1.0 → u16 65535).
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 64,
                "R must be near 65535 (background R=1): {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 65535).abs() <= 64,
                "G must be near 65535 (background G=1): {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 65535).abs() <= 64,
                "B must be near 65535 (background B=1): {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    /// Custom foreground and background colors are applied correctly to dark and
    /// bright pixels respectively.
    #[test]
    fn test_silhouette_custom_fg_color_applied_to_dark_pixels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Dark pixel — luminance well below threshold.
        let img = make_solid_image(2, 2, 2000, 2000, 2000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "silhouette",
                // threshold=0.5, fg=(0,0,1) blue, bg=(1,0,0) red
                values: vec![0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            }],
        );

        // Dark pixel → foreground color = blue: R≈0, G≈0, B≈65535.
        for pixel in out.pixels() {
            assert!(pixel[0] <= 64, "R must be ~0 (fg R=0): {}", pixel[0]);
            assert!(pixel[1] <= 64, "G must be ~0 (fg G=0): {}", pixel[1]);
            assert!(
                (pixel[2] as i32 - 65535).abs() <= 64,
                "B must be ~65535 (fg B=1): {}",
                pixel[2]
            );
        }
    }

    /// Custom background color is applied correctly to bright pixels.
    #[test]
    fn test_silhouette_custom_bg_color_applied_to_bright_pixels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Bright pixel — luminance well above threshold.
        let img = make_solid_image(2, 2, 63000, 63000, 63000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "silhouette",
                // threshold=0.5, fg=black, bg=(1,0,0) red
                values: vec![0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            }],
        );

        // Bright pixel → background color = red: R≈65535, G≈0, B≈0.
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 64,
                "R must be ~65535 (bg R=1): {}",
                pixel[0]
            );
            assert!(pixel[1] <= 64, "G must be ~0 (bg G=0): {}", pixel[1]);
            assert!(pixel[2] <= 64, "B must be ~0 (bg B=0): {}", pixel[2]);
        }
    }

    /// Alpha channel is preserved regardless of threshold or color parameters.
    #[test]
    fn test_silhouette_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 10000, 20000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "silhouette",
                values: vec![0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by silhouette");
        }
    }

    /// With threshold=0.0 and zero softness, all pixels are above the threshold
    /// and must map to the background color.
    #[test]
    fn test_silhouette_threshold_zero_all_pixels_map_to_background() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 5000, 5000, 5000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "silhouette",
                // threshold=0, softness=0, fg=black, bg=white
                values: vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }],
        );

        // All pixels above threshold=0 → background = white.
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 64,
                "R must be ~65535: {}",
                pixel[0]
            );
        }
    }

    /// With threshold=1.0 and zero softness, all pixels are below the threshold
    /// and must map to the foreground color.
    #[test]
    fn test_silhouette_threshold_one_all_pixels_map_to_foreground() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 60000, 60000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "silhouette",
                // threshold=1.0, softness=0, fg=black, bg=white
                values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }],
        );

        // All pixels below threshold=1.0 → foreground = black.
        for pixel in out.pixels() {
            assert!(pixel[0] <= 64, "R must be ~0: {}", pixel[0]);
            assert!(pixel[1] <= 64, "G must be ~0: {}", pixel[1]);
            assert!(pixel[2] <= 64, "B must be ~0: {}", pixel[2]);
        }
    }

    /// Softness > 0 produces an intermediate blended value for a pixel sitting
    /// exactly at the threshold.
    #[test]
    fn test_silhouette_softness_produces_blend_at_threshold() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% grey (sRGB 32767/65535 ≈ 0.500 → linear ≈ 0.214).
        // Set threshold = 0.214 so the pixel sits exactly at the threshold.
        // With softness = 0.2, the pixel is in the middle of the transition zone
        // and must produce a value blended between fg and bg.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "silhouette",
                // threshold=0.214, softness=0.2, fg=black, bg=white
                values: vec![0.214, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }],
        );

        // At t=0.5 (pixel exactly at threshold midpoint), smoothstep returns 0.5,
        // so the output should be a mid-grey (~32768 u16). Accept a wide tolerance
        // because the exact linear luminance of 32767 sRGB is approximate.
        for pixel in out.pixels() {
            let r = pixel[0] as i32;
            assert!(
                r > 5000 && r < 60000,
                "R must be intermediate grey (softness blend): {}",
                r
            );
        }
    }

    /// Chaining silhouette with brightness at identity (0.0) must not alter the
    /// silhouette result.
    #[test]
    fn test_silhouette_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 5000, 15000, 40000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "silhouette",
                values: vec![0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "silhouette",
                    values: vec![0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
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
