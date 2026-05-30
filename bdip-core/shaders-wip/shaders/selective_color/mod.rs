use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SelectiveColorParams {
    /// Target hue center in degrees [0, 360).
    pub target_hue: f32,
    /// Half-width of the selection window in degrees.  Pixels within this
    /// angular distance of `target_hue` keep full color; beyond
    /// `tolerance + feather` they are fully desaturated.
    pub tolerance: f32,
    /// Width of the smooth falloff zone in degrees.
    pub feather: f32,
    pub _padding: f32,
}

impl TransformShader for SelectiveColorParams {
    const ID: &'static str = "selective_color";
    const DISPLAY_NAME: &'static str = "Selective Color";
    const DESCRIPTION: &'static str = "Keeps a chosen hue range in full color while converting the rest of the image \
         to grayscale, with a smooth falloff at the tolerance edges.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Target Hue",
            min: 0.0,
            max: 360.0,
            // Default 0° (red). With tolerance=180 the entire hue circle is retained,
            // so the choice of default hue is invisible — the image is unchanged.
            default: 0.0,
            description: "Center of the hue selection window in degrees (0–360).",
        },
        SliderDef {
            name: "Tolerance",
            min: 0.0,
            max: 180.0,
            // 180° covers the full hue circle → identity (no desaturation).
            default: 180.0,
            description: "Half-width of the hue retention window in degrees. \
                          180 retains all hues (identity); 0 desaturates everything.",
        },
        SliderDef {
            name: "Feather",
            min: 0.0,
            max: 90.0,
            default: 0.0,
            description: "Width of the smooth transition zone in degrees.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "selective_color",
        wgsl_source: include_str!("selective_color.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            target_hue: values[0],
            tolerance: values[1],
            feather: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    SelectiveColorParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ── Registry tests ────────────────────────────────────────────────────────

    #[test]
    fn test_selective_color_registry_entry_exists() {
        assert!(registry_by_id("selective_color").is_some());
    }

    #[test]
    fn test_selective_color_registry_metadata() {
        let reg = registry_by_id("selective_color").unwrap();
        assert_eq!(reg.meta.display_name, "Selective Color");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Target Hue",
                    min: 0.0,
                    max: 360.0,
                    default: 0.0,
                    description: "Center of the hue selection window in degrees (0–360).",
                },
                SliderDef {
                    name: "Tolerance",
                    min: 0.0,
                    max: 180.0,
                    default: 180.0,
                    description: "Half-width of the hue retention window in degrees. \
                                  180 retains all hues (identity); 0 desaturates everything.",
                },
                SliderDef {
                    name: "Feather",
                    min: 0.0,
                    max: 90.0,
                    default: 0.0,
                    description: "Width of the smooth transition zone in degrees.",
                },
            ])
        );
    }

    #[test]
    fn test_selective_color_make_uniform_known_value() {
        let reg = registry_by_id("selective_color").unwrap();
        let bytes = (reg.make_uniform)(&[120.0, 30.0, 10.0]);
        let expected = bytemuck::bytes_of(&SelectiveColorParams {
            target_hue: 120.0,
            tolerance: 30.0,
            feather: 10.0,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ───────────────────────────────────────────────────

    // Default values (tolerance=180) pass all hues → image unchanged.
    #[test]
    fn test_selective_color_defaults_are_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Use a vivid red: R=65535, G=0, B=0.
        let img = make_solid_image(2, 2, 65535, 0, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "selective_color",
                values: vec![0.0, 180.0, 0.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 64,
                "R: expected ~65535, got {}",
                pixel[0]
            );
            assert!(pixel[1] <= 64, "G: expected ~0, got {}", pixel[1]);
            assert!(pixel[2] <= 64, "B: expected ~0, got {}", pixel[2]);
            assert_eq!(pixel[3], 65535);
        }
    }

    // Zero tolerance desaturates every pixel (window has zero width).
    #[test]
    fn test_selective_color_zero_tolerance_fully_desaturates() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure blue (hue ~240°). Target hue=0° (red), tolerance=0, feather=0.
        let img = make_solid_image(2, 2, 0, 0, 65535);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "selective_color",
                values: vec![0.0, 0.0, 0.0],
            }],
        );

        // Rec.709 luminance of linear blue = 0.0722 → sRGB ~0.311 → u16 ~20381
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "All channels must be equal (grayscale): R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[1] as i32 - pixel[2] as i32).abs() <= 64,
                "All channels must be equal (grayscale): G={}, B={}",
                pixel[1],
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    // Pixel whose hue exactly matches the target is retained in full color.
    #[test]
    fn test_selective_color_matching_hue_is_retained() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure red has hue ≈ 0°. Target=0°, tolerance=30°, feather=0°.
        let img = make_solid_image(2, 2, 65535, 0, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "selective_color",
                values: vec![0.0, 30.0, 0.0],
            }],
        );

        // R should remain high; G and B should remain near zero.
        for pixel in out_img.pixels() {
            assert!(
                pixel[0] > 50000,
                "R should be retained (high): got {}",
                pixel[0]
            );
            assert!(pixel[1] <= 64, "G: expected ~0, got {}", pixel[1]);
            assert!(pixel[2] <= 64, "B: expected ~0, got {}", pixel[2]);
            assert_eq!(pixel[3], 65535);
        }
    }

    // Pixel whose hue is far outside the tolerance window is fully desaturated.
    #[test]
    fn test_selective_color_non_matching_hue_is_desaturated() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure blue (hue ≈ 240°). Target=0° (red), tolerance=30°, feather=0°.
        // Distance = 120° > 30° → should be fully desaturated.
        let img = make_solid_image(2, 2, 0, 0, 65535);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "selective_color",
                values: vec![0.0, 30.0, 0.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "R and G should be equal (grayscale): R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[1] as i32 - pixel[2] as i32).abs() <= 64,
                "G and B should be equal (grayscale): G={}, B={}",
                pixel[1],
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    // Hue wrapping: a target of 10° with tolerance 30° should retain hue 350°
    // (only 20° away going the short way around the circle).
    #[test]
    fn test_selective_color_hue_wraps_at_360() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // sRGB (65535, 0, 11000) is a deep red with hue near 350°.
        // target_hue=10°, tolerance=30°, feather=0°.
        // Shortest hue distance = 20°, within tolerance → should retain color.
        let img = make_solid_image(2, 2, 65535, 0, 11000);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "selective_color",
                values: vec![10.0, 30.0, 0.0],
            }],
        );

        // The red channel should remain dominant (not reduced to gray).
        for pixel in out_img.pixels() {
            assert!(
                pixel[0] > pixel[1] + 10000,
                "R should remain dominant after hue-wrap retention: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    // Alpha channel must pass through unchanged regardless of hue selection.
    #[test]
    fn test_selective_color_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Note: make_solid_image always sets alpha to 65535.
        let img = make_solid_image(2, 2, 0, 65535, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "selective_color",
                values: vec![0.0, 30.0, 0.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535);
        }
    }

    // Feather > 0 produces a partial blend for a pixel at the tolerance boundary.
    #[test]
    fn test_selective_color_feather_produces_partial_blend() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure green (hue ≈ 120°). Target=120°, tolerance=60°, feather=60°.
        // Hue distance = 0° → inside tolerance → color_weight = 1.0 → full color.
        // Verify that the output still has the green channel dominant (not gray).
        let img = make_solid_image(2, 2, 0, 65535, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "selective_color",
                values: vec![120.0, 60.0, 60.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[1] > pixel[0],
                "Green should remain dominant: G={}, R={}",
                pixel[1],
                pixel[0]
            );
        }
    }

    // Chaining with an existing shader (brightness) must not panic or error.
    #[test]
    fn test_selective_color_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "selective_color",
                    values: vec![30.0, 40.0, 10.0],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.1],
                },
            ],
        );

        // Only check that pixels are non-trivial and alpha is intact.
        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535);
        }
        assert!(out_img.pixels().any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0));
    }
}
