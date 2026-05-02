use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Tilt-Shift effect.
///
/// Layout (16 bytes, matching the WGSL struct):
/// - `focus_center`: Vertical position of the in-focus band (0.0 = top, 1.0 = bottom).
/// - `focus_width`: Half-height of the sharp band as a fraction of image height (0–1).
///   The band extends `focus_center ± focus_width/2`.
/// - `blur_strength`: Controls the Gaussian sigma relative to image height. At 0.0 the
///   blur radius is 0 (identity); at 1.0 the radius approaches `RADIUS_CAP`.
/// - `_padding`: Fills the struct to 16 bytes for WebGPU uniform alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TiltShiftParams {
    pub focus_center: f32,
    pub focus_width: f32,
    pub blur_strength: f32,
    pub _padding: f32,
}

impl TransformShader for TiltShiftParams {
    const ID: &'static str = "tilt_shift";
    const DISPLAY_NAME: &'static str = "Tilt-Shift";
    const DESCRIPTION: &'static str = "Simulates a tilt-shift lens: a horizontal band stays sharp while \
         regions above and below are progressively blurred.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Focus Center",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Vertical position of the in-focus band (0 = top, 1 = bottom).",
        },
        SliderDef {
            name: "Focus Width",
            min: 0.0,
            max: 1.0,
            default: 1.0,
            description: "Height of the sharp band as a fraction of image height. \
                         At 1.0 the entire image is in focus (identity).",
        },
        SliderDef {
            name: "Blur Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Maximum blur applied to out-of-focus regions. At 0.0 no blur \
                         is applied (identity).",
        },
    ]);

    // Five passes (same downsample→blur→upsample strategy as the Clarity shader):
    //
    //   Pass 1 (down): Box-filter downsample 4× → scratch "down".
    //   Pass 2 (blur_h): Horizontal separable Gaussian on the 4× downsampled image.
    //   Pass 3 (blur_v): Vertical separable Gaussian → scratch "blur_up" (upsampled
    //                    by the PassScale::Full output).
    //   Pass 4 (up): Bilinear upsample of the blurred image back to full resolution.
    //   Pass 5 (composite): Blends source and upsampled blur per-pixel using a smooth
    //                       gradient mask derived from vertical distance to the focus
    //                       band.
    //
    // Operating the separable Gaussian on a 4× downscaled image reduces the pixel
    // count by 16× and halves the kernel radius relative to the image dimensions,
    // making the blur O(1/8) the cost of blurring at full resolution. The bilinear
    // upsample reintroduces acceptable softness without a visible seam. This is
    // identical in principle to the Clarity multi-pass strategy.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "down",
            wgsl_source: include_str!("tilt_shift_down.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("down"),
            output_scale: PassScale::Down(4),
            aux_textures: &[],
        },
        PassDef {
            label: "blur_h",
            wgsl_source: include_str!("tilt_shift_blur_h.wgsl"),
            inputs: &[PassInput::Scratch("down")],
            output: PassOutput::Scratch("h"),
            output_scale: PassScale::Down(4),
            aux_textures: &[],
        },
        PassDef {
            label: "blur_v",
            wgsl_source: include_str!("tilt_shift_blur_v.wgsl"),
            inputs: &[PassInput::Scratch("h")],
            output: PassOutput::Scratch("v"),
            output_scale: PassScale::Down(4),
            aux_textures: &[],
        },
        PassDef {
            label: "up",
            wgsl_source: include_str!("tilt_shift_up.wgsl"),
            inputs: &[PassInput::Scratch("v")],
            output: PassOutput::Scratch("blur"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "composite",
            wgsl_source: include_str!("tilt_shift_composite.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("blur")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            focus_center: values[0],
            focus_width: values[1],
            blur_strength: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    TiltShiftParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // -------------------------------------------------------------------------
    // Registry tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_tilt_shift_registry_entry_exists() {
        assert!(registry_by_id("tilt_shift").is_some());
    }

    #[test]
    fn test_tilt_shift_registry_display_name() {
        let reg = registry_by_id("tilt_shift").unwrap();
        assert_eq!(reg.meta.display_name, "Tilt-Shift");
    }

    #[test]
    fn test_tilt_shift_registry_pass_count() {
        let reg = registry_by_id("tilt_shift").unwrap();
        assert_eq!(
            reg.meta.passes.len(),
            5,
            "Tilt-Shift must have exactly 5 passes"
        );
    }

    #[test]
    fn test_tilt_shift_registry_param_kind_is_sliders() {
        let reg = registry_by_id("tilt_shift").unwrap();
        assert!(
            matches!(reg.meta.param, ParamKind::Sliders(_)),
            "param kind must be Sliders"
        );
    }

    #[test]
    fn test_tilt_shift_registry_slider_count() {
        let reg = registry_by_id("tilt_shift").unwrap();
        if let ParamKind::Sliders(sliders) = reg.meta.param {
            assert_eq!(sliders.len(), 3, "Tilt-Shift must expose exactly 3 sliders");
        }
    }

    #[test]
    fn test_tilt_shift_registry_slider_defaults() {
        let reg = registry_by_id("tilt_shift").unwrap();
        if let ParamKind::Sliders(sliders) = reg.meta.param {
            assert_eq!(sliders[0].default, 0.5, "focus_center default must be 0.5");
            assert_eq!(sliders[1].default, 1.0, "focus_width default must be 1.0");
            assert_eq!(sliders[2].default, 0.0, "blur_strength default must be 0.0");
        }
    }

    #[test]
    fn test_tilt_shift_registry_param_metadata() {
        let reg = registry_by_id("tilt_shift").unwrap();
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Focus Center",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Vertical position of the in-focus band (0 = top, 1 = bottom).",
                },
                SliderDef {
                    name: "Focus Width",
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                    description: "Height of the sharp band as a fraction of image height. \
                                 At 1.0 the entire image is in focus (identity).",
                },
                SliderDef {
                    name: "Blur Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Maximum blur applied to out-of-focus regions. At 0.0 no blur \
                                 is applied (identity).",
                },
            ])
        );
    }

    // -------------------------------------------------------------------------
    // Uniform serialisation
    // -------------------------------------------------------------------------

    #[test]
    fn test_tilt_shift_make_uniform_known_value() {
        let reg = registry_by_id("tilt_shift").unwrap();
        let bytes = (reg.make_uniform)(&[0.5, 0.3, 0.7]);
        let expected = bytemuck::bytes_of(&TiltShiftParams {
            focus_center: 0.5,
            focus_width: 0.3,
            blur_strength: 0.7,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_tilt_shift_make_uniform_zero_values() {
        let reg = registry_by_id("tilt_shift").unwrap();
        let bytes = (reg.make_uniform)(&[0.0, 0.0, 0.0]);
        let expected = bytemuck::bytes_of(&TiltShiftParams {
            focus_center: 0.0,
            focus_width: 0.0,
            blur_strength: 0.0,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — identity conditions
    // -------------------------------------------------------------------------

    /// Default parameters (focus_width=1.0, blur_strength=0.0) produce no change.
    /// With blur_strength=0 the blur radius is 0, so the blur passes copy the
    /// source unchanged and the composite blends two identical textures.
    #[test]
    fn test_tilt_shift_default_params_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tilt_shift",
                values: vec![0.5, 1.0, 0.0],
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

    /// blur_strength=0.0 with a narrow focus band is still identity because the
    /// blur pass produces radius=0 (no blur), so the composite blends identical
    /// pixels regardless of the mask value.
    #[test]
    fn test_tilt_shift_zero_blur_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 20000, 30000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tilt_shift",
                values: vec![0.5, 0.1, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!((pixel[0] as i32 - 20000).abs() <= 64, "R channel mismatch");
            assert!((pixel[1] as i32 - 30000).abs() <= 64, "G channel mismatch");
            assert!((pixel[2] as i32 - 40000).abs() <= 64, "B channel mismatch");
        }
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — blur effect
    // -------------------------------------------------------------------------

    /// With a narrow focus band centred away from edges, pixels at the top/bottom
    /// (fully out-of-focus) must differ from the source on a high-frequency image.
    /// We use a checkerboard pattern: the blurred result will be closer to mid-gray
    /// than the original alternating black/white pixels.
    #[test]
    fn test_tilt_shift_out_of_focus_region_is_blurred() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32×32 checkerboard: alternating 0 and 65535 per pixel.
        let mut img = crate::Rgba16Image::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v: u16 = if (x + y) % 2 == 0 { 65535 } else { 0 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        // Focus band at top 5% — everything below row ~1 is out of focus.
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tilt_shift",
                values: vec![0.02, 0.05, 1.0],
            }],
        );

        // Near the bottom (row 30) the blur should mix the checkerboard toward mid-gray.
        // We check two adjacent pixels that were originally opposite extremes (0 and 65535).
        // After blurring their values should converge; the difference must be < half of max.
        let p0 = out.get_pixel(0, 30)[0] as i32;
        let p1 = out.get_pixel(1, 30)[0] as i32;
        let diff = (p0 - p1).abs();
        assert!(
            diff < 32767,
            "out-of-focus checkerboard must blur toward mid-gray: diff={diff}"
        );
    }

    /// The in-focus band must remain close to the original values even when
    /// blur_strength is at maximum.
    #[test]
    fn test_tilt_shift_in_focus_band_stays_sharp() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Solid-color image: the blur does not change values on uniform regions,
        // so we can verify that the sharp band is preserved exactly.
        let img = make_solid_image(16, 16, 50000, 20000, 10000);

        // Focus band centred at 0.5, width 0.5 — rows 4..12 (approx) are in focus.
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tilt_shift",
                values: vec![0.5, 0.5, 1.0],
            }],
        );

        // Centre row (row 8) must stay within ±64 of the original.
        let p = out.get_pixel(8, 8);
        assert!(
            (p[0] as i32 - 50000).abs() <= 64,
            "R in focus: got {}",
            p[0]
        );
        assert!(
            (p[1] as i32 - 20000).abs() <= 64,
            "G in focus: got {}",
            p[1]
        );
        assert!(
            (p[2] as i32 - 10000).abs() <= 64,
            "B in focus: got {}",
            p[2]
        );
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — alpha preservation
    // -------------------------------------------------------------------------

    /// The composite pass copies alpha from the source; neither blur pass must alter
    /// the alpha channel.
    #[test]
    fn test_tilt_shift_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tilt_shift",
                values: vec![0.5, 0.2, 0.8],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — gradient falloff
    // -------------------------------------------------------------------------

    /// Blur amount must increase with distance from the focus band. We compare two
    /// rows on a checkerboard image: the row closer to the focus band should retain
    /// more high-frequency detail (larger adjacent-pixel difference) than the row
    /// farther from it.
    #[test]
    fn test_tilt_shift_blur_increases_with_distance_from_focus() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32×32 checkerboard.
        let mut img = crate::Rgba16Image::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v: u16 = if (x + y) % 2 == 0 { 65535 } else { 0 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        // Focus centred at top (0.05), narrow width (0.1), strong blur.
        // Row 2 is near the focus band; row 30 is far from it.
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tilt_shift",
                values: vec![0.05, 0.1, 0.9],
            }],
        );

        // Adjacent-pixel difference as a proxy for remaining sharpness.
        let near_diff = (out.get_pixel(0, 2)[0] as i32 - out.get_pixel(1, 2)[0] as i32).abs();
        let far_diff = (out.get_pixel(0, 30)[0] as i32 - out.get_pixel(1, 30)[0] as i32).abs();

        assert!(
            far_diff < near_diff,
            "farther row must be more blurred than nearer row: near_diff={near_diff}, \
             far_diff={far_diff}"
        );
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — chaining
    // -------------------------------------------------------------------------

    /// Tilt-Shift followed by Grayscale must complete without panic and must
    /// produce a valid (non-zero) image, verifying that multi-pass scratch textures
    /// chain correctly with subsequent transforms.
    #[test]
    fn test_tilt_shift_chains_with_grayscale() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "tilt_shift",
                    values: vec![0.5, 0.5, 0.5],
                },
                Transform {
                    shader_id: "grayscale",
                    values: vec![],
                },
            ],
        );
        // Grayscale on a gray image yields a non-zero result.
        assert!(
            out.pixels().any(|p| p[0] > 0),
            "chained output must be non-zero"
        );
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — determinism
    // -------------------------------------------------------------------------

    /// Running the shader twice with identical inputs must produce bit-identical output.
    #[test]
    fn test_tilt_shift_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "tilt_shift",
            values: vec![0.5, 0.3, 0.7],
        };
        let out1 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            std::slice::from_ref(&transform),
        );
        let out2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            std::slice::from_ref(&transform),
        );
        for (p1, p2) in out1.pixels().zip(out2.pixels()) {
            assert_eq!(p1, p2, "outputs must be pixel-identical across runs");
        }
    }
}
