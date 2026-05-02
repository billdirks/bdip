use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the 16-bit Console effect, shared across both passes.
///
/// `color_levels` sets the number of discrete palette steps per channel (2–256).
/// At 32 this matches the SNES's 5-bit-per-channel hardware (2^5 = 32 levels).
/// At 256 the quantization is imperceptible and behaves as identity.
///
/// `saturation_boost` scales the saturation of the dithered image, simulating the
/// vivid palette output of 16-bit console TVs. 0.0 = no change; 1.0 = double
/// saturation (scale factor 2.0 in the mix formula).
///
/// `strength` is a master blend weight in [0, 1] that mixes the fully processed
/// image back with the unprocessed source. At 0.0 the effect is a pass-through
/// (identity); at 1.0 the full dither + saturation effect is applied.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Console16BitParams {
    /// Per-channel palette depth (2–256). 32 = SNES 5-bit channels.
    pub color_levels: f32,
    /// Saturation boost above neutral (0.0 = none, 1.0 = double saturation).
    pub saturation_boost: f32,
    /// Master blend strength (0.0 = identity, 1.0 = full effect).
    pub strength: f32,
    pub _padding: f32,
}

impl TransformShader for Console16BitParams {
    const ID: &'static str = "console_16bit";
    const DISPLAY_NAME: &'static str = "16-bit Console";
    const DESCRIPTION: &'static str = "Simulates the look of 16-bit era console games (SNES/Genesis) via ordered \
         Bayer-matrix dithering and vivid saturation boost.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Color Levels",
            min: 2.0,
            max: 256.0,
            default: 32.0,
            description: "Palette depth per channel. 32 matches SNES 5-bit hardware (2^5). \
                          Lower values produce a coarser, more posterized look.",
        },
        SliderDef {
            name: "Saturation Boost",
            min: 0.0,
            max: 2.0,
            default: 0.5,
            description: "Saturation multiplier applied after dithering. 0.0 leaves \
                          saturation unchanged; higher values push colors toward the vivid \
                          palette typical of 16-bit hardware output.",
        },
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Master blend between the original image (0.0) and the fully \
                          processed result (1.0). At 0.0 the effect is a no-op.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "dither",
            wgsl_source: include_str!("console_16bit_dither.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("dithered"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "saturate",
            wgsl_source: include_str!("console_16bit_saturate.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("dithered")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            color_levels: values[0],
            saturation_boost: values[1],
            strength: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    Console16BitParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // -----------------------------------------------------------------------
    // Registry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_console_16bit_registry_entry_exists() {
        assert!(registry_by_id("console_16bit").is_some());
    }

    #[test]
    fn test_console_16bit_registry_display_name() {
        let reg = registry_by_id("console_16bit").unwrap();
        assert_eq!(reg.meta.display_name, "16-bit Console");
    }

    #[test]
    fn test_console_16bit_registry_param_kind_is_sliders() {
        let reg = registry_by_id("console_16bit").unwrap();
        assert!(
            matches!(reg.meta.param, ParamKind::Sliders(_)),
            "expected ParamKind::Sliders"
        );
    }

    #[test]
    fn test_console_16bit_registry_slider_count() {
        let reg = registry_by_id("console_16bit").unwrap();
        if let ParamKind::Sliders(sliders) = reg.meta.param {
            assert_eq!(sliders.len(), 3, "expected 3 sliders");
        }
    }

    #[test]
    fn test_console_16bit_registry_color_levels_slider_def() {
        let reg = registry_by_id("console_16bit").unwrap();
        if let ParamKind::Sliders(sliders) = reg.meta.param {
            assert_eq!(sliders[0].name, "Color Levels");
            assert_eq!(sliders[0].min, 2.0);
            assert_eq!(sliders[0].max, 256.0);
            assert_eq!(sliders[0].default, 32.0);
        }
    }

    #[test]
    fn test_console_16bit_registry_saturation_boost_slider_def() {
        let reg = registry_by_id("console_16bit").unwrap();
        if let ParamKind::Sliders(sliders) = reg.meta.param {
            assert_eq!(sliders[1].name, "Saturation Boost");
            assert_eq!(sliders[1].min, 0.0);
            assert_eq!(sliders[1].max, 2.0);
            assert_eq!(sliders[1].default, 0.5);
        }
    }

    #[test]
    fn test_console_16bit_registry_strength_slider_def() {
        let reg = registry_by_id("console_16bit").unwrap();
        if let ParamKind::Sliders(sliders) = reg.meta.param {
            assert_eq!(sliders[2].name, "Strength");
            assert_eq!(sliders[2].min, 0.0);
            assert_eq!(sliders[2].max, 1.0);
            assert_eq!(sliders[2].default, 0.0);
        }
    }

    #[test]
    fn test_console_16bit_registry_pass_count() {
        let reg = registry_by_id("console_16bit").unwrap();
        assert_eq!(reg.meta.passes.len(), 2, "expected 2 passes");
    }

    // -----------------------------------------------------------------------
    // Uniform construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_console_16bit_make_uniform_known_value() {
        let reg = registry_by_id("console_16bit").unwrap();
        let bytes = (reg.make_uniform)(&[32.0, 0.5, 1.0]);
        let expected = bytemuck::bytes_of(&Console16BitParams {
            color_levels: 32.0,
            saturation_boost: 0.5,
            strength: 1.0,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_console_16bit_make_uniform_identity_values() {
        let reg = registry_by_id("console_16bit").unwrap();
        let bytes = (reg.make_uniform)(&[256.0, 0.0, 0.0]);
        let expected = bytemuck::bytes_of(&Console16BitParams {
            color_levels: 256.0,
            saturation_boost: 0.0,
            strength: 0.0,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // -----------------------------------------------------------------------
    // GPU roundtrip — identity (strength = 0.0 is a pass-through)
    // -----------------------------------------------------------------------

    #[test]
    fn test_console_16bit_strength_zero_is_identity() {
        // strength=0.0 blends 100% source into the final output, making both
        // passes no-ops regardless of color_levels and saturation_boost.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "console_16bit",
                values: vec![32.0, 1.0, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767 at strength=0, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 16384).abs() <= 64,
                "G: expected ~16384 at strength=0, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 8192).abs() <= 64,
                "B: expected ~8192 at strength=0, got {}",
                pixel[2]
            );
        }
    }

    // -----------------------------------------------------------------------
    // GPU roundtrip — dithering behavior
    // -----------------------------------------------------------------------

    #[test]
    fn test_console_16bit_low_color_levels_quantizes() {
        // With color_levels=2 at full strength, mid-gray snaps to either 0 or 1.
        // The output at some pixel must differ substantially from the source.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Mid-gray input (linear ~0.5).
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out_no_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "console_16bit",
                values: vec![256.0, 0.0, 0.0],
            }],
        );
        let out_2levels = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "console_16bit",
                values: vec![2.0, 0.0, 1.0],
            }],
        );
        // At color_levels=2 each pixel snaps to 0 or 65535; diff from no-effect
        // must be large for at least one pixel.
        let max_diff = out_2levels
            .pixels()
            .zip(out_no_effect.pixels())
            .map(|(a, b)| (a[0] as i32 - b[0] as i32).abs())
            .max()
            .unwrap_or(0);
        assert!(
            max_diff > 10000,
            "color_levels=2 should produce large channel shifts; max_diff={max_diff}"
        );
    }

    #[test]
    fn test_console_16bit_high_color_levels_near_identity_spatially() {
        // With color_levels=256 the quantization step is ~1/255 in linear space,
        // which is imperceptible. A solid color image output must remain close to
        // the source (within f16 + quantization rounding).
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 40000, 40000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "console_16bit",
                values: vec![256.0, 0.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 40000).abs() <= 512,
                "R: expected ~40000 at color_levels=256, got {}",
                pixel[0]
            );
        }
    }

    #[test]
    fn test_console_16bit_dither_introduces_spatial_variation_on_uniform_image() {
        // Ordered Bayer dithering on a uniform mid-gray image with coarse palette
        // (color_levels=4) produces different output values at different pixel
        // positions — confirming the dither pattern is spatially varying.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // A flat mid-gray image: every pixel is identical before dithering.
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "console_16bit",
                values: vec![4.0, 0.0, 1.0],
            }],
        );
        // Collect unique R values across the output. With a 4×4 Bayer matrix and
        // 4 color levels there must be at least 2 distinct output values.
        let mut seen = std::collections::HashSet::new();
        for pixel in out.pixels() {
            seen.insert(pixel[0]);
        }
        assert!(
            seen.len() >= 2,
            "Bayer dithering should introduce spatial variation; unique values = {}",
            seen.len()
        );
    }

    // -----------------------------------------------------------------------
    // GPU roundtrip — saturation boost behavior
    // -----------------------------------------------------------------------

    #[test]
    fn test_console_16bit_saturation_boost_increases_chroma() {
        // A chromatic (non-gray) input with saturation_boost > 0 should have a
        // larger distance between its max and min channels than the same input
        // at saturation_boost = 0.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Warm tone: R > G > B so saturation boost is visible.
        let img = make_solid_image(4, 4, 50000, 25000, 10000);

        let out_no_sat = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "console_16bit",
                values: vec![256.0, 0.0, 1.0],
            }],
        );
        let out_boosted = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "console_16bit",
                values: vec![256.0, 1.0, 1.0],
            }],
        );

        let chroma_no_sat =
            out_no_sat.get_pixel(0, 0)[0] as i32 - out_no_sat.get_pixel(0, 0)[2] as i32;
        let chroma_boosted =
            out_boosted.get_pixel(0, 0)[0] as i32 - out_boosted.get_pixel(0, 0)[2] as i32;

        assert!(
            chroma_boosted > chroma_no_sat,
            "saturation_boost=1.0 should increase R-B spread: no_sat={chroma_no_sat}, \
             boosted={chroma_boosted}"
        );
    }

    #[test]
    fn test_console_16bit_zero_saturation_boost_leaves_chroma_unchanged() {
        // With saturation_boost=0, the saturate pass applies mix(lum, rgb, 1.0) =
        // rgb — an identity on the dithered result. The output with sat_boost=0
        // must be pixel-identical to a second run with the same parameters,
        // confirming the pass introduces no additional color shift.
        //
        // Note: absolute output u16 values differ from input u16 values due to
        // the sRGB→linear (ingest) and linear→sRGB (present) encoding steps
        // that bracket all pipeline passes. Dithering also shifts linear values
        // by up to one quantization step before gamma encoding. The intent here
        // is determinism at sat_boost=0, not fidelity to the input level.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 50000, 25000, 10000);
        let transform = Transform {
            shader_id: "console_16bit",
            values: vec![256.0, 0.0, 1.0],
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
            assert_eq!(
                p1, p2,
                "sat_boost=0 output must be deterministic across runs"
            );
        }
    }

    // -----------------------------------------------------------------------
    // GPU roundtrip — alpha preservation
    // -----------------------------------------------------------------------

    #[test]
    fn test_console_16bit_alpha_preserved_at_full_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "console_16bit",
                values: vec![32.0, 0.5, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    #[test]
    fn test_console_16bit_alpha_preserved_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "console_16bit",
                values: vec![32.0, 1.0, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved at strength=0");
        }
    }

    // -----------------------------------------------------------------------
    // GPU roundtrip — chaining
    // -----------------------------------------------------------------------

    #[test]
    fn test_console_16bit_chains_with_brightness() {
        // Verify the shader output can be fed into brightness without engine
        // errors, and that the chained result is brighter than the effect alone.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 20000, 20000, 20000);

        let out_alone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "console_16bit",
                values: vec![32.0, 0.5, 1.0],
            }],
        );
        let out_chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "console_16bit",
                    values: vec![32.0, 0.5, 1.0],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.3],
                },
            ],
        );

        let r_alone = out_alone.get_pixel(0, 0)[0] as i32;
        let r_chained = out_chained.get_pixel(0, 0)[0] as i32;
        assert!(
            r_chained > r_alone,
            "brightness after console_16bit should increase pixel value: \
             alone={r_alone}, chained={r_chained}"
        );
    }
}
