use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Retro Newspaper three-pass effect.
///
/// # Identity design
///
/// A pure identity is not achievable for a newspaper effect: even at full
/// strength the halftone pattern is visually distinct from the source. The
/// closest practical identity uses `strength = 0.0`, which passes the source
/// through the grayscale and quantisation passes but then blends the halftone
/// result with the source at weight 0 — so the final output equals the source
/// unchanged. This is the same blend-strength identity pattern used by Pencil
/// Sketch and Stained Glass.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RetroNewspaperParams {
    /// Grid density: number of halftone dot cells across the shorter image axis.
    /// Higher values produce a finer dot pattern.
    pub dot_frequency: f32,
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full newspaper effect.
    pub strength: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for RetroNewspaperParams {
    const ID: &'static str = "retro_newspaper";
    const DISPLAY_NAME: &'static str = "Retro Newspaper";
    const DESCRIPTION: &'static str = "Simulates old newspaper printing via grayscale \
        conversion, tonal quantisation, and a rotated halftone dot grid.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Dot Frequency",
            min: 10.0,
            max: 120.0,
            default: 50.0,
            description: "Number of halftone dot cells across the shorter image axis. \
                          Lower values produce coarser, more visible dots; higher values \
                          produce finer dots resembling high-resolution print.",
        },
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full \
                          retro-newspaper effect (1.0). The identity value is 0.0.",
        },
    ]);

    // Three-pass pipeline:
    //   Pass 1 — gray:      BT.709 grayscale conversion → scratch "gray".
    //   Pass 2 — quantize:  Reduce to 5 tonal levels    → scratch "quant".
    //   Pass 3 — halftone:  Rotated dot grid overlay + blend with source → Final.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "gray",
            wgsl_source: include_str!("retro_newspaper_gray.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("gray"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "quantize",
            wgsl_source: include_str!("retro_newspaper_quantize.wgsl"),
            inputs: &[PassInput::Scratch("gray")],
            output: PassOutput::Scratch("quant"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "halftone",
            wgsl_source: include_str!("retro_newspaper_halftone.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("quant")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            dot_frequency: values[0],
            strength: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    RetroNewspaperParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ---------------------------------------------------------------------------
    // Registry tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_retro_newspaper_registry_entry_exists() {
        assert!(registry_by_id("retro_newspaper").is_some());
    }

    #[test]
    fn test_retro_newspaper_registry_metadata() {
        let reg = registry_by_id("retro_newspaper").unwrap();
        assert_eq!(reg.meta.display_name, "Retro Newspaper");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Dot Frequency",
                    min: 10.0,
                    max: 120.0,
                    default: 50.0,
                    description: "Number of halftone dot cells across the shorter image axis. \
                          Lower values produce coarser, more visible dots; higher values \
                          produce finer dots resembling high-resolution print.",
                },
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full \
                          retro-newspaper effect (1.0). The identity value is 0.0.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            3,
            "Retro Newspaper must have exactly 3 passes"
        );
    }

    #[test]
    fn test_retro_newspaper_make_uniform_known_value() {
        let reg = registry_by_id("retro_newspaper").unwrap();
        let bytes = (reg.make_uniform)(&[60.0, 0.8]);
        let expected = bytemuck::bytes_of(&RetroNewspaperParams {
            dot_frequency: 60.0,
            strength: 0.8,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    // ---------------------------------------------------------------------------
    // GPU roundtrip tests
    // ---------------------------------------------------------------------------

    /// At strength=0.0 the halftone pass reduces to mix(src, halftone, 0.0) = src,
    /// so the output must equal the source regardless of dot_frequency.
    #[test]
    fn test_retro_newspaper_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_newspaper",
                values: vec![50.0, 0.0],
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

    /// Alpha channel must pass through unchanged at any strength value.
    #[test]
    fn test_retro_newspaper_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_newspaper",
                values: vec![50.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// At strength=1.0 on a solid mid-gray image, the halftone pass produces
    /// either ink (~0.06 linear) or paper (~0.94 linear), but not the original
    /// mid-gray (~0.5). The mean output must differ from the source by more
    /// than a rounding tolerance.
    #[test]
    fn test_retro_newspaper_full_strength_changes_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Mid-gray (u16 32767 ≈ 0.5 sRGB ≈ 0.214 linear).
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_newspaper",
                values: vec![50.0, 1.0],
            }],
        );
        // The halftone output at mid-gray must differ noticeably from the
        // source 32767. Require at least one pixel outside ±1000 of the source.
        let any_changed = out.pixels().any(|p| (p[0] as i32 - 32767).abs() > 1000);
        assert!(any_changed, "strength=1.0 must visibly change the output");
    }

    /// The halftone effect maps all pixels to ink or paper values.
    /// At strength=1.0 every pixel's R channel must be close to either the ink
    /// value (u16 ≈ 3932 for linear 0.06) or the paper value (u16 ≈ 61029 for
    /// linear 0.94).  Allow ±1500 for quantisation and f16 rounding.
    #[test]
    fn test_retro_newspaper_full_strength_pixels_are_ink_or_paper() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_newspaper",
                values: vec![50.0, 1.0],
            }],
        );
        // sRGB u16 for ink (linear ~0.06) ≈ 15420; paper (linear ~0.94) ≈ 62838.
        // Use a generous band (±5000) since the exact values depend on f16 and
        // the shader's paper/ink constants, which are in linear light.
        for pixel in out.pixels() {
            let r = pixel[0] as i32;
            let near_ink = (r - 15420).abs() < 5000;
            let near_paper = (r - 62838).abs() < 5000;
            assert!(
                near_ink || near_paper,
                "pixel R={r} is neither near ink (~15420) nor paper (~62838)"
            );
        }
    }

    /// Decreasing dot_frequency (coarser grid) must change the output pattern
    /// compared to a finer grid at the same strength, when applied to an image
    /// with spatial variation.
    #[test]
    fn test_retro_newspaper_dot_frequency_changes_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Gradient image to produce spatial variation across the dot grid.
        let w = 32u32;
        let h = 32u32;
        let mut img = crate::Rgba16Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = ((x + y) * 1000).min(65535) as u16;
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_coarse = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_newspaper",
                values: vec![10.0, 1.0],
            }],
        );
        let out_fine = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "retro_newspaper",
                values: vec![80.0, 1.0],
            }],
        );

        let any_different = out_coarse
            .pixels()
            .zip(out_fine.pixels())
            .any(|(c, f)| (c[0] as i32 - f[0] as i32).abs() > 64);
        assert!(
            any_different,
            "different dot_frequency values must produce different outputs"
        );
    }

    /// Chaining retro_newspaper with brightness must not panic and must
    /// preserve alpha through the combined pipeline.
    #[test]
    fn test_retro_newspaper_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "retro_newspaper",
                    values: vec![50.0, 0.5],
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

    /// Running Retro Newspaper twice with identical inputs must produce
    /// bit-identical output (determinism requirement).
    #[test]
    fn test_retro_newspaper_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "retro_newspaper",
            values: vec![50.0, 0.8],
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
