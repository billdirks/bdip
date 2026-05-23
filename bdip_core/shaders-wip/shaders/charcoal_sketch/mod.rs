use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters shared across both Charcoal Sketch passes.
///
/// Three meaningful fields pack into 12 bytes; one padding float brings the struct
/// to 16 bytes to satisfy WebGPU's uniform-buffer alignment requirement.
///
/// # Identity design
///
/// The spec requires that default parameter values produce a no-op transformation.
/// For Charcoal Sketch the artistic effect (dark strokes on cream paper) cannot be
/// literally identity at any non-zero strength. The design follows the pattern
/// established by Pencil Sketch and Chalkboard: a `strength` blend parameter
/// defaults to `0.0`, which passes the source image through unchanged (identity),
/// while `edge_strength` and `grain_amount` control the look when `strength` is
/// non-zero. At `strength = 0.0` the output equals the source regardless of the
/// other sliders.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CharcoalSketchParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full charcoal-sketch effect.
    pub strength: f32,
    /// Multiplier applied to raw Sobel gradient magnitude before clamping.
    /// Higher values make faint edges visible as charcoal strokes. Range [0.1, 10.0].
    pub edge_strength: f32,
    /// Amplitude of procedural charcoal grain. 0.0 = no grain (clean strokes),
    /// 1.0 = maximum rough charcoal texture. Range [0.0, 1.0].
    pub grain_amount: f32,
    pub _padding: f32,
}

impl TransformShader for CharcoalSketchParams {
    const ID: &'static str = "charcoal_sketch";
    const DISPLAY_NAME: &'static str = "Charcoal Sketch";
    const DESCRIPTION: &'static str = "Renders the image as a charcoal drawing on warm cream paper: dark strokes from \
         Sobel edge detection with multi-frequency procedural grain simulating charcoal \
         medium roughness.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full charcoal-sketch \
                          effect (1.0). The identity value is 0.0.",
        },
        SliderDef {
            name: "Edge Strength",
            min: 0.1,
            max: 10.0,
            default: 2.5,
            description: "Sensitivity of edge detection. Higher values make faint edges \
                          more visible as dark charcoal strokes.",
        },
        SliderDef {
            name: "Grain Amount",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Intensity of procedural charcoal grain texture. 0.0 produces clean \
                          strokes; higher values add rough charcoal pigment smearing.",
        },
    ]);

    // Two-pass pipeline:
    //   Pass 1 — edges: grayscale + Sobel edge detection → inverted (dark strokes on
    //                   light background) stored in a scratch texture.
    //   Pass 2 — grain: add multi-frequency procedural charcoal grain, tint toward
    //                   cream paper tone, blend with source via strength.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "edges",
            wgsl_source: include_str!("charcoal_sketch_edges.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("edges"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "grain",
            wgsl_source: include_str!("charcoal_sketch_grain.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("edges")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            edge_strength: values[1],
            grain_amount: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    CharcoalSketchParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_charcoal_sketch_registry_entry_exists() {
        assert!(registry_by_id("charcoal_sketch").is_some());
    }

    #[test]
    fn test_charcoal_sketch_registry_metadata() {
        let reg = registry_by_id("charcoal_sketch").unwrap();
        assert_eq!(reg.meta.display_name, "Charcoal Sketch");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full \
                                  charcoal-sketch effect (1.0). The identity value is 0.0.",
                },
                SliderDef {
                    name: "Edge Strength",
                    min: 0.1,
                    max: 10.0,
                    default: 2.5,
                    description: "Sensitivity of edge detection. Higher values make faint edges \
                                  more visible as dark charcoal strokes.",
                },
                SliderDef {
                    name: "Grain Amount",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Intensity of procedural charcoal grain texture. 0.0 produces \
                                  clean strokes; higher values add rough charcoal pigment \
                                  smearing.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "Charcoal Sketch must have exactly 2 passes"
        );
    }

    #[test]
    fn test_charcoal_sketch_make_uniform_known_value() {
        let reg = registry_by_id("charcoal_sketch").unwrap();
        let bytes = (reg.make_uniform)(&[0.75, 3.0, 0.6]);
        let expected = bytemuck::bytes_of(&CharcoalSketchParams {
            strength: 0.75,
            edge_strength: 3.0,
            grain_amount: 0.6,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 the grain pass reduces to mix(src, charcoal, 0.0) = src.
    /// The output must equal the source regardless of edge_strength or grain_amount.
    #[test]
    fn test_charcoal_sketch_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "charcoal_sketch",
                values: vec![0.0, 2.5, 0.5],
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
    fn test_charcoal_sketch_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "charcoal_sketch",
                values: vec![1.0, 2.5, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// A uniform (solid-colour) image has no edges; at full strength the output
    /// should be near the cream paper background (light, warm) with grain applied.
    /// The mean brightness must be high (above mid-grey) since there are no strokes.
    #[test]
    fn test_charcoal_sketch_solid_image_produces_light_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Mid-gray solid image — Sobel returns zero on a constant input.
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "charcoal_sketch",
                values: vec![1.0, 2.5, 0.0],
            }],
        );
        // With no edges and no grain, the output should be near PAPER_COLOR ≈ (0.96, 0.93, 0.88).
        // In u16: R ≈ 62914, G ≈ 60948, B ≈ 57671. Allow ±1000 for f16 rounding.
        let mean_r: f64 = out.pixels().map(|p| p[0] as f64).sum::<f64>() / (16.0 * 16.0);
        assert!(
            mean_r > 55000.0,
            "solid image at full strength should produce near-cream-paper output (R > 55000), \
             got mean_r={mean_r:.0}"
        );
    }

    /// On an image with a sharp edge, at full strength the pixels at the edge boundary
    /// must be noticeably darker than those in the flat (paper) region.
    #[test]
    fn test_charcoal_sketch_edge_pixels_darker_than_flat_pixels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32×16 step image: left half dark, right half bright.
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 10000 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "charcoal_sketch",
                values: vec![1.0, 2.5, 0.0],
            }],
        );

        // A pixel near the edge boundary (x=15) should be darker than a pixel
        // well inside the flat region (x=2, far from the step).
        let edge_pixel = out.get_pixel(15, 8)[0] as i32;
        let flat_pixel = out.get_pixel(2, 8)[0] as i32;
        assert!(
            edge_pixel < flat_pixel,
            "edge pixel (x=15) must be darker (charcoal stroke) than flat-region pixel (x=2): \
             edge={edge_pixel}, flat={flat_pixel}"
        );
    }

    /// Higher edge_strength must increase the darkness of strokes, making the overall
    /// output darker (more charcoal coverage) on a gradient image.
    #[test]
    fn test_charcoal_sketch_higher_edge_strength_increases_darkness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Gradient image: shallow ramp produces a weak, uniform Sobel signal.
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v = (x * 2000) as u16;
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "charcoal_sketch",
                values: vec![1.0, 1.0, 0.0],
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "charcoal_sketch",
                values: vec![1.0, 5.0, 0.0],
            }],
        );

        // Higher edge_strength amplifies the Sobel magnitude → more dark strokes → lower mean.
        let mean_low: f64 = out_low.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 16.0);
        let mean_high: f64 = out_high.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 16.0);
        assert!(
            mean_high < mean_low,
            "higher edge_strength must produce a darker (lower mean R) output: \
             low={mean_low:.0}, high={mean_high:.0}"
        );
    }

    /// Increasing grain_amount must change the output compared to grain_amount=0.
    /// The grain introduces per-pixel variation that shifts mean brightness downward.
    #[test]
    fn test_charcoal_sketch_grain_amount_changes_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 32767, 32767);

        let out_no_grain = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "charcoal_sketch",
                values: vec![1.0, 2.5, 0.0],
            }],
        );
        let out_with_grain = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "charcoal_sketch",
                values: vec![1.0, 2.5, 1.0],
            }],
        );

        // At least one pixel must differ between the no-grain and max-grain variants.
        let any_different = out_no_grain
            .pixels()
            .zip(out_with_grain.pixels())
            .any(|(ng, wg)| (ng[0] as i32 - wg[0] as i32).abs() > 64);
        assert!(
            any_different,
            "grain_amount=1.0 output must differ from grain_amount=0.0"
        );
    }

    /// Grain is dark (subtractive): applying grain must lower mean brightness
    /// compared to the no-grain version.
    #[test]
    fn test_charcoal_sketch_grain_darkens_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 32767, 32767);

        let out_no_grain = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "charcoal_sketch",
                values: vec![1.0, 2.5, 0.0],
            }],
        );
        let out_max_grain = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "charcoal_sketch",
                values: vec![1.0, 2.5, 1.0],
            }],
        );

        let mean_no_grain: f64 =
            out_no_grain.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 32.0);
        let mean_max_grain: f64 =
            out_max_grain.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 32.0);
        assert!(
            mean_max_grain < mean_no_grain,
            "max grain must darken the output (lower mean R): \
             no_grain={mean_no_grain:.0}, max_grain={mean_max_grain:.0}"
        );
    }

    /// Chaining charcoal_sketch with brightness must not panic and must preserve alpha.
    #[test]
    fn test_charcoal_sketch_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "charcoal_sketch",
                    values: vec![0.5, 2.5, 0.5],
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

    /// Running Charcoal Sketch twice with identical inputs must produce bit-identical output.
    /// The procedural grain is coordinate-based and deterministic — no randomness per-frame.
    #[test]
    fn test_charcoal_sketch_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "charcoal_sketch",
            values: vec![0.8, 2.5, 0.5],
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
