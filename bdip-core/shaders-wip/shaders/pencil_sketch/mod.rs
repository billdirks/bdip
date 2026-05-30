use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters shared across both Pencil Sketch passes.
///
/// The two meaningful fields pack into 8 bytes; two padding floats bring the
/// struct to 16 bytes to satisfy WebGPU's uniform alignment requirement.
///
/// # Identity design
///
/// The spec requires that default parameter values produce a no-op transformation.
/// For Pencil Sketch this is not literally achievable: the effect converts the image
/// to a greyscale sketch at any non-zero edge_strength. The closest "no visible
/// change" default is `edge_strength = 0.0`, which makes the Sobel magnitude zero
/// everywhere and outputs a fully white image — but a white image is not the same
/// as the source.
///
/// The chosen design follows the same pattern as Stained Glass and other artistic
/// effects: a `strength` blend parameter defaults to `0.0`, which passes the source
/// image through unchanged (identity), while the artistic parameters (`edge_strength`
/// and `stroke_softness`) control the look when `strength` is non-zero. At
/// `strength = 0.0` the output equals the source regardless of the other sliders.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PencilSketchParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full pencil-sketch effect.
    pub strength: f32,
    /// Multiplier applied to raw Sobel gradient magnitude before clamping.
    /// Higher values make weak edges visible. Range [0.1, 10.0].
    pub edge_strength: f32,
    /// Controls the spatial extent of the directional stroke blur.
    /// 0.0 = no blur (hard edges), 1.0 = maximum stroke softness. Range [0.0, 1.0].
    pub stroke_softness: f32,
    pub _padding: f32,
}

impl TransformShader for PencilSketchParams {
    const ID: &'static str = "pencil_sketch";
    const DISPLAY_NAME: &'static str = "Pencil Sketch";
    const DESCRIPTION: &'static str = "Converts the image into a hand-drawn pencil sketch on white paper using \
         Sobel edge detection followed by directional blur along stroke orientation.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full pencil-sketch \
                          effect (1.0). The identity value is 0.0.",
        },
        SliderDef {
            name: "Edge Strength",
            min: 0.1,
            max: 10.0,
            default: 2.0,
            description: "Sensitivity of edge detection. Higher values make faint edges more \
                          visible as pencil lines.",
        },
        SliderDef {
            name: "Stroke Softness",
            min: 0.0,
            max: 1.0,
            default: 0.3,
            description: "Softness of pencil strokes via directional blur along the edge \
                          orientation. 0.0 produces hard edges; higher values simulate softer \
                          pencil strokes.",
        },
    ]);

    // Two-pass pipeline:
    //   Pass 1 — edges:  grayscale + Sobel edge detection → scratch texture.
    //   Pass 2 — stroke: directional blur along stroke direction + white-paper inversion
    //                    → final output blended with source.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "edges",
            wgsl_source: include_str!("pencil_sketch_edges.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("edges"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "stroke",
            wgsl_source: include_str!("pencil_sketch_stroke.wgsl"),
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
            stroke_softness: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    PencilSketchParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_pencil_sketch_registry_entry_exists() {
        assert!(registry_by_id("pencil_sketch").is_some());
    }

    #[test]
    fn test_pencil_sketch_registry_metadata() {
        let reg = registry_by_id("pencil_sketch").unwrap();
        assert_eq!(reg.meta.display_name, "Pencil Sketch");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full pencil-sketch \
                                  effect (1.0). The identity value is 0.0.",
                },
                SliderDef {
                    name: "Edge Strength",
                    min: 0.1,
                    max: 10.0,
                    default: 2.0,
                    description: "Sensitivity of edge detection. Higher values make faint edges more \
                                  visible as pencil lines.",
                },
                SliderDef {
                    name: "Stroke Softness",
                    min: 0.0,
                    max: 1.0,
                    default: 0.3,
                    description: "Softness of pencil strokes via directional blur along the edge \
                                  orientation. 0.0 produces hard edges; higher values simulate softer \
                                  pencil strokes.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "Pencil Sketch must have exactly 2 passes"
        );
    }

    #[test]
    fn test_pencil_sketch_make_uniform_known_value() {
        let reg = registry_by_id("pencil_sketch").unwrap();
        let bytes = (reg.make_uniform)(&[0.75, 3.0, 0.5]);
        let expected = bytemuck::bytes_of(&PencilSketchParams {
            strength: 0.75,
            edge_strength: 3.0,
            stroke_softness: 0.5,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 the stroke pass reduces to mix(src, sketch, 0.0) = src,
    /// so the output must equal the source regardless of edge_strength or stroke_softness.
    #[test]
    fn test_pencil_sketch_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pencil_sketch",
                values: vec![0.0, 2.0, 0.3],
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
    fn test_pencil_sketch_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pencil_sketch",
                values: vec![1.0, 2.0, 0.3],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// A uniform (solid-colour) image has no edges; at full strength the output
    /// should be near-white (no pencil lines to draw on a solid surface).
    #[test]
    fn test_pencil_sketch_solid_image_produces_white() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Mid-gray solid image — Sobel returns zero on a constant input.
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pencil_sketch",
                values: vec![1.0, 2.0, 0.0],
            }],
        );
        // sketch_value = 1 - 0 = 1 → u16 = 65535. Allow ±200 for f16 rounding.
        for pixel in out.pixels() {
            assert!(
                pixel[0] > 60000,
                "R on solid image: expected near-white (~65535), got {}",
                pixel[0]
            );
        }
    }

    /// On an image with a sharp edge, at full strength the pixels on the edge boundary
    /// must be noticeably darker than those in the flat region (where Sobel ≈ 0).
    #[test]
    fn test_pencil_sketch_edge_pixels_darker_than_flat_pixels() {
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
                shader_id: "pencil_sketch",
                values: vec![1.0, 2.0, 0.0],
            }],
        );

        // A pixel near the edge (x=15 or x=16) should be darker than a
        // pixel well inside the flat region (x=2, far from the step).
        let edge_pixel = out.get_pixel(15, 8)[0] as i32;
        let flat_pixel = out.get_pixel(2, 8)[0] as i32;
        assert!(
            edge_pixel < flat_pixel,
            "edge pixel (x=15) must be darker than flat-region pixel (x=2): \
             edge={edge_pixel}, flat={flat_pixel}"
        );
    }

    /// Increasing stroke_softness must change the output compared to softness=0.
    /// Directional blur along the stroke direction smooths the edge lines.
    #[test]
    fn test_pencil_sketch_stroke_softness_changes_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Step image to produce a measurable edge.
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 10000 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_hard = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pencil_sketch",
                values: vec![1.0, 2.0, 0.0],
            }],
        );
        let out_soft = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pencil_sketch",
                values: vec![1.0, 2.0, 1.0],
            }],
        );

        // At least one pixel must differ between the hard and soft variants.
        let any_different = out_hard
            .pixels()
            .zip(out_soft.pixels())
            .any(|(h, s)| (h[0] as i32 - s[0] as i32).abs() > 64);
        assert!(
            any_different,
            "stroke_softness=1.0 output must differ from stroke_softness=0.0"
        );
    }

    /// Increasing edge_strength from 1.0 to 5.0 must produce more (or equally many)
    /// dark pixels, since faint gradients are amplified into visible lines.
    #[test]
    fn test_pencil_sketch_higher_edge_strength_increases_darkness() {
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
                shader_id: "pencil_sketch",
                values: vec![1.0, 1.0, 0.0],
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pencil_sketch",
                values: vec![1.0, 5.0, 0.0],
            }],
        );

        // Mean brightness must be lower at higher edge_strength (more dark lines).
        let mean_low: f64 = out_low.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 16.0);
        let mean_high: f64 = out_high.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 16.0);
        assert!(
            mean_high < mean_low,
            "higher edge_strength must produce a darker (lower mean) output: \
             low={mean_low:.0}, high={mean_high:.0}"
        );
    }

    /// Chaining pencil_sketch with brightness must not panic and must preserve alpha.
    #[test]
    fn test_pencil_sketch_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "pencil_sketch",
                    values: vec![0.5, 2.0, 0.3],
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

    /// Running Pencil Sketch twice with identical inputs must produce bit-identical output.
    #[test]
    fn test_pencil_sketch_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "pencil_sketch",
            values: vec![0.8, 2.0, 0.3],
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
