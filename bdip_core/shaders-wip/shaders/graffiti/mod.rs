use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters shared across both Graffiti passes.
///
/// Four floats pack into 16 bytes, satisfying WebGPU uniform alignment.
///
/// # Identity design
///
/// Like Pencil Sketch and other artistic effects, a pure identity transformation is not
/// achievable for the graffiti look (color quantization always changes the image). The
/// `strength` parameter defaults to 0.0, which passes the source image through unchanged
/// via a linear blend in the final pass. At `strength = 0.0` the output equals the source
/// regardless of the other slider values.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GraffitiParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full graffiti effect.
    pub strength: f32,
    /// Number of quantization levels per channel. Range [2, 16].
    /// Lower values produce coarser, bolder color zones.
    pub color_levels: f32,
    /// Multiplier applied to Sobel edge magnitude for edge darkening.
    /// Higher values produce thicker, bolder outlines. Range [0.5, 5.0].
    pub edge_strength: f32,
    /// Spatial extent of the spray-bleed blur relative to image size.
    /// 0.0 = no blur, 1.0 = maximum bleed. Range [0.0, 1.0].
    pub bleed: f32,
}

impl TransformShader for GraffitiParams {
    const ID: &'static str = "graffiti";
    const DISPLAY_NAME: &'static str = "Graffiti";
    const DESCRIPTION: &'static str = "Makes the image resemble spray-painted graffiti via color quantization, \
         edge darkening, and a spray-bleed blur — all achieved procedurally.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full graffiti \
                          effect (1.0). The identity value is 0.0.",
        },
        SliderDef {
            name: "Color Levels",
            min: 2.0,
            max: 16.0,
            default: 6.0,
            description: "Number of quantization levels per channel. Lower values produce \
                          broader, flatter color zones for a bolder graffiti look.",
        },
        SliderDef {
            name: "Edge Strength",
            min: 0.5,
            max: 5.0,
            default: 2.0,
            description: "Multiplier on Sobel edge magnitude. Higher values produce thicker, \
                          darker outlines typical of spray-paint stencil work.",
        },
        SliderDef {
            name: "Bleed",
            min: 0.0,
            max: 1.0,
            default: 0.3,
            description: "Spatial extent of spray-paint bleed. Higher values soften edges \
                          further, simulating diffuse overspray.",
        },
    ]);

    // Three-pass pipeline:
    //   Pass 1a — bleed_h:  horizontal box-blur of the source → scratch texture.
    //   Pass 1b — bleed_v:  vertical box-blur of bleed_h → scratch texture.
    //                       Together these two passes form a separable 2D box blur
    //                       that simulates isotropic spray-paint overspray.
    //   Pass 2  — quantize: posterize colors, darken edges via Sobel, blend with source.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "bleed_h",
            wgsl_source: include_str!("graffiti_bleed.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("bleed_h"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "bleed_v",
            wgsl_source: include_str!("graffiti_bleed_v.wgsl"),
            inputs: &[PassInput::Scratch("bleed_h")],
            output: PassOutput::Scratch("bleed"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "quantize",
            wgsl_source: include_str!("graffiti_quantize.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("bleed")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            color_levels: values[1],
            edge_strength: values[2],
            bleed: values[3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<GraffitiParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_graffiti_registry_entry_exists() {
        assert!(registry_by_id("graffiti").is_some());
    }

    #[test]
    fn test_graffiti_registry_metadata() {
        let reg = registry_by_id("graffiti").unwrap();
        assert_eq!(reg.meta.display_name, "Graffiti");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full graffiti \
                                  effect (1.0). The identity value is 0.0.",
                },
                SliderDef {
                    name: "Color Levels",
                    min: 2.0,
                    max: 16.0,
                    default: 6.0,
                    description: "Number of quantization levels per channel. Lower values produce \
                                  broader, flatter color zones for a bolder graffiti look.",
                },
                SliderDef {
                    name: "Edge Strength",
                    min: 0.5,
                    max: 5.0,
                    default: 2.0,
                    description: "Multiplier on Sobel edge magnitude. Higher values produce thicker, \
                                  darker outlines typical of spray-paint stencil work.",
                },
                SliderDef {
                    name: "Bleed",
                    min: 0.0,
                    max: 1.0,
                    default: 0.3,
                    description: "Spatial extent of spray-paint bleed. Higher values soften edges \
                                  further, simulating diffuse overspray.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            3,
            "Graffiti must have exactly 3 passes"
        );
    }

    #[test]
    fn test_graffiti_make_uniform_known_value() {
        let reg = registry_by_id("graffiti").unwrap();
        let bytes = (reg.make_uniform)(&[0.8, 4.0, 3.0, 0.5]);
        let expected = bytemuck::bytes_of(&GraffitiParams {
            strength: 0.8,
            color_levels: 4.0,
            edge_strength: 3.0,
            bleed: 0.5,
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0 the quantize pass reduces to mix(src, graffiti, 0.0) = src,
    /// so the output must equal the source regardless of other parameters.
    #[test]
    fn test_graffiti_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "graffiti",
                values: vec![0.0, 6.0, 2.0, 0.3],
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
    fn test_graffiti_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "graffiti",
                values: vec![1.0, 6.0, 2.0, 0.3],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// A uniform (solid-color) image has no edges; at full strength the output
    /// should be close to the quantized version of the input color (no darkening
    /// from edges on a flat surface).
    #[test]
    fn test_graffiti_solid_image_no_edge_darkening() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Mid-gray solid — Sobel returns zero everywhere.
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "graffiti",
                values: vec![1.0, 6.0, 2.0, 0.0],
            }],
        );
        // With no edges the output must not be significantly darker than the
        // quantized value. Quantized mid-gray with 6 levels ~ floor(0.5 * 6) / 6
        // = 3/6 = 0.5 → u16 ≈ 32768. Allow ±4000 for quantization step.
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() < 4000,
                "R on solid image: expected near quantized mid-gray, got {}",
                pixel[0]
            );
        }
    }

    /// On a step-edge image, pixels at the boundary must be darker than pixels
    /// in the flat region when edge_strength is high and strength=1.
    #[test]
    fn test_graffiti_edge_pixels_darker_than_flat_region() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32×16 step: left half mid-gray, right half bright.
        // Values chosen so that both halves quantize to a non-zero level
        // with 6 color levels, leaving room for edge darkening to reduce them.
        // 32767/65535 ≈ 0.500 → floor(0.500 * 6) / 6 = 3/6 ≈ 32767 (non-zero).
        // 55000/65535 ≈ 0.839 → floor(0.839 * 6) / 6 = 5/6 ≈ 54612 (non-zero).
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 32767 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "graffiti",
                values: vec![1.0, 6.0, 5.0, 0.0],
            }],
        );

        // Pixel near edge (x=15) should be darker than pixel well inside
        // the flat dark region (x=4).
        let edge_pixel = out.get_pixel(15, 8)[0] as i32;
        let flat_pixel = out.get_pixel(4, 8)[0] as i32;
        assert!(
            edge_pixel < flat_pixel,
            "edge pixel (x=15) must be darker than flat pixel (x=4): \
             edge={edge_pixel}, flat={flat_pixel}"
        );
    }

    /// Fewer color levels must produce more visible color banding than more levels
    /// on a gradient image (higher mean absolute deviation from the original).
    #[test]
    fn test_graffiti_fewer_levels_produce_more_quantization() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Horizontal gradient covering the full [0, 65535] range.
        let mut img = crate::Rgba16Image::new(32, 4);
        for y in 0..4u32 {
            for x in 0..32u32 {
                let v = (x * 65535 / 31) as u16;
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_coarse = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "graffiti",
                values: vec![1.0, 2.0, 0.5, 0.0],
            }],
        );
        let out_fine = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "graffiti",
                values: vec![1.0, 16.0, 0.5, 0.0],
            }],
        );

        // Coarse quantization produces larger deviations from the source than fine.
        let mad_coarse: f64 = img
            .pixels()
            .zip(out_coarse.pixels())
            .map(|(s, o)| (s[0] as i64 - o[0] as i64).unsigned_abs() as f64)
            .sum::<f64>();
        let mad_fine: f64 = img
            .pixels()
            .zip(out_fine.pixels())
            .map(|(s, o)| (s[0] as i64 - o[0] as i64).unsigned_abs() as f64)
            .sum::<f64>();

        assert!(
            mad_coarse > mad_fine,
            "2 levels must deviate more from source than 16 levels: \
             coarse_mad={mad_coarse:.0}, fine_mad={mad_fine:.0}"
        );
    }

    /// Increasing bleed must change the output compared to bleed=0 on a step image.
    ///
    /// The bleed pass blurs the source before quantization. On a step image,
    /// pixels near the boundary receive blurred intermediate values, which can
    /// land in a different quantization bin than the original sharp-edge values.
    /// The comparison uses a large image (128 wide) so the blur radius is non-zero
    /// at bleed=1.0 (BLEED_FRACTION * 128 = ~1.9 px → radius=2 at ceil).
    #[test]
    fn test_graffiti_bleed_changes_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 128×16 step: left half at a quantization boundary so that even a
        // small amount of blur shifts boundary pixels into a different bin.
        // 21845 / 65535 ≈ 0.333 → floor(0.333 * 6) / 6 = 1/6 ≈ 10922
        // 43690 / 65535 ≈ 0.667 → floor(0.667 * 6) / 6 = 4/6 ≈ 43690
        // Blurred transition pixels land between these bins and quantize differently.
        let mut img = crate::Rgba16Image::new(128, 16);
        for y in 0..16u32 {
            for x in 0..128u32 {
                let v: u16 = if x < 64 { 21845 } else { 43690 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_no_bleed = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "graffiti",
                values: vec![1.0, 6.0, 0.5, 0.0],
            }],
        );
        let out_bleed = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "graffiti",
                values: vec![1.0, 6.0, 0.5, 1.0],
            }],
        );

        // At least one pixel must differ between the two runs.
        let any_different = out_no_bleed
            .pixels()
            .zip(out_bleed.pixels())
            .any(|(a, b)| (a[0] as i32 - b[0] as i32).abs() > 64);
        assert!(
            any_different,
            "bleed=1.0 output must differ from bleed=0.0 output"
        );
    }

    /// Chaining graffiti with brightness must not panic and must preserve alpha.
    #[test]
    fn test_graffiti_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "graffiti",
                    values: vec![0.5, 6.0, 2.0, 0.3],
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

    /// A single bright pixel should blur symmetrically in both axes, not just horizontally.
    ///
    /// Before the two-pass fix the bleed was horizontal-only; this test verifies the
    /// vertical pass is now contributing equally.
    #[test]
    fn test_graffiti_blur_is_isotropic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 64×64 black image with a single bright pixel at center.
        let mut img = crate::Rgba16Image::new(64, 64);
        for y in 0..64u32 {
            for x in 0..64u32 {
                img.put_pixel(x, y, image::Rgba([0, 0, 0, 65535]));
            }
        }
        img.put_pixel(32, 32, image::Rgba([65535, 65535, 65535, 65535]));

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "graffiti",
                values: vec![1.0, 16.0, 0.5, 1.0],
            }],
        );

        // Equal-distance neighbours in X and Y must have comparable brightness.
        let horizontal_neighbor = out.get_pixel(32 + 5, 32)[0] as i32;
        let vertical_neighbor = out.get_pixel(32, 32 + 5)[0] as i32;
        let diff = (horizontal_neighbor - vertical_neighbor).abs();
        assert!(
            diff < 5000,
            "blur should be isotropic: horizontal neighbor={}, vertical neighbor={}, diff={}",
            horizontal_neighbor,
            vertical_neighbor,
            diff
        );
    }

    /// Bleed must soften horizontal edges (top-to-bottom transitions), which only a
    /// vertical blur can accomplish.  The pre-fix horizontal-only blur left these edges
    /// entirely unaffected.
    #[test]
    fn test_graffiti_bleed_affects_horizontal_edges() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 128×64 step: top half dark, bottom half bright — a horizontal edge.
        let mut img = crate::Rgba16Image::new(128, 64);
        for y in 0..64u32 {
            for x in 0..128u32 {
                let v: u16 = if y < 32 { 21845 } else { 43690 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_no_bleed = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "graffiti",
                values: vec![1.0, 6.0, 0.5, 0.0],
            }],
        );
        let out_bleed = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "graffiti",
                values: vec![1.0, 6.0, 0.5, 1.0],
            }],
        );

        // At least one pixel near the horizontal boundary must differ between runs.
        let any_different = (0..128u32).any(|x| {
            let a = out_no_bleed.get_pixel(x, 31)[0] as i32;
            let b = out_bleed.get_pixel(x, 31)[0] as i32;
            (a - b).abs() > 64
        });
        assert!(
            any_different,
            "bleed should affect horizontal edges (vertical blur must work)"
        );
    }

    /// Running Graffiti twice with identical inputs must produce bit-identical output.
    #[test]
    fn test_graffiti_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "graffiti",
            values: vec![0.8, 6.0, 2.0, 0.3],
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
