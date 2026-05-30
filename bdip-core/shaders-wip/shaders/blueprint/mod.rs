use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlueprintParams {
    pub strength: f32,
    pub edge_threshold: f32,
    pub edge_thickness: f32,
    pub _padding: f32,
}

impl TransformShader for BlueprintParams {
    const ID: &'static str = "blueprint";
    const DISPLAY_NAME: &'static str = "Blueprint";
    const DESCRIPTION: &'static str = "Renders the image as a technical blueprint: deep blue background with bright \
         structural lines derived from edge detection and inverted luminance.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0, // Identity: no effect until the user moves the slider.
            description: "Blend factor between the original image and the blueprint result.",
        },
        SliderDef {
            name: "Edge Threshold",
            min: 0.0,
            max: 1.0,
            default: 0.10,
            description: "Minimum Sobel gradient magnitude that triggers a structural edge line.",
        },
        SliderDef {
            name: "Edge Thickness",
            min: 0.01,
            max: 0.5,
            default: 0.15,
            description: "Controls the sharpness of the transition from unlined to outlined pixels.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "edges",
            wgsl_source: include_str!("blueprint_edges.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("edges"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "combine",
            wgsl_source: include_str!("blueprint_combine.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("edges")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            edge_threshold: values[1],
            edge_thickness: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    BlueprintParams,
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
    fn test_blueprint_registry_entry_exists() {
        assert!(registry_by_id("blueprint").is_some());
    }

    #[test]
    fn test_blueprint_registry_metadata() {
        let reg = registry_by_id("blueprint").unwrap();
        assert_eq!(reg.meta.display_name, "Blueprint");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend factor between the original image and the blueprint result.",
                },
                SliderDef {
                    name: "Edge Threshold",
                    min: 0.0,
                    max: 1.0,
                    default: 0.10,
                    description: "Minimum Sobel gradient magnitude that triggers a structural edge line.",
                },
                SliderDef {
                    name: "Edge Thickness",
                    min: 0.01,
                    max: 0.5,
                    default: 0.15,
                    description: "Controls the sharpness of the transition from unlined to outlined pixels.",
                },
            ])
        );
    }

    #[test]
    fn test_blueprint_passes_count() {
        let reg = registry_by_id("blueprint").unwrap();
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "Blueprint must have exactly 2 passes"
        );
    }

    #[test]
    fn test_blueprint_make_uniform_known_value() {
        let reg = registry_by_id("blueprint").unwrap();
        let bytes = (reg.make_uniform)(&[0.8, 0.05, 0.20]);
        let expected = bytemuck::bytes_of(&BlueprintParams {
            strength: 0.8,
            edge_threshold: 0.05,
            edge_thickness: 0.20,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the output must be within the Rgba16Float
    /// rounding tolerance of the input (±8 u16 units — the pipeline stores values
    /// as f16 in the edges scratch texture, which has ~3 decimal-digit precision).
    #[test]
    fn test_blueprint_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 15000, 8000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "blueprint",
                values: vec![0.0, 0.10, 0.15],
            }],
        );

        // The combine pass blends with mix(src.rgb, blueprint, 0.0) = src.rgb, but the
        // source values are re-read from the GPU texture which has Rgba16Float (f16)
        // precision. Allow ±8 u16 rounding units per channel.
        for (p_in, p_out) in img.pixels().zip(out.pixels()) {
            assert!(
                (p_in[0] as i32 - p_out[0] as i32).abs() <= 8,
                "R: strength=0 should be near-identity: in={} out={}",
                p_in[0],
                p_out[0]
            );
            assert!(
                (p_in[1] as i32 - p_out[1] as i32).abs() <= 8,
                "G: strength=0 should be near-identity: in={} out={}",
                p_in[1],
                p_out[1]
            );
            assert!(
                (p_in[2] as i32 - p_out[2] as i32).abs() <= 8,
                "B: strength=0 should be near-identity: in={} out={}",
                p_in[2],
                p_out[2]
            );
            assert_eq!(p_out[3], 65535, "alpha must be preserved");
        }
    }

    /// At full strength on a uniform (no-edge) image, R must be much less than B,
    /// confirming the blue-dominant output of the blueprint effect.
    #[test]
    fn test_blueprint_full_strength_produces_blue_dominant_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Uniform grey — no edges, so the output is purely the inverted-luminance
        // blue layer with no edge overlay.
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "blueprint",
                values: vec![1.0, 0.10, 0.15],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[2] > pixel[0],
                "B must exceed R in blueprint output: R={} B={}",
                pixel[0],
                pixel[2]
            );
        }
    }

    /// Alpha channel must not be modified regardless of the strength value.
    #[test]
    fn test_blueprint_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 30000, 20000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "blueprint",
                values: vec![1.0, 0.10, 0.15],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by blueprint");
        }
    }

    /// Edge pixels must appear brighter (nearer to white) than flat-field pixels
    /// when blueprint is applied at full strength, because Sobel edges are overlaid
    /// with a near-white tint.
    #[test]
    fn test_blueprint_edges_are_brighter_than_flat_field() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Build a 16-wide image with a sharp black/white boundary at x=8.
        let mut img = crate::Rgba16Image::new(16, 4);
        for y in 0..4u32 {
            for x in 0..16u32 {
                let v: u16 = if x < 8 { 0 } else { 65535 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "blueprint",
                values: vec![1.0, 0.05, 0.15],
            }],
        );

        // Pixel at the boundary (x=7, the last dark pixel) should be significantly
        // brighter after blueprint than the deep-field dark pixel at x=2.
        let boundary_brightness = out.get_pixel(7, 2)[0] as i32
            + out.get_pixel(7, 2)[1] as i32
            + out.get_pixel(7, 2)[2] as i32;
        let flat_brightness = out.get_pixel(2, 2)[0] as i32
            + out.get_pixel(2, 2)[1] as i32
            + out.get_pixel(2, 2)[2] as i32;

        assert!(
            boundary_brightness > flat_brightness,
            "edge pixel must be brighter than deep flat-field pixel: edge={boundary_brightness} \
             flat={flat_brightness}"
        );
    }

    /// Applying blueprint then brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_blueprint_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 15000, 10000, 5000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "blueprint",
                values: vec![0.7, 0.10, 0.15],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "blueprint",
                    values: vec![0.7, 0.10, 0.15],
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

    /// Increasing strength must move the output further from the original image.
    #[test]
    fn test_blueprint_higher_strength_moves_further_from_original() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);

        let out_low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "blueprint",
                values: vec![0.3, 0.10, 0.15],
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "blueprint",
                values: vec![1.0, 0.10, 0.15],
            }],
        );

        // The higher-strength output must differ more from the original (i32 to avoid u16 wrap).
        let src_val = 20000i32;
        let diff_low: i32 = out_low
            .pixels()
            .map(|p| (p[0] as i32 - src_val).abs() + (p[2] as i32 - src_val).abs())
            .sum();
        let diff_high: i32 = out_high
            .pixels()
            .map(|p| (p[0] as i32 - src_val).abs() + (p[2] as i32 - src_val).abs())
            .sum();

        assert!(
            diff_high > diff_low,
            "strength=1.0 must move pixels further from original than strength=0.3: \
             diff_high={diff_high} diff_low={diff_low}"
        );
    }
}
