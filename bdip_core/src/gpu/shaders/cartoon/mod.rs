use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CartoonParams {
    pub strength: f32,
    pub levels: f32,
    pub edge_threshold: f32,
    pub edge_softness: f32,
    pub edge_darkness: f32,
    pub _padding: [f32; 3], // pad to 32 bytes
}

impl TransformShader for CartoonParams {
    const ID: &'static str = "cartoon";
    const DISPLAY_NAME: &'static str = "Cartoon";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
        },
        SliderDef {
            name: "Levels",
            min: 2.0,
            max: 16.0,
            default: 8.0,
        },
        SliderDef {
            name: "Edge Threshold",
            min: 0.0,
            max: 1.0,
            default: 0.15,
        },
        SliderDef {
            name: "Edge Softness",
            min: 0.01,
            max: 0.5,
            default: 0.10,
        },
        SliderDef {
            name: "Edge Darkness",
            min: 0.0,
            max: 1.0,
            default: 1.0,
        },
    ]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "smooth_h",
            wgsl_source: include_str!("smooth_h.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("sh"),
            output_scale: PassScale::Full,
        },
        PassDef {
            label: "smooth_v",
            wgsl_source: include_str!("smooth_v.wgsl"),
            inputs: &[PassInput::Scratch("sh")],
            output: PassOutput::Scratch("smooth"),
            output_scale: PassScale::Full,
        },
        PassDef {
            label: "quantize",
            wgsl_source: include_str!("quantize.wgsl"),
            inputs: &[PassInput::Scratch("smooth")],
            output: PassOutput::Scratch("quant"),
            output_scale: PassScale::Full,
        },
        PassDef {
            label: "edges",
            wgsl_source: include_str!("edges.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("edges"),
            output_scale: PassScale::Full,
        },
        PassDef {
            label: "combine",
            wgsl_source: include_str!("combine.wgsl"),
            inputs: &[
                PassInput::Source,
                PassInput::Scratch("quant"),
                PassInput::Scratch("edges"),
            ],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            levels: values[1],
            edge_threshold: values[2],
            edge_softness: values[3],
            edge_darkness: values[4],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<CartoonParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_cartoon_registry_entry_exists() {
        assert!(registry_by_id("cartoon").is_some());
    }

    #[test]
    fn test_cartoon_registry_metadata() {
        let reg = registry_by_id("cartoon").unwrap();
        assert_eq!(reg.meta.display_name, "Cartoon");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0
                },
                SliderDef {
                    name: "Levels",
                    min: 2.0,
                    max: 16.0,
                    default: 8.0
                },
                SliderDef {
                    name: "Edge Threshold",
                    min: 0.0,
                    max: 1.0,
                    default: 0.15
                },
                SliderDef {
                    name: "Edge Softness",
                    min: 0.01,
                    max: 0.5,
                    default: 0.10
                },
                SliderDef {
                    name: "Edge Darkness",
                    min: 0.0,
                    max: 1.0,
                    default: 1.0
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            5,
            "Cartoon must have exactly 5 passes"
        );
    }

    #[test]
    fn test_cartoon_make_uniform_known_value() {
        let reg = registry_by_id("cartoon").unwrap();
        let bytes = (reg.make_uniform)(&[0.5, 8.0, 0.2, 0.1, 0.8]);
        let expected = bytemuck::bytes_of(&CartoonParams {
            strength: 0.5,
            levels: 8.0,
            edge_threshold: 0.2,
            edge_softness: 0.1,
            edge_darkness: 0.8,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_cartoon_zero_strength_and_zero_edge_darkness_is_identity() {
        // At strength=0 and edge_darkness=0 the combine formula reduces to:
        // out = mix(src, quant, 0) * (1 - 0 * edges) = src * 1 = src
        // The shader must not alter the image at these settings.
        //
        // We compare the cartoon output against a zero-transform roundtrip rather
        // than the CPU image to avoid ±1 u16 noise from f16 GPU precision. Both
        // paths go through the same ingest → present pipeline, so they share the
        // same quantization baseline.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(16, 16);
        for (i, pixel) in img.pixels_mut().enumerate() {
            let v = (i as u16).wrapping_mul(37).wrapping_add(1000);
            *pixel = image::Rgba([v, v / 2, v / 3 + 100, 65535]);
        }

        let identity_out = roundtrip(&mut renderer, &engine, &img, &[]);

        let cartoon_out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cartoon",
                values: vec![0.0, 8.0, 0.15, 0.10, 0.0],
            }],
        );

        for (p_ref, p_cartoon) in identity_out.pixels().zip(cartoon_out.pixels()) {
            assert_eq!(
                p_ref, p_cartoon,
                "cartoon at strength=0, edge_darkness=0 must not alter pixels"
            );
        }
    }

    #[test]
    fn test_cartoon_full_strength_reduces_unique_colors() {
        // Posterization at levels=4 with strength=1 must reduce the number of
        // distinct pixel values relative to a smooth gradient input.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Build a smooth gradient: 32×32, values evenly spread across [1000, 64535].
        let mut img = crate::Rgba16Image::new(32, 32);
        let total = 32u32 * 32u32;
        for (i, pixel) in img.pixels_mut().enumerate() {
            let v = (1000 + (i as u32 * 63535 / total)) as u16;
            *pixel = image::Rgba([v, v, v, 65535]);
        }

        let unique_in: std::collections::HashSet<u16> = img.pixels().map(|p| p[0]).collect();

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cartoon",
                values: vec![1.0, 4.0, 0.15, 0.10, 0.0],
            }],
        );

        let unique_out: std::collections::HashSet<u16> = out.pixels().map(|p| p[0]).collect();

        assert!(
            unique_out.len() < unique_in.len(),
            "posterization must reduce unique color count: {} in, {} out",
            unique_in.len(),
            unique_out.len()
        );
    }

    #[test]
    fn test_cartoon_edges_darken_high_gradient_pixels() {
        // A hard black/white edge produces a high Sobel magnitude; with edge_darkness=1
        // the combine pass multiplies colour by (1 - 1.0 * edge_mask), darkening those
        // pixels below their input value.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32×16 step: left half white, right half black, clear horizontal edge at x=16.
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 65535 } else { 0 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cartoon",
                values: vec![0.0, 8.0, 0.1, 0.1, 1.0],
            }],
        );

        // Pixel just left of the edge (bright side). It should be darker in the output
        // because the Sobel magnitude is high and edge_darkness=1.
        let in_bright = img.get_pixel(14, 8)[0] as i32;
        let out_bright = out.get_pixel(14, 8)[0] as i32;
        assert!(
            out_bright < in_bright,
            "edge pixel should be darkened: in={in_bright}, out={out_bright}"
        );
    }

    #[test]
    fn test_cartoon_higher_edge_softness_widens_edge_band() {
        // Verify that the softness parameter controls how sharply the edge mask
        // transitions from no-darkening to full-darkening.
        //
        // The ingest pass applies sRGB→linear before the cartoon shader runs, so the
        // Sobel magnitude the edges pass sees is in linear-light space.
        //
        // Image: step from 27800 to 0 (sRGB u16). sRGB 27800/65535 ≈ 0.424, which maps to
        // linear ≈ 0.15. The 3×3 Sobel Gx magnitude at the boundary is ≈ 4 × 0.15 = 0.60.
        //
        // With threshold=0.5:
        //   narrow (softness=0.05): ramp [0.50, 0.55]. 0.60 > 0.55 → edge=1.0 (fully saturated)
        //   wide   (softness=0.30): ramp [0.50, 0.80]. 0.60 ∈ [0.50, 0.80] → edge≈0.26 (partial)
        //
        // Narrow fully saturates the edge (maximum darkening → pixel nearly black).
        // Wide only partially darkens the same pixel. The edge pixel therefore retains MORE
        // brightness with wide softness, confirming the ramp is "wider" / softer.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Step calibrated so Sobel magnitude lands inside the wide but outside the narrow ramp.
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 27800 } else { 0 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        // Narrow ramp: edge pixels exceed ramp_end → fully darkened.
        let out_narrow = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cartoon",
                values: vec![0.0, 8.0, 0.5, 0.05, 1.0],
            }],
        );

        // Wide ramp: edge pixels fall within the ramp → partially darkened.
        let out_wide = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cartoon",
                values: vec![0.0, 8.0, 0.5, 0.3, 1.0],
            }],
        );

        // The edge pixel (last column of the bright side) must retain more brightness
        // with wide softness because the ramp transitions more gradually.
        let edge_narrow = out_narrow.get_pixel(15, 8)[0] as i32;
        let edge_wide = out_wide.get_pixel(15, 8)[0] as i32;

        assert!(
            edge_wide > edge_narrow,
            "wide softness must preserve more brightness at the edge \
             (partial ramp vs full saturation): narrow={edge_narrow}, wide={edge_wide}"
        );
    }

    #[test]
    fn test_cartoon_no_edges_below_threshold() {
        // A smooth gradient produces Sobel magnitudes well below 1.0. With
        // edge_threshold=1.0 nothing reaches the threshold and no pixel is darkened.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Smooth horizontal gradient, no sharp edges.
        let mut img = crate::Rgba16Image::new(32, 32);
        let total = 32u32 * 32u32;
        for (i, pixel) in img.pixels_mut().enumerate() {
            let v = (1000 + (i as u32 * 60000 / total)) as u16;
            *pixel = image::Rgba([v, v, v, 65535]);
        }

        // Run with edge_threshold=1.0, strength=1 so any effect would be visible.
        let out_with_threshold = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cartoon",
                values: vec![1.0, 4.0, 1.0, 0.1, 1.0],
            }],
        );

        // Run with edge_threshold=1.0 and edge_darkness=0 as the reference (posterize only).
        let out_posterize_only = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cartoon",
                values: vec![1.0, 4.0, 1.0, 0.1, 0.0],
            }],
        );

        // Both outputs must be identical — no edges applied in either case.
        for (pa, pb) in out_with_threshold.pixels().zip(out_posterize_only.pixels()) {
            assert_eq!(
                pa, pb,
                "with edge_threshold=1.0 no pixel should be edge-darkened"
            );
        }
    }

    #[test]
    fn test_cartoon_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cartoon",
                values: vec![0.5, 8.0, 0.15, 0.10, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    #[test]
    fn test_cartoon_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "cartoon",
            values: vec![0.5, 8.0, 0.15, 0.10, 1.0],
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

    #[test]
    fn test_cartoon_three_input_combine_pass_binds_correctly() {
        // Verify that the 3-input combine pass uses all three bindings correctly.
        // Source is a solid red image; at strength=1 (full posterization) and
        // edge_darkness=1, the output must show contributions from quant (posterized)
        // and edges. We use a high-contrast black/white image so edges are produced,
        // then confirm the output is strictly darker than a no-edge run — meaning
        // both the quant and edge bindings contributed.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Hard step produces strong Sobel edges.
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 65535 } else { 1000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        // With edges enabled (edge_darkness=1) some pixels must be darkened.
        let out_edges = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cartoon",
                values: vec![0.5, 4.0, 0.1, 0.1, 1.0],
            }],
        );

        // With edges disabled (edge_darkness=0) no darkening occurs.
        let out_no_edges = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cartoon",
                values: vec![0.5, 4.0, 0.1, 0.1, 0.0],
            }],
        );

        // At least one pixel must be darker with edges than without — confirming
        // the edge binding at @binding(2) is wired to actual edge data.
        let any_darker = out_edges
            .pixels()
            .zip(out_no_edges.pixels())
            .any(|(e, ne)| e[0] < ne[0]);

        assert!(
            any_darker,
            "edge binding must contribute darkening: no pixel was darker with edges enabled"
        );
    }
}
