use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters shared across all Stained Glass passes.
///
/// All three f32 fields are meaningful to the shader; the fourth is padding to
/// satisfy WebGPU's 16-byte uniform alignment requirement.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StainedGlassParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full stained-glass effect.
    pub strength: f32,
    /// Voronoi cell size as a fraction of the shorter image dimension. Larger values
    /// produce bigger, more visible cells. Range [0.01, 0.25].
    pub cell_size: f32,
    /// Relative width of the dark edge lines as a fraction of cell proximity range.
    /// Range [0.0, 1.0].
    pub edge_width: f32,
    pub _padding: f32,
}

impl TransformShader for StainedGlassParams {
    const ID: &'static str = "stained_glass";
    const DISPLAY_NAME: &'static str = "Stained Glass";
    const DESCRIPTION: &'static str = "Segments the image into coloured Voronoi cells with dark lead-came borders, \
         simulating a stained-glass window.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full stained-glass \
                          effect (1.0). The identity value is 0.0.",
        },
        SliderDef {
            name: "Cell Size",
            min: 0.01,
            max: 0.25,
            default: 0.08,
            description: "Size of each Voronoi cell as a fraction of the shorter image dimension. \
                          Larger values produce bigger glass panes.",
        },
        SliderDef {
            name: "Edge Width",
            min: 0.0,
            max: 1.0,
            default: 0.3,
            description: "Relative width of the dark lead-came lines between cells. \
                          0.0 removes the borders entirely.",
        },
    ]);

    // Two-pass pipeline:
    //   Pass 1 — voronoi: compute Voronoi cell colour + boundary proximity → scratch.
    //   Pass 2 — edges:   composite edge lines + blend with source → final output.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "voronoi",
            wgsl_source: include_str!("stained_glass_voronoi.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("voronoi"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "edges",
            wgsl_source: include_str!("stained_glass_edges.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("voronoi")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            cell_size: values[1],
            edge_width: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    StainedGlassParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_stained_glass_registry_entry_exists() {
        assert!(registry_by_id("stained_glass").is_some());
    }

    #[test]
    fn test_stained_glass_registry_metadata() {
        let reg = registry_by_id("stained_glass").unwrap();
        assert_eq!(reg.meta.display_name, "Stained Glass");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full stained-glass \
                                  effect (1.0). The identity value is 0.0.",
                },
                SliderDef {
                    name: "Cell Size",
                    min: 0.01,
                    max: 0.25,
                    default: 0.08,
                    description: "Size of each Voronoi cell as a fraction of the shorter image \
                                  dimension. Larger values produce bigger glass panes.",
                },
                SliderDef {
                    name: "Edge Width",
                    min: 0.0,
                    max: 1.0,
                    default: 0.3,
                    description: "Relative width of the dark lead-came lines between cells. \
                                  0.0 removes the borders entirely.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "Stained Glass must have exactly 2 passes"
        );
    }

    #[test]
    fn test_stained_glass_make_uniform_known_value() {
        let reg = registry_by_id("stained_glass").unwrap();
        let bytes = (reg.make_uniform)(&[0.5, 0.1, 0.4]);
        let expected = bytemuck::bytes_of(&StainedGlassParams {
            strength: 0.5,
            cell_size: 0.1,
            edge_width: 0.4,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 (the identity default) the output must equal the source image
    /// exactly, regardless of cell_size or edge_width. The edges pass formula reduces
    /// to mix(src, stained, 0.0) = src.
    #[test]
    fn test_stained_glass_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "stained_glass",
                values: vec![0.0, 0.08, 0.3],
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
    fn test_stained_glass_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "stained_glass",
                values: vec![1.0, 0.08, 0.3],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// At full strength (1.0) on a solid-colour image with edge_width=0, every
    /// pixel's Voronoi site maps back to the same colour (solid), so the output
    /// must remain the same colour (no visible cells since all cells share the same
    /// colour, no edges since edge_width=0).
    #[test]
    fn test_stained_glass_solid_image_no_edges_color_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 40000, 20000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "stained_glass",
                values: vec![1.0, 0.08, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 40000).abs() <= 200,
                "R: expected ~40000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 20000).abs() <= 200,
                "G: expected ~20000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 10000).abs() <= 200,
                "B: expected ~10000, got {}",
                pixel[2]
            );
        }
    }

    /// Increasing edge_width must reduce average brightness compared to edge_width=0.
    ///
    /// With edge_width=0 no borders are drawn (edge_mask=0 everywhere), so the output
    /// equals the flat Voronoi cell colour. With edge_width=1.0 the smoothstep ramp
    /// covers the full proximity range, causing pixels near cell boundaries to blend
    /// toward the near-black edge colour. On a non-uniform image the population of
    /// boundary-adjacent pixels is non-trivial, so the mean R value must be lower.
    #[test]
    fn test_stained_glass_wider_edges_reduce_average_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a bright image so the darkening effect is clearly measurable.
        let img = make_solid_image(32, 32, 60000, 60000, 60000);
        let out_no_edge = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "stained_glass",
                values: vec![1.0, 0.08, 0.0],
            }],
        );
        let out_wide_edge = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "stained_glass",
                values: vec![1.0, 0.08, 1.0],
            }],
        );
        // Compute mean R for both outputs.
        let mean_no_edge: f64 =
            out_no_edge.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 32.0);
        let mean_wide_edge: f64 =
            out_wide_edge.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 32.0);
        assert!(
            mean_wide_edge < mean_no_edge,
            "wider edge_width must produce a darker average output: \
             no_edge={mean_no_edge:.0}, wide_edge={mean_wide_edge:.0}"
        );
    }

    /// Chaining stained_glass with brightness must not panic and must preserve alpha.
    #[test]
    fn test_stained_glass_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "stained_glass",
                    values: vec![0.5, 0.08, 0.3],
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

    /// Increasing strength from 0 to 1 must change at least some pixels relative to
    /// the source. This confirms the effect is actually applied at non-zero strength.
    #[test]
    fn test_stained_glass_nonzero_strength_changes_pixels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // A non-uniform image ensures the Voronoi cell sampling produces different
        // colours than the source at cell boundaries.
        let mut img = crate::Rgba16Image::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v = ((x + y) * 1000) as u16;
                img.put_pixel(x, y, image::Rgba([v, v / 2, v / 3, 65535]));
            }
        }
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "stained_glass",
                values: vec![1.0, 0.08, 0.3],
            }],
        );
        // At least one pixel must differ from the source.
        let any_changed = out.pixels().zip(img.pixels()).any(|(o, s)| {
            (o[0] as i32 - s[0] as i32).abs() > 64
                || (o[1] as i32 - s[1] as i32).abs() > 64
                || (o[2] as i32 - s[2] as i32).abs() > 64
        });
        assert!(
            any_changed,
            "at strength=1.0 the output must differ from the source"
        );
    }
}
