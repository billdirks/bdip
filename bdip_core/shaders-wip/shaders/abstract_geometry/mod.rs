use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension, ParamKind, PassDef, PassInput,
    PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Abstract Geometry shader.
///
/// Four floats pack into 16 bytes, satisfying WebGPU uniform alignment.
///
/// # Identity design
///
/// `strength` defaults to 0.0, which linearly blends between the source image
/// and the geometric overlay with weight 0 — producing a pure passthrough at
/// the default configuration regardless of the other slider values.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AbstractGeometryParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full overlay.
    pub strength: f32,
    /// Size of each hexagonal cell in pixels at a 1000-pixel reference width.
    /// Range [10.0, 120.0]. Larger values yield fewer, bigger hexagons.
    pub cell_size: f32,
    /// Fractional width of the hexagon edge lines relative to the cell radius.
    /// Range [0.0, 0.5]. 0.0 = no visible edges, 0.5 = thick edges.
    pub edge_width: f32,
    /// Opacity of the noise-derived color fill inside each cell.
    /// Range [0.0, 1.0]. 0.0 = edges only, 1.0 = fully colored cells.
    pub fill_opacity: f32,
}

impl TransformShader for AbstractGeometryParams {
    const ID: &'static str = "abstract_geometry";
    const DISPLAY_NAME: &'static str = "Abstract Geometry";
    const DESCRIPTION: &'static str = "Overlays a procedural hexagonal grid with noise-derived cell colors, \
         producing a geometric stained-glass or mosaic effect.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the geometric overlay (1.0). \
                 The identity value is 0.0.",
        },
        SliderDef {
            name: "Cell Size",
            min: 10.0,
            max: 120.0,
            default: 40.0,
            description: "Size of each hexagonal cell in pixels at a 1000-pixel reference width. \
                 Larger values produce fewer, bigger hexagons.",
        },
        SliderDef {
            name: "Edge Width",
            min: 0.0,
            max: 0.5,
            default: 0.08,
            description: "Width of hexagon edge lines as a fraction of the cell radius. \
                 0.0 = no edges, 0.5 = thick edges filling most of the cell.",
        },
        SliderDef {
            name: "Fill Opacity",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Opacity of the noise-derived color tint inside each cell. \
                 0.0 = edges only, 1.0 = fully saturated colored cells.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "abstract_geometry",
        wgsl_source: include_str!("abstract_geometry.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[AuxTextureDef {
            name: "blue_noise_128",
            dimension: AuxTextureDimension::D2,
            filter: AuxSamplerFilter::Nearest,
        }],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            cell_size: values[1],
            edge_width: values[2],
            fill_opacity: values[3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    AbstractGeometryParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_abstract_geometry_registry_entry_exists() {
        assert!(registry_by_id("abstract_geometry").is_some());
    }

    #[test]
    fn test_abstract_geometry_registry_metadata() {
        let reg = registry_by_id("abstract_geometry").unwrap();
        assert_eq!(reg.meta.display_name, "Abstract Geometry");
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the geometric overlay (1.0). \
                         The identity value is 0.0.",
                },
                SliderDef {
                    name: "Cell Size",
                    min: 10.0,
                    max: 120.0,
                    default: 40.0,
                    description: "Size of each hexagonal cell in pixels at a 1000-pixel reference width. \
                         Larger values produce fewer, bigger hexagons.",
                },
                SliderDef {
                    name: "Edge Width",
                    min: 0.0,
                    max: 0.5,
                    default: 0.08,
                    description: "Width of hexagon edge lines as a fraction of the cell radius. \
                         0.0 = no edges, 0.5 = thick edges filling most of the cell.",
                },
                SliderDef {
                    name: "Fill Opacity",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Opacity of the noise-derived color tint inside each cell. \
                         0.0 = edges only, 1.0 = fully saturated colored cells.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes[0].aux_textures.len(),
            1,
            "must declare exactly 1 aux texture"
        );
    }

    #[test]
    fn test_abstract_geometry_make_uniform_known_value() {
        let reg = registry_by_id("abstract_geometry").unwrap();
        let bytes = (reg.make_uniform)(&[0.7, 50.0, 0.1, 0.8]);
        let expected = bytemuck::bytes_of(&AbstractGeometryParams {
            strength: 0.7,
            cell_size: 50.0,
            edge_width: 0.1,
            fill_opacity: 0.8,
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0 the output must equal the source regardless of other params.
    #[test]
    fn test_abstract_geometry_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "abstract_geometry",
                values: vec![0.0, 40.0, 0.08, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767 at strength=0, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 20000).abs() <= 64,
                "G: expected ~20000 at strength=0, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 50000).abs() <= 64,
                "B: expected ~50000 at strength=0, got {}",
                pixel[2]
            );
        }
    }

    /// Alpha channel must pass through unchanged at any strength value.
    #[test]
    fn test_abstract_geometry_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "abstract_geometry",
                values: vec![1.0, 40.0, 0.08, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    /// Full strength must produce output that differs from the source,
    /// demonstrating that the overlay is actually applied.
    #[test]
    fn test_abstract_geometry_full_strength_modifies_image() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a large-enough image so multiple hex cells are present and the
        // noise-derived colors differ from mid-gray.
        let img = make_solid_image(64, 64, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "abstract_geometry",
                values: vec![1.0, 20.0, 0.1, 1.0],
            }],
        );
        let any_different = out.pixels().any(|p| (p[0] as i32 - 32767).abs() > 200);
        assert!(
            any_different,
            "full strength must produce pixels that differ from the source"
        );
    }

    /// Larger cell size must produce fewer distinct color regions than smaller
    /// cell size on a fixed image, detectable as fewer unique pixel values.
    #[test]
    fn test_abstract_geometry_larger_cells_produce_fewer_regions() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(64, 64, 32767, 32767, 32767);

        let out_small = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "abstract_geometry",
                values: vec![1.0, 12.0, 0.05, 1.0],
            }],
        );
        let out_large = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "abstract_geometry",
                values: vec![1.0, 60.0, 0.05, 1.0],
            }],
        );

        // Count distinct R values as a proxy for number of unique hex cells.
        let unique_small: std::collections::HashSet<u16> =
            out_small.pixels().map(|p| p[0]).collect();
        let unique_large: std::collections::HashSet<u16> =
            out_large.pixels().map(|p| p[0]).collect();

        assert!(
            unique_small.len() >= unique_large.len(),
            "smaller cells must produce at least as many distinct regions as larger cells: \
             small={}, large={}",
            unique_small.len(),
            unique_large.len()
        );
    }

    /// Wider edge_width must produce more solidly-dark edge pixels than narrow
    /// edge_width when fill is disabled.
    ///
    /// Uses a 256×256 image with cell_size=60 so the effective hex radius is
    /// r = 60 × (256/1000) ≈ 15.4 px.  At edge_width=0.01 the edge half-width
    /// is ~0.15 px (sub-pixel — near-zero solidly-dark pixels).  At
    /// edge_width=0.4 the half-width is ~6.1 px (many solidly-dark pixels).
    #[test]
    fn test_abstract_geometry_wider_edges_produce_more_dark_pixels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // White source so edge pixels (darkened to black) are easy to detect.
        let img = make_solid_image(256, 256, 65535, 65535, 65535);

        let out_narrow = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "abstract_geometry",
                // sub-pixel-thin edges, no fill
                values: vec![1.0, 60.0, 0.01, 0.0],
            }],
        );
        let out_wide = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "abstract_geometry",
                // wide edges (~6 px half-width), no fill
                values: vec![1.0, 60.0, 0.4, 0.0],
            }],
        );

        // Threshold at 8192 (≈ 12.5 % of full white): only pixels deep inside
        // an edge core qualify.  Sub-pixel narrow edges produce near-zero such
        // pixels; wide edges produce many.
        let dark_narrow = out_narrow.pixels().filter(|p| p[0] < 8192).count();
        let dark_wide = out_wide.pixels().filter(|p| p[0] < 8192).count();

        assert!(
            dark_wide > dark_narrow,
            "wider edges must produce more solidly-dark pixels than narrow edges: \
             narrow={dark_narrow}, wide={dark_wide}"
        );
    }

    /// Chaining with brightness must not panic and must preserve alpha.
    #[test]
    fn test_abstract_geometry_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "abstract_geometry",
                    values: vec![0.5, 40.0, 0.08, 0.5],
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

    /// Two runs with identical inputs must produce bit-identical output.
    #[test]
    fn test_abstract_geometry_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "abstract_geometry",
            values: vec![0.8, 40.0, 0.08, 0.5],
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
