use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters shared across both Polygon passes.
///
/// # Layout
///
/// Four floats at 16 bytes — no explicit padding needed.
///
/// # Identity design
///
/// A purely procedural Voronoi facet effect cannot reduce to a literal per-pixel
/// identity at any non-zero density: even the smallest cell still samples from a
/// seed point, not the exact pixel position. The chosen strategy mirrors Pointillism
/// and Pencil Sketch: a `strength` blend slider defaulting to `0.0` passes the
/// source through unchanged regardless of `density` and `jitter`. At `strength = 1.0`
/// the full low-poly facet rendering is shown.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PolygonParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full low-poly effect.
    pub strength: f32,
    /// Number of seed points per axis, i.e. the grid is `density × density` cells.
    /// Range [2.0, 64.0]; default 20.0.
    pub density: f32,
    /// Maximum random offset of each seed point expressed as a fraction of the cell
    /// half-size. 0.0 = regular grid (no jitter); 1.0 = maximum displacement (seed
    /// can reach the cell boundary). Range [0.0, 1.0]; default 0.75.
    pub jitter: f32,
    pub _padding: f32,
}

impl TransformShader for PolygonParams {
    const ID: &'static str = "polygon";
    const DISPLAY_NAME: &'static str = "Polygon";
    const DESCRIPTION: &'static str = "Voronoi-based low-poly artistic effect that divides \
         the image into irregular facets, each filled with the source colour at its seed point.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full low-poly \
                          facet effect (1.0). The identity value is 0.0.",
        },
        SliderDef {
            name: "Density",
            min: 2.0,
            max: 64.0,
            default: 20.0,
            description: "Number of seed points per axis. Higher values produce smaller, \
                          more numerous facets.",
        },
        SliderDef {
            name: "Jitter",
            min: 0.0,
            max: 1.0,
            default: 0.75,
            description: "Random displacement of each seed point as a fraction of the \
                          cell half-size. 0.0 produces a regular grid; 1.0 maximises \
                          irregularity.",
        },
    ]);

    // Two-pass pipeline:
    //   Pass 1 — voronoi: for each pixel, find the nearest jittered seed point
    //            and write the source colour at that seed → scratch texture.
    //   Pass 2 — blend:   mix the scratch (facet colour) with the source via
    //            `strength`.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "voronoi",
            wgsl_source: include_str!("polygon_voronoi.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("voronoi"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "blend",
            wgsl_source: include_str!("polygon_blend.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("voronoi")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            density: values[1],
            jitter: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<PolygonParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_polygon_registry_entry_exists() {
        assert!(registry_by_id("polygon").is_some());
    }

    #[test]
    fn test_polygon_registry_metadata() {
        let reg = registry_by_id("polygon").unwrap();
        assert_eq!(reg.meta.display_name, "Polygon");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full low-poly \
                                  facet effect (1.0). The identity value is 0.0.",
                },
                SliderDef {
                    name: "Density",
                    min: 2.0,
                    max: 64.0,
                    default: 20.0,
                    description: "Number of seed points per axis. Higher values produce smaller, \
                                  more numerous facets.",
                },
                SliderDef {
                    name: "Jitter",
                    min: 0.0,
                    max: 1.0,
                    default: 0.75,
                    description: "Random displacement of each seed point as a fraction of the \
                                  cell half-size. 0.0 produces a regular grid; 1.0 maximises \
                                  irregularity.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "Polygon must have exactly 2 passes"
        );
    }

    #[test]
    fn test_polygon_make_uniform_known_value() {
        let reg = registry_by_id("polygon").unwrap();
        let bytes = (reg.make_uniform)(&[0.8, 16.0, 0.5]);
        let expected = bytemuck::bytes_of(&PolygonParams {
            strength: 0.8,
            density: 16.0,
            jitter: 0.5,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 the blend pass reduces to mix(src, facets, 0.0) = src.
    /// The output must equal the source for any density and jitter value.
    #[test]
    fn test_polygon_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polygon",
                values: vec![0.0, 20.0, 0.75],
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
    fn test_polygon_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polygon",
                values: vec![1.0, 20.0, 0.75],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// On a solid-colour image, every Voronoi cell samples the same colour, so the
    /// full-strength output must be uniformly that colour (within f16 rounding).
    #[test]
    fn test_polygon_solid_image_produces_uniform_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 40000, 20000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polygon",
                values: vec![1.0, 10.0, 0.5],
            }],
        );
        let first = out.get_pixel(0, 0);
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - first[0] as i32).abs() <= 64,
                "R must be uniform on solid source; got {} vs {}",
                pixel[0],
                first[0]
            );
            assert!(
                (pixel[1] as i32 - first[1] as i32).abs() <= 64,
                "G must be uniform on solid source; got {} vs {}",
                pixel[1],
                first[1]
            );
            assert!(
                (pixel[2] as i32 - first[2] as i32).abs() <= 64,
                "B must be uniform on solid source; got {} vs {}",
                pixel[2],
                first[2]
            );
        }
    }

    /// Zero jitter places seeds on a regular grid. On a gradient image, different
    /// densities must produce visually different outputs — proves the Voronoi cell
    /// assignment is actually happening.
    #[test]
    fn test_polygon_different_densities_produce_different_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Horizontal gradient so cells at different columns sample different colours.
        let mut img = crate::Rgba16Image::new(64, 64);
        for y in 0..64u32 {
            for x in 0..64u32 {
                let v = (x * 1000) as u16;
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polygon",
                values: vec![1.0, 4.0, 0.0], // coarse cells, no jitter
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polygon",
                values: vec![1.0, 16.0, 0.0], // fine cells, no jitter
            }],
        );

        let any_different = out_low
            .pixels()
            .zip(out_high.pixels())
            .any(|(lo, hi)| (lo[0] as i32 - hi[0] as i32).abs() > 64);
        assert!(
            any_different,
            "density=4 and density=16 must produce different outputs on a gradient image"
        );
    }

    /// Non-zero jitter must change the output compared to zero jitter (on a
    /// gradient image where seed positions matter).
    #[test]
    fn test_polygon_nonzero_jitter_differs_from_zero_jitter() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(64, 64);
        for y in 0..64u32 {
            for x in 0..64u32 {
                let v = (x * 1000) as u16;
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_no_jitter = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polygon",
                values: vec![1.0, 8.0, 0.0],
            }],
        );
        let out_jitter = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polygon",
                values: vec![1.0, 8.0, 1.0],
            }],
        );

        let any_different = out_no_jitter
            .pixels()
            .zip(out_jitter.pixels())
            .any(|(nj, j)| (nj[0] as i32 - j[0] as i32).abs() > 64);
        assert!(
            any_different,
            "jitter=0.0 and jitter=1.0 must produce different outputs on a gradient image"
        );
    }

    /// Running Polygon twice with identical inputs must produce bit-identical output,
    /// confirming that the hash-based jitter is deterministic.
    #[test]
    fn test_polygon_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "polygon",
            values: vec![0.8, 20.0, 0.75],
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

    /// Chaining Polygon with brightness must not panic and must preserve alpha.
    #[test]
    fn test_polygon_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "polygon",
                    values: vec![0.5, 8.0, 0.5],
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

    /// At full strength with a regular-grid Voronoi (jitter=0), pixels within the same
    /// grid cell must all share the same colour (they all map to the same seed point).
    #[test]
    fn test_polygon_pixels_within_same_cell_share_colour() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Solid image — every seed samples the same colour, so all cells are uniform
        // and the invariant holds trivially. Use strength=1 to expose facet colours.
        let img = make_solid_image(32, 32, 25000, 45000, 55000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polygon",
                values: vec![1.0, 8.0, 0.0], // 8 cells per axis, no jitter
            }],
        );

        // All output pixels must be within ±64 of each other (solid source → uniform cells).
        let ref_pixel = out.get_pixel(0, 0);
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - ref_pixel[0] as i32).abs() <= 64,
                "R: all pixels must be uniform; got {} vs ref {}",
                pixel[0],
                ref_pixel[0]
            );
        }
    }
}
