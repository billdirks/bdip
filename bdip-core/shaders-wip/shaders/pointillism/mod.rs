use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters shared across both Pointillism passes.
///
/// The three meaningful fields pack into 12 bytes; one padding float brings the
/// struct to 16 bytes to satisfy WebGPU's uniform alignment requirement.
///
/// # Identity design
///
/// The spec requires that default parameter values produce a no-op transformation.
/// For Pointillism this is not literally achievable at any non-zero strength: the
/// effect replaces continuous tone with discrete dots on a white background.
///
/// The chosen design uses a `strength` blend parameter defaulting to `0.0`, which
/// passes the source image through unchanged regardless of the dot/grid parameters.
/// At `strength = 1.0`, the full pointillist rendering is shown. This pattern
/// matches Pencil Sketch and Stained Glass.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointillismParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full pointillism effect.
    pub strength: f32,
    /// Grid cell size in pixels. Determines the spacing between dot centers.
    /// Range [4.0, 64.0].
    pub grid_size: f32,
    /// Dot radius as a fraction of the grid cell half-size. 1.0 fills the cell
    /// fully; 0.5 leaves visible gaps. Range [0.1, 1.0].
    pub dot_size: f32,
    pub _padding: f32,
}

impl TransformShader for PointillismParams {
    const ID: &'static str = "pointillism";
    const DISPLAY_NAME: &'static str = "Pointillism";
    const DESCRIPTION: &'static str = "Simulates the pointillism painting technique by rendering filled \
         colour dots sampled from a regular grid on a white background.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full \
                          pointillism effect (1.0). The identity value is 0.0.",
        },
        SliderDef {
            name: "Grid Size",
            min: 4.0,
            max: 64.0,
            default: 16.0,
            description: "Spacing between dot centres in pixels. Larger values \
                          produce bigger, more widely spaced dots.",
        },
        SliderDef {
            name: "Dot Size",
            min: 0.1,
            max: 1.0,
            default: 0.8,
            description: "Dot radius as a fraction of the grid cell half-size. \
                          1.0 fills the cell completely; smaller values leave \
                          visible gaps between dots.",
        },
    ]);

    // Two-pass pipeline:
    //   Pass 1 — quantize: for each pixel, sample the source at its grid-cell
    //             centre and write that colour → scratch texture.
    //   Pass 2 — dots:     for each pixel, check distance to its cell centre;
    //             inside the dot radius → cell colour, outside → white; then
    //             blend with source via `strength`.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "quantize",
            wgsl_source: include_str!("pointillism_quantize.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("quantized"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "dots",
            wgsl_source: include_str!("pointillism_dots.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("quantized")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            grid_size: values[1],
            dot_size: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    PointillismParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_pointillism_registry_entry_exists() {
        assert!(registry_by_id("pointillism").is_some());
    }

    #[test]
    fn test_pointillism_registry_metadata() {
        let reg = registry_by_id("pointillism").unwrap();
        assert_eq!(reg.meta.display_name, "Pointillism");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full \
                                  pointillism effect (1.0). The identity value is 0.0.",
                },
                SliderDef {
                    name: "Grid Size",
                    min: 4.0,
                    max: 64.0,
                    default: 16.0,
                    description: "Spacing between dot centres in pixels. Larger values \
                                  produce bigger, more widely spaced dots.",
                },
                SliderDef {
                    name: "Dot Size",
                    min: 0.1,
                    max: 1.0,
                    default: 0.8,
                    description: "Dot radius as a fraction of the grid cell half-size. \
                                  1.0 fills the cell completely; smaller values leave \
                                  visible gaps between dots.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "Pointillism must have exactly 2 passes"
        );
    }

    #[test]
    fn test_pointillism_make_uniform_known_value() {
        let reg = registry_by_id("pointillism").unwrap();
        let bytes = (reg.make_uniform)(&[0.75, 20.0, 0.6]);
        let expected = bytemuck::bytes_of(&PointillismParams {
            strength: 0.75,
            grid_size: 20.0,
            dot_size: 0.6,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 the dots pass reduces to mix(src, dots, 0.0) = src,
    /// so the output must equal the source regardless of grid_size or dot_size.
    #[test]
    fn test_pointillism_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pointillism",
                values: vec![0.0, 16.0, 0.8],
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
    fn test_pointillism_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pointillism",
                values: vec![1.0, 16.0, 0.8],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// At full strength (1.0) with a solid-colour image, pixels at grid-cell
    /// centres (inside the dot) should retain the source colour. The centre of
    /// a grid cell (with grid_size=8, dot_size=1.0) is at (3,3), (11,3), etc.
    /// Use a large grid relative to image size so we can pinpoint centres.
    #[test]
    fn test_pointillism_dot_centre_retains_source_colour() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // 32×32 image, solid red-ish colour, grid_size=16, dot_size=1.0 (full fill).
        let img = make_solid_image(32, 32, 40000, 10000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pointillism",
                values: vec![1.0, 16.0, 1.0],
            }],
        );
        // Grid-cell centre of the first cell with grid_size=16: x=7, y=7
        // (floor(7/16)*16 + 7 = 7, which is the centre of cell [0,0]).
        // At dot_size=1.0 the entire cell is filled, so the centre pixel
        // should have the source colour (with ±200 f16 tolerance).
        let centre = out.get_pixel(7, 7);
        assert!(
            (centre[0] as i32 - 40000).abs() <= 200,
            "dot-centre R: expected ~40000, got {}",
            centre[0]
        );
        assert!(
            (centre[1] as i32 - 10000).abs() <= 200,
            "dot-centre G: expected ~10000, got {}",
            centre[1]
        );
    }

    /// With dot_size < 1.0 and full strength, pixels at the very corners of grid
    /// cells (far from any dot centre) should be near-white. The exact corner of
    /// a 16×16 grid cell is at (15, 15) — 8 px from the cell centre at (7, 7),
    /// which is outside a dot of radius 0.5 * 8 = 4 px.
    #[test]
    fn test_pointillism_gap_between_dots_is_white() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Solid dark image; gaps should be white regardless of source colour.
        let img = make_solid_image(32, 32, 5000, 5000, 5000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pointillism",
                values: vec![1.0, 16.0, 0.4],
            }],
        );
        // Pixel at (15, 15) is the corner between four cells — well outside any dot.
        let corner = out.get_pixel(15, 15);
        assert!(
            corner[0] > 55000,
            "gap pixel R: expected near-white (>55000), got {}",
            corner[0]
        );
        assert!(
            corner[1] > 55000,
            "gap pixel G: expected near-white (>55000), got {}",
            corner[1]
        );
        assert!(
            corner[2] > 55000,
            "gap pixel B: expected near-white (>55000), got {}",
            corner[2]
        );
    }

    /// Changing grid_size must change the output (different cell quantization).
    #[test]
    fn test_pointillism_different_grid_sizes_produce_different_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Gradient image so different cells sample different colours.
        let mut img = crate::Rgba16Image::new(64, 64);
        for y in 0..64u32 {
            for x in 0..64u32 {
                let v = (x * 1000) as u16;
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_small = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pointillism",
                values: vec![1.0, 8.0, 0.8],
            }],
        );
        let out_large = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pointillism",
                values: vec![1.0, 32.0, 0.8],
            }],
        );

        let any_different = out_small
            .pixels()
            .zip(out_large.pixels())
            .any(|(s, l)| (s[0] as i32 - l[0] as i32).abs() > 64);
        assert!(
            any_different,
            "grid_size=8 and grid_size=32 must produce different outputs on a gradient image"
        );
    }

    /// Chaining pointillism with brightness must not panic and must preserve alpha.
    #[test]
    fn test_pointillism_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "pointillism",
                    values: vec![0.5, 8.0, 0.8],
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

    /// Running Pointillism twice with identical inputs must produce bit-identical output.
    #[test]
    fn test_pointillism_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "pointillism",
            values: vec![0.8, 16.0, 0.8],
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
