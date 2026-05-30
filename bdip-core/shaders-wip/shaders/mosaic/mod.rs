use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MosaicParams {
    pub tile_width: f32,
    pub tile_height: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for MosaicParams {
    const ID: &'static str = "mosaic";
    const DISPLAY_NAME: &'static str = "Mosaic";
    const DESCRIPTION: &'static str = "Divides the image into rectangular tiles and fills each tile with the color \
         sampled from the tile center, producing a stained-glass mosaic appearance.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Tile Width",
            min: 1.0,
            max: 128.0,
            default: 1.0,
            description: "Tile width in pixels. 1.0 = identity (no tiling); \
                          larger values produce wider mosaic tiles.",
        },
        SliderDef {
            name: "Tile Height",
            min: 1.0,
            max: 128.0,
            default: 1.0,
            description: "Tile height in pixels. 1.0 = identity (no tiling); \
                          larger values produce taller mosaic tiles.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "mosaic",
        wgsl_source: include_str!("mosaic.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            tile_width: values[0],
            tile_height: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<MosaicParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_mosaic_registry_entry_exists() {
        assert!(registry_by_id("mosaic").is_some());
    }

    #[test]
    fn test_mosaic_registry_metadata() {
        let reg = registry_by_id("mosaic").unwrap();
        assert_eq!(reg.meta.display_name, "Mosaic");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Tile Width",
                    min: 1.0,
                    max: 128.0,
                    default: 1.0,
                    description: "Tile width in pixels. 1.0 = identity (no tiling); \
                                  larger values produce wider mosaic tiles.",
                },
                SliderDef {
                    name: "Tile Height",
                    min: 1.0,
                    max: 128.0,
                    default: 1.0,
                    description: "Tile height in pixels. 1.0 = identity (no tiling); \
                                  larger values produce taller mosaic tiles.",
                },
            ])
        );
    }

    #[test]
    fn test_mosaic_make_uniform_known_value() {
        let reg = registry_by_id("mosaic").unwrap();
        let bytes = (reg.make_uniform)(&[16.0, 24.0]);
        let expected = bytemuck::bytes_of(&MosaicParams {
            tile_width: 16.0,
            tile_height: 24.0,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    /// At tile_width=1 and tile_height=1 (identity), every output pixel samples from
    /// the center of its own 1x1 tile, which is itself. A solid-color image makes all
    /// pixels trivially verifiable regardless of exact UV mapping.
    #[test]
    fn test_mosaic_identity_at_tile_size_one() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 40000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "mosaic",
                values: vec![1.0, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 20000).abs() <= 64,
                "R: expected ~20000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 40000).abs() <= 64,
                "G: expected ~40000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 60000).abs() <= 64,
                "B: expected ~60000, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// With a tile covering the entire image, all output pixels sample from the center
    /// of that one tile, producing a uniform color across the whole image.
    #[test]
    fn test_mosaic_full_image_tile_produces_uniform_color() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(8, 8, 50000, 25000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "mosaic",
                values: vec![128.0, 128.0],
            }],
        );

        // All pixels must share the same color (the center sample of the single tile).
        let first = out.get_pixel(0, 0);
        for pixel in out.pixels() {
            assert_eq!(
                pixel[0], first[0],
                "R values must be uniform across one tile"
            );
            assert_eq!(
                pixel[1], first[1],
                "G values must be uniform across one tile"
            );
            assert_eq!(
                pixel[2], first[2],
                "B values must be uniform across one tile"
            );
        }
    }

    /// Rectangular tiles (different width and height) must still produce uniform colors
    /// within each tile when the source image is solid.
    #[test]
    fn test_mosaic_rectangular_tiles_produce_uniform_color_per_tile() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Solid image so every tile's center sample equals every other pixel.
        let img = make_solid_image(8, 8, 30000, 45000, 15000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "mosaic",
                values: vec![4.0, 8.0], // 4-wide × 8-tall tiles
            }],
        );

        let first = out.get_pixel(0, 0);
        for pixel in out.pixels() {
            assert_eq!(pixel[0], first[0], "R must be uniform (solid source)");
            assert_eq!(pixel[1], first[1], "G must be uniform (solid source)");
            assert_eq!(pixel[2], first[2], "B must be uniform (solid source)");
        }
    }

    /// Alpha channel must pass through unchanged regardless of tile size.
    #[test]
    fn test_mosaic_alpha_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "mosaic",
                values: vec![8.0, 8.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    /// Chaining mosaic with brightness must not panic and must preserve alpha.
    #[test]
    fn test_mosaic_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "mosaic",
                    values: vec![4.0, 4.0],
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
}
