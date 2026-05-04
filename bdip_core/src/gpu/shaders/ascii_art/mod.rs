use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension, ParamKind, PassDef, PassInput,
    PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the ASCII Art two-pass effect.
///
/// # Effect design
///
/// Pass 1 converts the source to BT.709 greyscale. Pass 2 divides the image
/// into square character cells, computes the average luminance per cell, maps
/// that luminance to one of 16 ASCII characters (ordered by ink density), and
/// renders the character bitmask from a pre-baked 128×128 character atlas.
/// Each output pixel is either the ink colour (a tinted version of the cell's
/// average colour) or the background colour, blended toward the original image
/// by `1.0 - strength`.
///
/// # Identity design
///
/// At `strength = 0.0` the output is the source image unchanged. At
/// `strength = 1.0` the full ASCII art effect is applied. Cell size controls
/// the granularity: smaller values produce denser character grids; larger
/// values produce a coarser, more legible effect.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AsciiArtParams {
    /// Size (in pixels) of each square character cell. Larger values produce
    /// fewer, more legible characters; smaller values produce a denser mosaic.
    pub cell_size: f32,
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full ASCII art.
    pub strength: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for AsciiArtParams {
    const ID: &'static str = "ascii_art";
    const DISPLAY_NAME: &'static str = "ASCII Art";
    const DESCRIPTION: &'static str = "Renders the image as a grid of ASCII characters whose ink density \
         matches the local brightness, producing a classic text-art effect.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Cell Size",
            min: 4.0,
            max: 32.0,
            default: 8.0,
            description: "Width and height of each character cell in pixels. \
                          Larger values produce fewer, larger characters; smaller \
                          values produce a finer, denser character grid.",
        },
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full \
                          ASCII art effect (1.0). The identity value is 0.0.",
        },
    ]);

    // Two-pass pipeline:
    //   Pass 1 — gray: BT.709 greyscale conversion → scratch "gray".
    //   Pass 2 — ascii: Character-cell quantisation + atlas lookup + blend → Final.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "gray",
            wgsl_source: include_str!("ascii_art_gray.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("gray"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "ascii",
            wgsl_source: include_str!("ascii_art_ascii.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("gray")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[AuxTextureDef {
                name: "ascii_char_map_16x16",
                dimension: AuxTextureDimension::D2,
                filter: AuxSamplerFilter::Nearest,
            }],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            cell_size: values[0],
            strength: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<AsciiArtParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ---------------------------------------------------------------------------
    // Registry tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_ascii_art_registry_entry_exists() {
        assert!(registry_by_id("ascii_art").is_some());
    }

    #[test]
    fn test_ascii_art_registry_metadata() {
        let reg = registry_by_id("ascii_art").unwrap();
        assert_eq!(reg.meta.display_name, "ASCII Art");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Cell Size",
                    min: 4.0,
                    max: 32.0,
                    default: 8.0,
                    description: "Width and height of each character cell in pixels. \
                          Larger values produce fewer, larger characters; smaller \
                          values produce a finer, denser character grid.",
                },
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full \
                          ASCII art effect (1.0). The identity value is 0.0.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "ASCII Art must have exactly 2 passes"
        );
    }

    #[test]
    fn test_ascii_art_make_uniform_known_value() {
        let reg = registry_by_id("ascii_art").unwrap();
        let bytes = (reg.make_uniform)(&[12.0, 0.7]);
        let expected = bytemuck::bytes_of(&AsciiArtParams {
            cell_size: 12.0,
            strength: 0.7,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    // ---------------------------------------------------------------------------
    // GPU roundtrip tests
    // ---------------------------------------------------------------------------

    /// At strength=0.0 the ascii pass blends with mix(ascii, src, 0.0) = src,
    /// so the output must equal the source regardless of cell_size.
    #[test]
    fn test_ascii_art_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "ascii_art",
                values: vec![8.0, 0.0],
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
    fn test_ascii_art_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "ascii_art",
                values: vec![8.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// At strength=1.0 a uniform mid-grey image must produce output that
    /// differs from the source (the character grid is rendered).
    #[test]
    fn test_ascii_art_full_strength_changes_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "ascii_art",
                values: vec![8.0, 1.0],
            }],
        );
        // Character cells contain ink (brighter) and background (darker) pixels,
        // so the output must vary spatially rather than being uniformly grey.
        let any_changed = out.pixels().any(|p| (p[0] as i32 - 32767).abs() > 500);
        assert!(any_changed, "strength=1.0 must visibly change the output");
    }

    /// Changing cell_size must produce a different output pattern.
    #[test]
    fn test_ascii_art_cell_size_changes_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Gradient image to ensure spatial variation across cell boundaries.
        let w = 32u32;
        let h = 32u32;
        let mut img = crate::Rgba16Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = ((x + y) * 1000).min(65535) as u16;
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_small = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "ascii_art",
                values: vec![4.0, 1.0],
            }],
        );
        let out_large = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "ascii_art",
                values: vec![16.0, 1.0],
            }],
        );

        let any_different = out_small
            .pixels()
            .zip(out_large.pixels())
            .any(|(s, l)| (s[0] as i32 - l[0] as i32).abs() > 64);
        assert!(
            any_different,
            "different cell_size values must produce different outputs"
        );
    }

    /// Chaining ascii_art with brightness must not panic and must preserve alpha.
    #[test]
    fn test_ascii_art_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "ascii_art",
                    values: vec![8.0, 0.5],
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

    /// Running ASCII Art twice with identical inputs must produce bit-identical
    /// output (determinism requirement).
    #[test]
    fn test_ascii_art_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "ascii_art",
            values: vec![8.0, 0.8],
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
