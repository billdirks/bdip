use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters shared across both passes of the 8-bit pixel art effect.
///
/// `pixel_size` controls the block size in output pixels. At 1.0, each output
/// pixel maps to exactly one source pixel (identity). At larger values, groups of
/// pixels adopt the color of the block's top-left sample, producing the
/// characteristic blocky look.
///
/// `color_levels` sets the number of quantization steps per channel (1–256). At
/// 256, all color values pass through unmodified (identity). Lower values snap each
/// channel to a coarser step, reducing the apparent palette size. The effect is
/// equivalent to `floor(c * levels) / (levels - 1)` per channel.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PixelArt8BitParams {
    /// Block size in pixels. Default 1.0 = identity (no pixelation).
    pub pixel_size: f32,
    /// Per-channel quantization steps. Default 256.0 = identity (no posterization).
    pub color_levels: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for PixelArt8BitParams {
    const ID: &'static str = "pixel_art_8bit";
    const DISPLAY_NAME: &'static str = "8-bit Pixel Art";
    const DESCRIPTION: &'static str = "Simulates old 8-bit video game graphics by pixelating the image and \
         limiting the color palette.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Pixel Size",
            min: 1.0,
            max: 64.0,
            default: 1.0,
            description: "Block size in pixels. 1 = no pixelation; higher values produce \
                          larger, blockier pixels.",
        },
        SliderDef {
            name: "Color Levels",
            min: 2.0,
            max: 256.0,
            default: 256.0,
            description: "Number of quantization steps per color channel. 256 = no \
                          posterization; lower values reduce the palette.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "pixelate",
            wgsl_source: include_str!("pixel_art_8bit_pixelate.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("pixelated"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "quantize",
            wgsl_source: include_str!("pixel_art_8bit_quantize.wgsl"),
            inputs: &[PassInput::Scratch("pixelated")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            pixel_size: values[0],
            color_levels: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    PixelArt8BitParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // -----------------------------------------------------------------------
    // Registry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pixel_art_8bit_registry_entry_exists() {
        assert!(registry_by_id("pixel_art_8bit").is_some());
    }

    #[test]
    fn test_pixel_art_8bit_registry_display_name() {
        let reg = registry_by_id("pixel_art_8bit").unwrap();
        assert_eq!(reg.meta.display_name, "8-bit Pixel Art");
    }

    #[test]
    fn test_pixel_art_8bit_registry_param_kind_is_sliders() {
        let reg = registry_by_id("pixel_art_8bit").unwrap();
        assert!(
            matches!(reg.meta.param, ParamKind::Sliders(_)),
            "expected ParamKind::Sliders"
        );
    }

    #[test]
    fn test_pixel_art_8bit_registry_slider_count() {
        let reg = registry_by_id("pixel_art_8bit").unwrap();
        if let ParamKind::Sliders(sliders) = reg.meta.param {
            assert_eq!(sliders.len(), 2, "expected 2 sliders");
        }
    }

    #[test]
    fn test_pixel_art_8bit_registry_pixel_size_slider_def() {
        let reg = registry_by_id("pixel_art_8bit").unwrap();
        if let ParamKind::Sliders(sliders) = reg.meta.param {
            assert_eq!(sliders[0].name, "Pixel Size");
            assert_eq!(sliders[0].min, 1.0);
            assert_eq!(sliders[0].max, 64.0);
            assert_eq!(sliders[0].default, 1.0);
        }
    }

    #[test]
    fn test_pixel_art_8bit_registry_color_levels_slider_def() {
        let reg = registry_by_id("pixel_art_8bit").unwrap();
        if let ParamKind::Sliders(sliders) = reg.meta.param {
            assert_eq!(sliders[1].name, "Color Levels");
            assert_eq!(sliders[1].min, 2.0);
            assert_eq!(sliders[1].max, 256.0);
            assert_eq!(sliders[1].default, 256.0);
        }
    }

    #[test]
    fn test_pixel_art_8bit_registry_pass_count() {
        let reg = registry_by_id("pixel_art_8bit").unwrap();
        assert_eq!(reg.meta.passes.len(), 2, "expected 2 passes");
    }

    // -----------------------------------------------------------------------
    // Uniform construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_pixel_art_8bit_make_uniform_pixel_size() {
        let reg = registry_by_id("pixel_art_8bit").unwrap();
        let bytes = (reg.make_uniform)(&[8.0, 256.0]);
        let expected = bytemuck::bytes_of(&PixelArt8BitParams {
            pixel_size: 8.0,
            color_levels: 256.0,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_pixel_art_8bit_make_uniform_color_levels() {
        let reg = registry_by_id("pixel_art_8bit").unwrap();
        let bytes = (reg.make_uniform)(&[1.0, 4.0]);
        let expected = bytemuck::bytes_of(&PixelArt8BitParams {
            pixel_size: 1.0,
            color_levels: 4.0,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    // -----------------------------------------------------------------------
    // GPU roundtrip — identity (default parameters = no-op)
    // -----------------------------------------------------------------------

    #[test]
    fn test_pixel_art_8bit_identity_pixel_size_1_levels_256() {
        // pixel_size=1 snaps each pixel to itself; color_levels=256 maps each
        // u16 channel to the nearest 1/255 step, which is within ±128 u16
        // rounding error for values in the 0–65535 range.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixel_art_8bit",
                values: vec![1.0, 256.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 256,
                "R: expected ~32767 at identity, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 32767).abs() <= 256,
                "G: expected ~32767 at identity, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 32767).abs() <= 256,
                "B: expected ~32767 at identity, got {}",
                pixel[2]
            );
        }
    }

    // -----------------------------------------------------------------------
    // GPU roundtrip — pixelation behavior
    // -----------------------------------------------------------------------

    #[test]
    fn test_pixel_art_8bit_large_pixel_size_makes_uniform_blocks() {
        // A checkerboard image (alternating dark/bright pixels) passed through a
        // large pixel_size should produce uniform-colored blocks: all pixels
        // within each block take the color of the block's representative sample.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 16×16 checkerboard: even (x+y) pixels are dark, odd are bright.
        let mut img = crate::Rgba16Image::new(16, 16);
        for y in 0..16u32 {
            for x in 0..16u32 {
                let v: u16 = if (x + y) % 2 == 0 { 10000 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        // pixel_size=8 means each 8×8 block maps to the same source pixel
        // (the block's top-left corner). All pixels in a block must be equal.
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixel_art_8bit",
                values: vec![8.0, 256.0],
            }],
        );

        // All pixels in the top-left 8×8 block must be identical.
        let reference = out.get_pixel(0, 0)[0];
        for y in 0..8u32 {
            for x in 0..8u32 {
                let got = out.get_pixel(x, y)[0];
                assert_eq!(
                    got, reference,
                    "block uniformity failure at ({x},{y}): expected {reference}, got {got}"
                );
            }
        }
    }

    #[test]
    fn test_pixel_art_8bit_pixel_size_1_preserves_spatial_variation() {
        // At pixel_size=1 each output pixel samples from its own exact source
        // position. A step image must retain its step edge.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(16, 16);
        for y in 0..16u32 {
            for x in 0..16u32 {
                let v: u16 = if x < 8 { 10000 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixel_art_8bit",
                values: vec![1.0, 256.0],
            }],
        );

        // Left half should remain darker than right half.
        let left = out.get_pixel(4, 8)[0] as i32;
        let right = out.get_pixel(12, 8)[0] as i32;
        assert!(
            left < right,
            "step edge not preserved at pixel_size=1: left={left}, right={right}"
        );
    }

    // -----------------------------------------------------------------------
    // GPU roundtrip — color quantization behavior
    // -----------------------------------------------------------------------

    #[test]
    fn test_pixel_art_8bit_low_color_levels_posterizes() {
        // With color_levels=2, each channel is quantized to either the low or
        // high quantization step. For a mid-gray input, the output must differ
        // from the input (it snaps to the nearest step).
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Mid-gray (u16 ≈ 32767 maps to linear ~0.5).
        let img = make_solid_image(4, 4, 32767, 32767, 32767);

        let out_256 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixel_art_8bit",
                values: vec![1.0, 256.0],
            }],
        );
        let out_2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixel_art_8bit",
                values: vec![1.0, 2.0],
            }],
        );

        // levels=2 snaps to either 0 or 65535; the mid-gray input must shift
        // substantially from the unquantized output.
        let diff = (out_2.get_pixel(0, 0)[0] as i32 - out_256.get_pixel(0, 0)[0] as i32).abs();
        assert!(
            diff > 1000,
            "color_levels=2 should shift mid-gray substantially; diff={diff}"
        );
    }

    #[test]
    fn test_pixel_art_8bit_color_levels_256_preserves_value() {
        // color_levels=256 with pixel_size=1 should reproduce the source within
        // the quantization rounding budget (±256 u16).
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 50000, 20000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixel_art_8bit",
                values: vec![1.0, 256.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 50000).abs() <= 256,
                "R: expected ~50000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 20000).abs() <= 256,
                "G: expected ~20000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 40000).abs() <= 256,
                "B: expected ~40000, got {}",
                pixel[2]
            );
        }
    }

    // -----------------------------------------------------------------------
    // GPU roundtrip — alpha preservation
    // -----------------------------------------------------------------------

    #[test]
    fn test_pixel_art_8bit_alpha_preserved_at_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixel_art_8bit",
                values: vec![1.0, 256.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    #[test]
    fn test_pixel_art_8bit_alpha_preserved_with_pixelation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixel_art_8bit",
                values: vec![8.0, 4.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be preserved through pixelation"
            );
        }
    }

    // -----------------------------------------------------------------------
    // GPU roundtrip — chaining with another shader
    // -----------------------------------------------------------------------

    #[test]
    fn test_pixel_art_8bit_chains_with_brightness() {
        // Verify that the output of pixel_art_8bit can be fed as input to the
        // brightness shader without engine errors, and that the combined
        // pipeline produces a result brighter than the pixel_art_8bit-only run.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 20000, 20000, 20000);

        let out_art_only = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixel_art_8bit",
                values: vec![4.0, 8.0],
            }],
        );
        let out_chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "pixel_art_8bit",
                    values: vec![4.0, 8.0],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.3],
                },
            ],
        );

        // After positive brightness adjustment, output should be brighter.
        let art_r = out_art_only.get_pixel(0, 0)[0] as i32;
        let chain_r = out_chained.get_pixel(0, 0)[0] as i32;
        assert!(
            chain_r > art_r,
            "chained brightness should increase pixel value: art={art_r}, chained={chain_r}"
        );
    }
}
