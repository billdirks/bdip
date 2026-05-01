use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PixelateParams {
    pub block_size: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for PixelateParams {
    const ID: &'static str = "pixelate";
    const DISPLAY_NAME: &'static str = "Pixelate";
    const DESCRIPTION: &'static str = "Snaps UV coordinates to a grid, giving each block a uniform color \
         sampled from its top-left corner (pixelation effect).";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Block Size",
        min: 1.0,
        max: 128.0,
        default: 1.0,
        description: "Block size in pixels. 1.0 = identity (no pixelation); \
                      larger values produce coarser, blockier output.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "pixelate",
        wgsl_source: include_str!("pixelate.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            block_size: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<PixelateParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_pixelate_registry_entry_exists() {
        assert!(registry_by_id("pixelate").is_some());
    }

    #[test]
    fn test_pixelate_registry_metadata() {
        let reg = registry_by_id("pixelate").unwrap();
        assert_eq!(reg.meta.display_name, "Pixelate");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Block Size",
                min: 1.0,
                max: 128.0,
                default: 1.0,
                description: "Block size in pixels. 1.0 = identity (no pixelation); \
                              larger values produce coarser, blockier output.",
            }])
        );
    }

    #[test]
    fn test_pixelate_make_uniform_known_value() {
        let reg = registry_by_id("pixelate").unwrap();
        let bytes = (reg.make_uniform)(&[16.0]);
        let expected = bytemuck::bytes_of(&PixelateParams {
            block_size: 16.0,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    /// At block_size=1.0 (identity), every output pixel must equal its source pixel.
    /// A solid-color image ensures every pixel is verifiable regardless of UV mapping.
    #[test]
    fn test_pixelate_identity_at_block_size_one() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 40000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixelate",
                values: vec![1.0],
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

    /// With a block_size equal to the full image dimension, all output pixels should
    /// sample from the top-left corner, producing a uniform color equal to that corner.
    #[test]
    fn test_pixelate_full_image_block_produces_uniform_color() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Solid image: every pixel has the same value, so "uniform color" is trivially
        // verifiable even when different corners have the same pixel value.
        let img = make_solid_image(8, 8, 50000, 25000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixelate",
                values: vec![128.0], // block_size >> image dims, one block for whole image
            }],
        );

        // All pixels should have the same color (sampled from the single block's corner).
        let first = out.get_pixel(0, 0);
        for pixel in out.pixels() {
            assert_eq!(
                pixel[0], first[0],
                "R values must be uniform across one block"
            );
            assert_eq!(
                pixel[1], first[1],
                "G values must be uniform across one block"
            );
            assert_eq!(
                pixel[2], first[2],
                "B values must be uniform across one block"
            );
        }
    }

    /// Alpha channel must pass through unchanged regardless of pixelation strength.
    #[test]
    fn test_pixelate_alpha_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pixelate",
                values: vec![8.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    /// Chaining pixelate with brightness must not panic and must preserve alpha.
    #[test]
    fn test_pixelate_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "pixelate",
                    values: vec![4.0],
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
