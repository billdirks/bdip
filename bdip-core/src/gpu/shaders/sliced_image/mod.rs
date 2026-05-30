use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SlicedImageParams {
    /// Number of horizontal slices. Range [1, 50]. Default 10 is visually neutral
    /// in combination with offset=0.0 (identity).
    pub slice_count: f32,
    /// Horizontal UV offset per slice. Range [0.0, 0.5]. Default 0.0 = identity.
    pub slice_offset: f32,
    /// Alternating direction flag. 0.0 = all slices shift in the same direction;
    /// 1.0 = alternating slices shift left/right. Default 1.0 (alternating on) is
    /// still identity when slice_offset is 0.0.
    pub alternating_direction: f32,
    pub _padding: f32,
}

impl TransformShader for SlicedImageParams {
    const ID: &'static str = "sliced_image";
    const DISPLAY_NAME: &'static str = "Sliced Image";
    const DESCRIPTION: &'static str = "Divides the image into horizontal slices and offsets alternating slices \
         horizontally, creating a cut-and-shifted fragmented appearance.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Slice Count",
            min: 1.0,
            max: 50.0,
            default: 10.0,
            description: "Number of horizontal slices. Higher values produce thinner, \
                more numerous bands.",
        },
        SliderDef {
            name: "Slice Offset",
            min: 0.0,
            max: 0.5,
            default: 0.0,
            description: "Horizontal UV shift per slice in [0, 0.5]. 0.0 = no shift \
                (identity). UVs wrap, so shifted pixels always show image content.",
        },
        SliderDef {
            name: "Alternating Direction",
            min: 0.0,
            max: 1.0,
            default: 1.0,
            description: "0.0 = all slices shift in the same direction; 1.0 = odd \
                slices shift right and even slices shift left.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "sliced_image",
        wgsl_source: include_str!("sliced_image.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            slice_count: values[0],
            slice_offset: values[1],
            alternating_direction: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    SlicedImageParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // -------------------------------------------------------------------------
    // Registry tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sliced_image_registry_entry_exists() {
        assert!(registry_by_id("sliced_image").is_some());
    }

    #[test]
    fn test_sliced_image_registry_metadata() {
        let reg = registry_by_id("sliced_image").unwrap();
        assert_eq!(reg.meta.display_name, "Sliced Image");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Slice Count",
                    min: 1.0,
                    max: 50.0,
                    default: 10.0,
                    description: "Number of horizontal slices. Higher values produce thinner, \
                        more numerous bands.",
                },
                SliderDef {
                    name: "Slice Offset",
                    min: 0.0,
                    max: 0.5,
                    default: 0.0,
                    description: "Horizontal UV shift per slice in [0, 0.5]. 0.0 = no shift \
                        (identity). UVs wrap, so shifted pixels always show image content.",
                },
                SliderDef {
                    name: "Alternating Direction",
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                    description: "0.0 = all slices shift in the same direction; 1.0 = odd \
                        slices shift right and even slices shift left.",
                },
            ])
        );
    }

    #[test]
    fn test_sliced_image_make_uniform_known_value() {
        let reg = registry_by_id("sliced_image").unwrap();
        let bytes = (reg.make_uniform)(&[20.0, 0.25, 1.0]);
        let expected = bytemuck::bytes_of(&SlicedImageParams {
            slice_count: 20.0,
            slice_offset: 0.25,
            alternating_direction: 1.0,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip tests
    // -------------------------------------------------------------------------

    /// At slice_offset=0.0 no horizontal shift is applied regardless of slice_count
    /// or alternating_direction, so the output must match the source exactly.
    #[test]
    fn test_sliced_image_identity_at_zero_offset() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 40000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sliced_image",
                values: vec![10.0, 0.0, 1.0],
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

    /// A solid-color image is invariant to any horizontal UV shift because every
    /// pixel has the same value. Wrapping the UV within the image still samples the
    /// same color, so the output must be identical to the input even at max offset.
    #[test]
    fn test_sliced_image_solid_color_invariant_to_offset() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 50000, 10000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sliced_image",
                values: vec![10.0, 0.5, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 50000).abs() <= 64,
                "R: expected ~50000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 10000).abs() <= 64,
                "G: expected ~10000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 30000).abs() <= 64,
                "B: expected ~30000, got {}",
                pixel[2]
            );
        }
    }

    /// Alpha channel must pass through unchanged regardless of slice parameters.
    #[test]
    fn test_sliced_image_alpha_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sliced_image",
                values: vec![5.0, 0.25, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    /// Extreme slice_count (50) must not panic and must produce an image of the
    /// correct dimensions.
    #[test]
    fn test_sliced_image_extreme_slice_count_does_not_panic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sliced_image",
                values: vec![50.0, 0.3, 1.0],
            }],
        );

        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
    }

    /// Maximum slice_offset (0.5) must not panic and must produce an image of the
    /// correct dimensions.
    #[test]
    fn test_sliced_image_max_offset_does_not_panic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 10000, 20000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sliced_image",
                values: vec![10.0, 0.5, 0.0],
            }],
        );

        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
    }

    /// With alternating_direction=0.0, all slices shift in the same direction.
    /// A solid-color image is still invariant, confirming the non-alternating path
    /// executes without error.
    #[test]
    fn test_sliced_image_non_alternating_mode_does_not_panic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 40000, 40000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sliced_image",
                values: vec![5.0, 0.25, 0.0],
            }],
        );

        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
    }

    /// Chaining sliced_image with brightness must not panic and must preserve alpha.
    #[test]
    fn test_sliced_image_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "sliced_image",
                    values: vec![5.0, 0.1, 1.0],
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
