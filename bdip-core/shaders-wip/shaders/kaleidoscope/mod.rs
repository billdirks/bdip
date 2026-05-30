use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KaleidoscopeParams {
    pub segments: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for KaleidoscopeParams {
    const ID: &'static str = "kaleidoscope";
    const DISPLAY_NAME: &'static str = "Kaleidoscope";
    const DESCRIPTION: &'static str =
        "Mirrors the image in polar coordinates to create a kaleidoscope pattern.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Segments",
        min: 1.0,
        max: 32.0,
        default: 1.0,
        description: "Number of mirror segments. 1.0 = single reflection (minimal effect); \
             higher values produce more repetitions.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "kaleidoscope",
        wgsl_source: include_str!("kaleidoscope.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            segments: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    KaleidoscopeParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_kaleidoscope_registry_entry_exists() {
        assert!(registry_by_id("kaleidoscope").is_some());
    }

    #[test]
    fn test_kaleidoscope_registry_metadata() {
        let reg = registry_by_id("kaleidoscope").unwrap();
        assert_eq!(reg.meta.display_name, "Kaleidoscope");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Segments",
                min: 1.0,
                max: 32.0,
                default: 1.0,
                description: "Number of mirror segments. 1.0 = single reflection (minimal effect); \
                     higher values produce more repetitions.",
            }])
        );
    }

    #[test]
    fn test_kaleidoscope_make_uniform_known_value() {
        let reg = registry_by_id("kaleidoscope").unwrap();
        let bytes = (reg.make_uniform)(&[8.0]);
        let expected = bytemuck::bytes_of(&KaleidoscopeParams {
            segments: 8.0,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    /// At segments=1.0 (default/identity) the solid-colour image must pass through
    /// largely unchanged. With a solid-colour source every pixel maps back to the
    /// same colour regardless of which wedge is chosen, so any polar mapping is a
    /// no-op for uniform images.
    #[test]
    fn test_kaleidoscope_identity_solid_image() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 40000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kaleidoscope",
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

    /// Alpha channel must be passed through unchanged for any number of segments.
    #[test]
    fn test_kaleidoscope_alpha_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kaleidoscope",
                values: vec![8.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    /// High segment count should produce valid output without panicking or
    /// generating out-of-range pixel values. Checks only alpha channel integrity
    /// since colour values are sampling-pattern-dependent on non-solid images.
    #[test]
    fn test_kaleidoscope_high_segment_count_no_panic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(8, 8, 50000, 10000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "kaleidoscope",
                values: vec![32.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 at max segments");
        }
    }

    /// Chaining kaleidoscope with another shader (brightness) must not panic and
    /// must keep the alpha channel intact.
    #[test]
    fn test_kaleidoscope_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "kaleidoscope",
                    values: vec![6.0],
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
