use crate::gpu::shaders::{ParamKind, ShaderMeta, SliderDef, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HighlightsParams {
    pub amt: f32,
    pub range: f32,
    pub end: f32,
    pub _padding: f32,
}

impl TransformShader for HighlightsParams {
    const META: ShaderMeta = ShaderMeta {
        id: "highlights",
        display_name: "Highlights",
        wgsl_source: include_str!("highlights.wgsl"),
        param: ParamKind::Sliders(&[
            SliderDef {
                name: "Amount",
                min: -1.0,
                max: 1.0,
                default: 0.0,
            },
            SliderDef {
                name: "Range",
                min: 0.0,
                max: 1.0,
                default: 0.6,
            },
            SliderDef {
                name: "End",
                min: 0.0,
                max: 1.0,
                default: 0.95,
            },
        ]),
    };

    fn from_values(values: &[f32]) -> Self {
        Self {
            amt: values[0],
            range: values[1],
            end: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    HighlightsParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_highlights_registry_entry_exists() {
        assert!(registry_by_id("highlights").is_some());
    }

    #[test]
    fn test_highlights_registry_metadata() {
        let reg = registry_by_id("highlights").unwrap();
        assert_eq!(reg.meta.display_name, "Highlights");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Amount",
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
                SliderDef {
                    name: "Range",
                    min: 0.0,
                    max: 1.0,
                    default: 0.6,
                },
                SliderDef {
                    name: "End",
                    min: 0.0,
                    max: 1.0,
                    default: 0.95,
                },
            ])
        );
    }

    #[test]
    fn test_highlights_make_uniform_known_value() {
        let reg = registry_by_id("highlights").unwrap();
        let bytes = (reg.make_uniform)(&[0.5, 0.6, 0.95]);
        let expected = bytemuck::bytes_of(&HighlightsParams {
            amt: 0.5,
            range: 0.6,
            end: 0.95,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_highlights_identity_zero_amount() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 10000, 5000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "highlights",
                values: vec![0.0, 0.6, 0.95],
            }],
        );
        for pixel in out.pixels() {
            assert!((pixel[0] as i32 - 20000).abs() <= 64);
            assert!((pixel[1] as i32 - 10000).abs() <= 64);
            assert!((pixel[2] as i32 - 5000).abs() <= 64);
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_highlights_darkens_bright_pixels() {
        // amt=-1.0 with range=0, end=0.5 → all bright pixels get W_h=1.0 → output approaches 0.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 60000, 60000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "highlights",
                values: vec![-1.0, 0.0, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert!(pixel[0] < 60000, "expected pixel darker than input");
        }
    }

    #[test]
    fn test_highlights_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 60000, 60000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "highlights",
                values: vec![-1.0, 0.6, 0.95],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_highlights_chained_with_brightness() {
        // Highlights identity (amt=0) followed by brightness zero should equal brightness zero alone.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 20000, 20000);

        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "highlights",
                    values: vec![0.0, 0.6, 0.95],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );
        let brightness_only = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "brightness",
                values: vec![0.0],
            }],
        );

        for (a, b) in chained.pixels().zip(brightness_only.pixels()) {
            assert!((a[0] as i32 - b[0] as i32).abs() <= 64);
            assert!((a[1] as i32 - b[1] as i32).abs() <= 64);
            assert!((a[2] as i32 - b[2] as i32).abs() <= 64);
            assert_eq!(a[3], b[3]);
        }
    }
}
