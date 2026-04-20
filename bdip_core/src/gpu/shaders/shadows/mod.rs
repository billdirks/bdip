use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, SliderDef, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowsParams {
    pub amt: f32,
    pub range: f32,
    pub start: f32,
    pub _padding: f32,
}

impl TransformShader for ShadowsParams {
    const ID: &'static str = "shadows";
    const DISPLAY_NAME: &'static str = "Shadows";
    const PARAM: ParamKind = ParamKind::Sliders(&[
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
            default: 0.4,
        },
        SliderDef {
            name: "Start",
            min: 0.0,
            max: 1.0,
            default: 0.05,
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "shadows",
        wgsl_source: include_str!("shadows.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            amt: values[0],
            range: values[1],
            start: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ShadowsParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_shadows_registry_entry_exists() {
        assert!(registry_by_id("shadows").is_some());
    }

    #[test]
    fn test_shadows_registry_metadata() {
        let reg = registry_by_id("shadows").unwrap();
        assert_eq!(reg.meta.display_name, "Shadows");
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
                    default: 0.4,
                },
                SliderDef {
                    name: "Start",
                    min: 0.0,
                    max: 1.0,
                    default: 0.05,
                },
            ])
        );
    }

    #[test]
    fn test_shadows_make_uniform_known_value() {
        let reg = registry_by_id("shadows").unwrap();
        let bytes = (reg.make_uniform)(&[0.5, 0.4, 0.05]);
        let expected = bytemuck::bytes_of(&ShadowsParams {
            amt: 0.5,
            range: 0.4,
            start: 0.05,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_shadows_identity_zero_amount() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 10000, 5000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "shadows",
                values: vec![0.0, 0.4, 0.05],
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
    fn test_shadows_brightens_dark_pixels() {
        // amt=1.0 with full range covering all luminance → dark pixels should brighten.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 10000, 10000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "shadows",
                values: vec![1.0, 1.0, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(pixel[0] > 10000, "expected pixel brighter than input");
        }
    }

    #[test]
    fn test_shadows_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 10000, 10000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "shadows",
                values: vec![1.0, 0.4, 0.05],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_shadows_chained_with_brightness() {
        // Shadows identity (amt=0) followed by brightness zero should equal brightness zero alone.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 20000, 20000);

        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "shadows",
                    values: vec![0.0, 0.4, 0.05],
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
