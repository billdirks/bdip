use crate::gpu::shaders::{ParamKind, ShaderMeta, SliderDef, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VignetteParams {
    pub radius: f32,
    pub softness: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for VignetteParams {
    const META: ShaderMeta = ShaderMeta {
        id: "vignette",
        display_name: "Vignette",
        wgsl_source: include_str!("vignette.wgsl"),
        param: ParamKind::Sliders(&[
            SliderDef {
                name: "Radius",
                min: 0.0,
                max: 1.5,
                default: 0.8,
            },
            SliderDef {
                name: "Softness",
                min: 0.0,
                max: 1.0,
                default: 0.5,
            },
        ]),
    };

    fn from_values(values: &[f32]) -> Self {
        Self {
            radius: values[0],
            softness: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<VignetteParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_vignette_registry_entry_exists() {
        assert!(registry_by_id("vignette").is_some());
    }

    #[test]
    fn test_vignette_registry_metadata() {
        let reg = registry_by_id("vignette").unwrap();
        assert_eq!(reg.meta.display_name, "Vignette");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Radius",
                    min: 0.0,
                    max: 1.5,
                    default: 0.8
                },
                SliderDef {
                    name: "Softness",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5
                },
            ])
        );
    }

    #[test]
    fn test_vignette_make_uniform_known_value() {
        let reg = registry_by_id("vignette").unwrap();
        let bytes = (reg.make_uniform)(&[0.6, 0.3]);
        let expected = bytemuck::bytes_of(&VignetteParams {
            radius: 0.6,
            softness: 0.3,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_vignette_identity_large_radius() {
        // radius=1.5, softness=0: all pixels have d < 1.5, so v=1 → no change.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "vignette",
                values: vec![1.5, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!((pixel[0] as i32 - 32767).abs() <= 64);
            assert!((pixel[1] as i32 - 16384).abs() <= 64);
            assert!((pixel[2] as i32 - 8192).abs() <= 64);
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_vignette_zero_radius_produces_black() {
        // radius=0, softness=0: all pixels at d>0 have v=0 → black.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "vignette",
                values: vec![0.0, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[0], 0);
            assert_eq!(pixel[1], 0);
            assert_eq!(pixel[2], 0);
        }
    }

    #[test]
    fn test_vignette_alpha_preserved() {
        // Identity vignette — alpha channel must be unchanged.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "vignette",
                values: vec![1.5, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_vignette_chained_with_brightness() {
        // Vignette identity followed by brightness zero should equal brightness zero alone.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(2, 2, 20000, 20000, 20000);

        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "vignette",
                    values: vec![1.5, 0.0],
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
