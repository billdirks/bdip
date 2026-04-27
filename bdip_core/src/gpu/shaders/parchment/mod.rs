use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension, ParamKind, PassDef, PassInput,
    PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParchmentParams {
    pub intensity: f32,
    pub scale: f32,
    pub _padding1: f32,
    pub _padding2: f32,
}

impl TransformShader for ParchmentParams {
    const ID: &'static str = "parchment";
    const DISPLAY_NAME: &'static str = "Parchment";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Intensity",
            min: 0.0,
            max: 1.0,
            default: 0.0,
        },
        SliderDef {
            name: "Scale",
            min: 0.5,
            max: 4.0,
            default: 1.0,
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "parchment",
        wgsl_source: include_str!("parchment.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[AuxTextureDef {
            name: "paper_grain_256",
            dimension: AuxTextureDimension::D2,
            filter: AuxSamplerFilter::Linear,
        }],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            intensity: values[0],
            scale: values[1],
            _padding1: 0.0,
            _padding2: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    ParchmentParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_parchment_registry_entry_exists() {
        assert!(registry_by_id("parchment").is_some());
    }

    #[test]
    fn test_parchment_registry_metadata() {
        let reg = registry_by_id("parchment").unwrap();
        assert_eq!(reg.meta.display_name, "Parchment");
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Intensity",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
                SliderDef {
                    name: "Scale",
                    min: 0.5,
                    max: 4.0,
                    default: 1.0,
                },
            ])
        );
        assert_eq!(
            reg.meta.passes[0].aux_textures.len(),
            1,
            "must declare exactly 1 aux texture"
        );
    }

    #[test]
    fn test_parchment_intensity_zero_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "parchment",
                values: vec![0.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 128,
                "R: intensity=0 must return original within ±128, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 16384).abs() <= 128,
                "G: intensity=0 must return original within ±128, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 8192).abs() <= 128,
                "B: intensity=0 must return original within ±128, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_parchment_full_intensity_darkens_image() {
        // The paper grain texture has values in [0, 1]. Multiplicative blend
        // with a sub-1.0 paper texture must produce output at most as bright as
        // the input. A white input with intensity=1.0 will be darkened.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "parchment",
                values: vec![1.0, 1.0],
            }],
        );
        let mean_r: f64 = out.pixels().map(|p| p[0] as f64).sum::<f64>() / 256.0;
        assert!(
            mean_r < 65535.0,
            "full intensity parchment must darken a white image, mean_r={mean_r}"
        );
    }

    #[test]
    fn test_parchment_scale_changes_pattern() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out_a = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "parchment",
                values: vec![1.0, 1.0],
            }],
        );
        let out_b = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "parchment",
                values: vec![1.0, 2.0],
            }],
        );
        let any_different = out_a
            .pixels()
            .zip(out_b.pixels())
            .any(|(a, b)| (a[0] as i32 - b[0] as i32).unsigned_abs() > 64);
        assert!(
            any_different,
            "different scale values must produce different grain patterns"
        );
    }

    #[test]
    fn test_parchment_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "parchment",
                values: vec![1.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    #[test]
    fn test_parchment_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let params = vec![0.8f32, 1.5];
        let out_1 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "parchment",
                values: params.clone(),
            }],
        );
        let out_2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "parchment",
                values: params,
            }],
        );
        for (a, b) in out_1.pixels().zip(out_2.pixels()) {
            assert_eq!(a, b, "identical params must produce identical output");
        }
    }

    #[test]
    fn test_parchment_make_uniform_known_value() {
        let reg = registry_by_id("parchment").unwrap();
        let bytes = (reg.make_uniform)(&[0.5, 2.0]);
        let expected = bytemuck::bytes_of(&ParchmentParams {
            intensity: 0.5,
            scale: 2.0,
            _padding1: 0.0,
            _padding2: 0.0,
        });
        assert_eq!(bytes, expected);
    }
}
