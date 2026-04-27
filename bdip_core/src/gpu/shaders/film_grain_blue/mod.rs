use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension, ParamKind, PassDef, PassInput,
    PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FilmGrainBlueParams {
    pub amount: f32,
    pub variation: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for FilmGrainBlueParams {
    const ID: &'static str = "film_grain_blue";
    const DISPLAY_NAME: &'static str = "Film Grain (Blue)";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Amount",
            min: 0.0,
            max: 0.1,
            default: 0.0,
        },
        SliderDef {
            name: "Variation",
            min: 0.0,
            max: 1.0,
            default: 0.0,
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "film_grain_blue",
        wgsl_source: include_str!("film_grain_blue.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[AuxTextureDef {
            name: "blue_noise_128",
            dimension: AuxTextureDimension::D2,
            filter: AuxSamplerFilter::Linear,
        }],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            amount: values[0],
            variation: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    FilmGrainBlueParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_film_grain_blue_registry_entry_exists() {
        assert!(registry_by_id("film_grain_blue").is_some());
    }

    #[test]
    fn test_film_grain_blue_registry_metadata() {
        let reg = registry_by_id("film_grain_blue").unwrap();
        assert_eq!(reg.meta.display_name, "Film Grain (Blue)");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Amount",
                    min: 0.0,
                    max: 0.1,
                    default: 0.0,
                },
                SliderDef {
                    name: "Variation",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            ])
        );
    }

    #[test]
    fn test_film_grain_blue_make_uniform_known_value() {
        let reg = registry_by_id("film_grain_blue").unwrap();
        let bytes = (reg.make_uniform)(&[0.05, 0.3]);
        let expected = bytemuck::bytes_of(&FilmGrainBlueParams {
            amount: 0.05,
            variation: 0.3,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_film_grain_blue_zero_amount_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "film_grain_blue",
                values: vec![0.0, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 32767).abs() <= 64,
                "G: expected ~32767, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 32767).abs() <= 64,
                "B: expected ~32767, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_film_grain_blue_nonzero_amount_perturbs_pixels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "film_grain_blue",
                values: vec![0.1, 0.0],
            }],
        );
        let any_perturbed = out
            .pixels()
            .any(|p| (p[0] as i32 - 32767).unsigned_abs() > 128);
        assert!(
            any_perturbed,
            "nonzero amount must perturb at least one pixel"
        );
    }

    #[test]
    fn test_film_grain_blue_variation_changes_pattern() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 32767, 32767);
        let out_a = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "film_grain_blue",
                values: vec![0.1, 0.2],
            }],
        );
        let out_b = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "film_grain_blue",
                values: vec![0.1, 0.7],
            }],
        );
        let any_different = out_a
            .pixels()
            .zip(out_b.pixels())
            .any(|(a, b)| (a[0] as i32 - b[0] as i32).unsigned_abs() > 128);
        assert!(
            any_different,
            "different variation values must produce different grain patterns"
        );
    }

    #[test]
    fn test_film_grain_blue_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 32767, 32767);
        let params = vec![0.1, 0.5];
        let out_1 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "film_grain_blue",
                values: params.clone(),
            }],
        );
        let out_2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "film_grain_blue",
                values: params,
            }],
        );
        for (a, b) in out_1.pixels().zip(out_2.pixels()) {
            assert_eq!(a, b, "identical params must produce identical output");
        }
    }

    #[test]
    fn test_film_grain_blue_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "film_grain_blue",
                values: vec![0.1, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    #[test]
    fn test_film_grain_blue_black_pixels_have_minimal_grain() {
        // sqrt(luma) = sqrt(0) = 0, so grain weight is zero for black pixels.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "film_grain_blue",
                values: vec![0.1, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32).abs() <= 8,
                "black pixel R must stay near 0, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32).abs() <= 8,
                "black pixel G must stay near 0, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32).abs() <= 8,
                "black pixel B must stay near 0, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_film_grain_blue_requires_aux_texture() {
        let reg = registry_by_id("film_grain_blue").unwrap();
        let has_blue_noise = reg
            .meta
            .passes
            .iter()
            .flat_map(|p| p.aux_textures)
            .any(|a| a.name == "blue_noise_128");
        assert!(
            has_blue_noise,
            "film_grain_blue must declare 'blue_noise_128' in its aux_textures"
        );
    }

    #[test]
    fn test_film_grain_blue_missing_aux_returns_error() {
        use crate::error::BdipError;
        use crate::gpu::assets::AuxTextureCache;

        // Inventory-registered assets cannot be unregistered at runtime, so we test
        // the error path with a deliberately non-existent name. The same code path
        // in `apply` would fire if "blue_noise_128" were ever missing.
        let engine = GpuEngine::new().unwrap();
        let mut cache = AuxTextureCache::new();
        let err = cache
            .get_or_upload(&engine.device, &engine.queue, "blue_noise_128_missing")
            .unwrap_err();
        assert!(
            matches!(err, BdipError::MissingAuxTexture(_)),
            "missing aux texture must return BdipError::MissingAuxTexture, got {err:?}"
        );
    }
}
