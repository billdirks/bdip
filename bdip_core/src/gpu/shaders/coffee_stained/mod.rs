use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CoffeeStainedParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for CoffeeStainedParams {
    const ID: &'static str = "coffee_stained";
    const DISPLAY_NAME: &'static str = "Coffee Stained";
    const DESCRIPTION: &'static str = "Simulates coffee or tea stains on a photograph using procedural noise-based stain shapes \
         with a warm brown multiplicative tint.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Intensity of the stain effect; 0 is unchanged, 1 is the full coffee-stain look.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "coffee_stained",
        wgsl_source: include_str!("coffee_stained.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    CoffeeStainedParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_coffee_stained_registry_entry_exists() {
        assert!(registry_by_id("coffee_stained").is_some());
    }

    #[test]
    fn test_coffee_stained_registry_metadata() {
        let reg = registry_by_id("coffee_stained").unwrap();
        assert_eq!(reg.meta.display_name, "Coffee Stained");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Intensity of the stain effect; 0 is unchanged, 1 is the full coffee-stain \
                     look.",
            }])
        );
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
    }

    #[test]
    fn test_coffee_stained_make_uniform_known_value() {
        let reg = registry_by_id("coffee_stained").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&CoffeeStainedParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_coffee_stained_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "coffee_stained",
                values: vec![0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: strength=0 must return original, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 16384).abs() <= 64,
                "G: strength=0 must return original, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 8192).abs() <= 64,
                "B: strength=0 must return original, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_coffee_stained_full_strength_warms_image() {
        // A white input with full strength must be warmed/tinted: R should be
        // brighter than B after the brownish multiplicative tint.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "coffee_stained",
                values: vec![1.0],
            }],
        );
        let mean_r: f64 = out.pixels().map(|p| p[0] as f64).sum::<f64>() / 256.0;
        let mean_b: f64 = out.pixels().map(|p| p[2] as f64).sum::<f64>() / 256.0;
        assert!(
            mean_r > mean_b,
            "full strength stain on white must produce warmer (R>B) output, R={mean_r:.0} B={mean_b:.0}"
        );
    }

    #[test]
    fn test_coffee_stained_full_strength_darkens_image() {
        // Multiplicative tint with values < 1 must darken a white image.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "coffee_stained",
                values: vec![1.0],
            }],
        );
        let mean_r: f64 = out.pixels().map(|p| p[0] as f64).sum::<f64>() / 256.0;
        assert!(
            mean_r < 65535.0,
            "full strength stain must darken a white image, mean_r={mean_r:.0}"
        );
    }

    #[test]
    fn test_coffee_stained_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "coffee_stained",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    #[test]
    fn test_coffee_stained_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 32767, 32767);
        let params = vec![0.8f32];
        let out_1 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "coffee_stained",
                values: params.clone(),
            }],
        );
        let out_2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "coffee_stained",
                values: params,
            }],
        );
        for (a, b) in out_1.pixels().zip(out_2.pixels()) {
            assert_eq!(a, b, "identical params must produce identical output");
        }
    }

    #[test]
    fn test_coffee_stained_chaining_with_brightness() {
        // Verify that the shader output can be fed into another shader without
        // errors, confirming correct texture format and pipeline wiring.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "coffee_stained",
                    values: vec![0.5],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.1],
                },
            ],
        );
        // After chaining, output must still be a valid (non-black) image.
        let any_nonzero = out.pixels().any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0);
        assert!(any_nonzero, "chained output must contain non-zero pixels");
    }
}
