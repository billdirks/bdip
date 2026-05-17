use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CoffeeStainedParams {
    pub strength: f32,
    pub ring_width: f32,
    pub inner_clarity: f32,
    pub _padding: f32,
}

impl TransformShader for CoffeeStainedParams {
    const ID: &'static str = "coffee_stained";
    const DISPLAY_NAME: &'static str = "Coffee Stained";
    const DESCRIPTION: &'static str = "Simulates coffee or tea stains on a photograph using procedural noise-based stain shapes \
         with a warm brown multiplicative tint.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Intensity of the stain effect; 0 is unchanged, 1 is the full coffee-stain \
                 look.",
        },
        SliderDef {
            name: "Ring Width",
            min: 0.0,
            max: 1.0,
            default: 0.3,
            description: "Thickness of the dark ring edge. Lower values produce thin, defined edges; \
                 higher values spread the darkening wider.",
        },
        SliderDef {
            name: "Inner Clarity",
            min: 0.0,
            max: 1.0,
            default: 0.7,
            description: "How clear the center of each stain is. 1.0 = center nearly unchanged \
                 (realistic ring); 0.0 = center also darkened (filled stain).",
        },
    ]);
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
            ring_width: values[1],
            inner_clarity: values[2],
            _padding: 0.0,
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
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Intensity of the stain effect; 0 is unchanged, 1 is the full \
                         coffee-stain look.",
                },
                SliderDef {
                    name: "Ring Width",
                    min: 0.0,
                    max: 1.0,
                    default: 0.3,
                    description: "Thickness of the dark ring edge. Lower values produce thin, \
                         defined edges; higher values spread the darkening wider.",
                },
                SliderDef {
                    name: "Inner Clarity",
                    min: 0.0,
                    max: 1.0,
                    default: 0.7,
                    description: "How clear the center of each stain is. 1.0 = center nearly \
                         unchanged (realistic ring); 0.0 = center also darkened (filled stain).",
                },
            ])
        );
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
    }

    #[test]
    fn test_coffee_stained_make_uniform_known_value() {
        let reg = registry_by_id("coffee_stained").unwrap();
        let bytes = (reg.make_uniform)(&[0.75, 0.3, 0.7]);
        let expected = bytemuck::bytes_of(&CoffeeStainedParams {
            strength: 0.75,
            ring_width: 0.3,
            inner_clarity: 0.7,
            _padding: 0.0,
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
                values: vec![0.0, 0.3, 0.7],
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
                values: vec![1.0, 0.3, 0.7],
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
                values: vec![1.0, 0.3, 0.7],
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
                values: vec![1.0, 0.3, 0.7],
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
        let params = vec![0.8f32, 0.3, 0.7];
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
                    values: vec![0.5, 0.3, 0.7],
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

    #[test]
    fn test_coffee_stained_ring_effect_darker_at_edge() {
        // Verify the coffee ring effect: the ring perimeter is darker than the
        // stain center.  This matches real coffee physics where particles
        // concentrate at the ring edge during evaporation.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(128, 128, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "coffee_stained",
                values: vec![1.0, 0.3, 0.7],
            }],
        );
        // CENTRE_0 = (0.18, 0.22) → pixel (23, 28) in 128×128.
        // RING_RADIUS_0 = 0.15 → ~19 pixels from center.
        let center_pixel = out.get_pixel(23, 28);
        let edge_pixel = out.get_pixel(23 + 19, 28);
        assert!(
            edge_pixel[0] < center_pixel[0],
            "ring edge should be darker than center: edge R={}, center R={}",
            edge_pixel[0],
            center_pixel[0]
        );
    }

    #[test]
    fn test_coffee_stained_ring_width_affects_edge_thickness() {
        // Verify that a wider ring_width darkens more pixels than a thin one,
        // because the exponential falloff from the ring edge spreads further.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(64, 64, 65535, 65535, 65535);

        let out_thin = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "coffee_stained",
                values: vec![1.0, 0.1, 0.8],
            }],
        );
        let out_wide = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "coffee_stained",
                values: vec![1.0, 0.5, 0.8],
            }],
        );

        // Count pixels noticeably darkened (R < 90 % of white).
        let count_darkened =
            |img: &crate::Rgba16Image| img.pixels().filter(|p| p[0] < 59000).count();
        let thin_darkened = count_darkened(&out_thin);
        let wide_darkened = count_darkened(&out_wide);
        assert!(
            wide_darkened > thin_darkened,
            "wider ring_width should darken more pixels: thin={thin_darkened}, wide={wide_darkened}"
        );
    }

    #[test]
    fn test_coffee_stained_inner_clarity_affects_center() {
        // Verify that inner_clarity controls how dark the stain center is.
        // At 1.0 the center is nearly unchanged; at 0.0 it is fully darkened.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use 128×128 so that pixel (23, 28) falls on CENTRE_0 = (0.18, 0.22).
        let img = make_solid_image(128, 128, 65535, 65535, 65535);

        let out_clear = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "coffee_stained",
                values: vec![1.0, 0.3, 1.0],
            }],
        );
        let out_filled = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "coffee_stained",
                values: vec![1.0, 0.3, 0.0],
            }],
        );

        // Sample at the blob center — inner_clarity=1.0 must leave it lighter.
        let center_clear = out_clear.get_pixel(23, 28)[0];
        let center_filled = out_filled.get_pixel(23, 28)[0];
        assert!(
            center_clear > center_filled,
            "inner_clarity=1.0 should leave center lighter: clear={center_clear}, filled={center_filled}"
        );
    }
}
