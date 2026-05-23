use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OldMapParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for OldMapParams {
    const ID: &'static str = "old_map";
    const DISPLAY_NAME: &'static str = "Old Map";
    const DESCRIPTION: &'static str = "Makes the image look like an antique map on aged parchment: \
         sepia-toned with a procedurally generated warm, grainy paper texture.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend factor for the old-map effect; 0 is unchanged, 1 is full effect.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "old_map",
        wgsl_source: include_str!("old_map.wgsl"),
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

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<OldMapParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_old_map_registry_entry_exists() {
        assert!(registry_by_id("old_map").is_some());
    }

    #[test]
    fn test_old_map_registry_metadata() {
        let reg = registry_by_id("old_map").unwrap();
        assert_eq!(reg.meta.display_name, "Old Map");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend factor for the old-map effect; 0 is unchanged, 1 is full effect.",
            }])
        );
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
    }

    #[test]
    fn test_old_map_passes_have_no_aux_textures() {
        let reg = registry_by_id("old_map").unwrap();
        assert_eq!(
            reg.meta.passes[0].aux_textures.len(),
            0,
            "old_map uses procedural parchment and must not declare any aux textures"
        );
    }

    #[test]
    fn test_old_map_make_uniform_known_value() {
        let reg = registry_by_id("old_map").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&OldMapParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_old_map_strength_zero_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a coloured image so we can verify all three channels.
        let img = make_solid_image(4, 4, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "old_map",
                values: vec![0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 128,
                "R: strength=0 must return original within ±128, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 16384).abs() <= 128,
                "G: strength=0 must return original within ±128, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 8192).abs() <= 128,
                "B: strength=0 must return original within ±128, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_old_map_full_strength_shifts_color_toward_sepia_warm() {
        // At full strength a grey input should produce a warm (R > G > B) output,
        // consistent with sepia-toning.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "old_map",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                pixel[0] >= pixel[1],
                "R should be >= G for warm sepia tone: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                pixel[1] >= pixel[2],
                "G should be >= B for warm sepia tone: G={}, B={}",
                pixel[1],
                pixel[2]
            );
        }
    }

    #[test]
    fn test_old_map_full_strength_changes_image() {
        // Full strength must visually change the image compared with strength=0.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out_zero = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "old_map",
                values: vec![0.0],
            }],
        );
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "old_map",
                values: vec![1.0],
            }],
        );
        let any_different = out_zero
            .pixels()
            .zip(out_full.pixels())
            .any(|(a, b)| (a[0] as i32 - b[0] as i32).unsigned_abs() > 64);
        assert!(
            any_different,
            "strength=1.0 must produce output different from strength=0.0"
        );
    }

    #[test]
    fn test_old_map_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "old_map",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    #[test]
    fn test_old_map_deterministic() {
        // Identical params on identical input must produce identical output.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let params = vec![0.8_f32];
        let out_1 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "old_map",
                values: params.clone(),
            }],
        );
        let out_2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "old_map",
                values: params,
            }],
        );
        for (a, b) in out_1.pixels().zip(out_2.pixels()) {
            assert_eq!(a, b, "identical params must produce identical output");
        }
    }

    #[test]
    fn test_old_map_partial_strength_is_between_identity_and_full() {
        // At strength=0.5, the output brightness should sit between the
        // strength=0 result and the strength=1 result (monotonic blend).
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);

        let out_zero = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "old_map",
                values: vec![0.0],
            }],
        );
        let out_half = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "old_map",
                values: vec![0.5],
            }],
        );
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "old_map",
                values: vec![1.0],
            }],
        );

        // Compute mean R across each result.
        let mean = |img: &image::ImageBuffer<image::Rgba<u16>, Vec<u16>>| -> f64 {
            let pixels: Vec<_> = img.pixels().collect();
            pixels.iter().map(|p| p[0] as f64).sum::<f64>() / pixels.len() as f64
        };

        let r0 = mean(&out_zero);
        let r1 = mean(&out_half);
        let r2 = mean(&out_full);

        let lo = r0.min(r2);
        let hi = r0.max(r2);
        assert!(
            r1 >= lo - 256.0 && r1 <= hi + 256.0,
            "half-strength mean R={r1:.0} must sit between zero-strength {r0:.0} \
             and full-strength {r2:.0} (±256 tolerance)"
        );
    }

    #[test]
    fn test_old_map_chained_with_brightness() {
        // Applying brightness before old_map must not panic or produce NaN/Inf.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "brightness",
                    values: vec![0.1],
                },
                Transform {
                    shader_id: "old_map",
                    values: vec![1.0],
                },
            ],
        );
        // The warm-tone property (R >= G >= B) must still hold after chaining.
        for pixel in out.pixels() {
            assert!(
                pixel[0] >= pixel[1],
                "R should be >= G after brightness+old_map: R={}, G={}",
                pixel[0],
                pixel[1]
            );
        }
    }
}
