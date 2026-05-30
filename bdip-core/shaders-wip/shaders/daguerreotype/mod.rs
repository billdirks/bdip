use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DaguerreotypeParams {
    /// Effect strength: 0.0 = identity, 1.0 = full daguerreotype look.
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for DaguerreotypeParams {
    const ID: &'static str = "daguerreotype";
    const DISPLAY_NAME: &'static str = "Daguerreotype";
    const DESCRIPTION: &'static str = "Simulates the silver-toned, high-contrast look of early 19th-century \
         daguerreotype photographs with metallic tint, strong vignette, and fine grain.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend strength of the effect; 0 is the original image, 1 is full effect.",
    }]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "daguerreotype_tone",
            wgsl_source: include_str!("daguerreotype_pass0.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("toned"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "daguerreotype_vignette_grain",
            wgsl_source: include_str!("daguerreotype_pass1.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("toned")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    DaguerreotypeParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_daguerreotype_registry_entry_exists() {
        assert!(registry_by_id("daguerreotype").is_some());
    }

    #[test]
    fn test_daguerreotype_registry_metadata() {
        let reg = registry_by_id("daguerreotype").unwrap();
        assert_eq!(reg.meta.display_name, "Daguerreotype");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend strength of the effect; 0 is the original image, 1 is full effect.",
            }])
        );
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "Daguerreotype must have exactly 2 passes"
        );
    }

    #[test]
    fn test_daguerreotype_make_uniform_known_value() {
        let reg = registry_by_id("daguerreotype").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&DaguerreotypeParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_daguerreotype_zero_strength_is_identity() {
        // At strength=0.0 the final pass blends 0% effect and 100% source,
        // so the output must be pixel-identical to the source within f16 tolerance.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 20000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "daguerreotype",
                values: vec![0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 20000).abs() <= 64,
                "G: expected ~20000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 10000).abs() <= 64,
                "B: expected ~10000, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_daguerreotype_full_strength_desaturates() {
        // At strength=1.0 a fully saturated color image should appear largely
        // desaturated. A pure red input (R=max, G=0, B=0) must produce an output
        // where the R, G, B channels are significantly closer to each other than
        // in the source.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 65535, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "daguerreotype",
                values: vec![1.0],
            }],
        );
        // Source R/G channel difference is 65535. After desaturation the difference
        // should collapse to a fraction of that — require within 8000 of each other.
        let pixel = out.get_pixel(2, 2);
        let rg_diff = (pixel[0] as i32 - pixel[1] as i32).abs();
        assert!(
            rg_diff < 8000,
            "full-strength daguerreotype must largely desaturate: R={}, G={}, diff={}",
            pixel[0],
            pixel[1],
            rg_diff
        );
    }

    #[test]
    fn test_daguerreotype_full_strength_increases_contrast() {
        // At strength=1.0 a mid-gray image run through tone processing should
        // have increased contrast relative to a simple grayscale conversion.
        // A bright near-white pixel (55000) must stay above 45000, confirming the
        // S-curve contrast boost is not collapsing highlights.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 55000, 55000, 55000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "daguerreotype",
                values: vec![1.0],
            }],
        );
        let pixel = out.get_pixel(2, 2);
        assert!(
            pixel[0] > 45000,
            "highlight pixel must remain bright with contrast boost: R={}",
            pixel[0]
        );
    }

    #[test]
    fn test_daguerreotype_full_strength_has_metallic_tint() {
        // The metallic tint shifts the grey point slightly blue-grey (B > R on a
        // neutral grey). On a mid-grey source, B must exceed R after full processing.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "daguerreotype",
                values: vec![1.0],
            }],
        );
        // Sample center pixels to avoid vignette contamination
        let pixel = out.get_pixel(4, 4);
        assert!(
            pixel[2] > pixel[0],
            "full-strength daguerreotype must add blue-grey metallic tint: R={}, B={}",
            pixel[0],
            pixel[2]
        );
    }

    #[test]
    fn test_daguerreotype_vignette_darkens_corners() {
        // At full strength, corner pixels must be significantly darker than center
        // pixels due to vignette. Use a uniform grey source to isolate the effect.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 40000, 40000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "daguerreotype",
                values: vec![1.0],
            }],
        );
        let center = out.get_pixel(15, 15)[0] as i32;
        let corner = out.get_pixel(0, 0)[0] as i32;
        assert!(
            corner < center - 2000,
            "corner must be darker than center due to vignette: center={center}, corner={corner}"
        );
    }

    #[test]
    fn test_daguerreotype_alpha_preserved() {
        // Both passes must leave the alpha channel unchanged.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "daguerreotype",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    #[test]
    fn test_daguerreotype_chained_with_brightness() {
        // Daguerreotype at identity followed by a brightness no-op must equal
        // brightness alone — confirms no extraneous color shift at strength=0.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 20000, 20000, 20000);

        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "daguerreotype",
                    values: vec![0.0],
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
            assert!(
                (a[0] as i32 - b[0] as i32).abs() <= 64,
                "chained R must equal brightness-only R"
            );
            assert!(
                (a[1] as i32 - b[1] as i32).abs() <= 64,
                "chained G must equal brightness-only G"
            );
            assert!(
                (a[2] as i32 - b[2] as i32).abs() <= 64,
                "chained B must equal brightness-only B"
            );
            assert_eq!(a[3], b[3], "alpha must be equal");
        }
    }

    #[test]
    fn test_daguerreotype_deterministic() {
        // Identical inputs and parameters must produce bit-identical output across
        // two runs. This confirms the hash-based grain is coordinate-deterministic.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "daguerreotype",
            values: vec![0.8],
        };
        let out1 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            std::slice::from_ref(&transform),
        );
        let out2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            std::slice::from_ref(&transform),
        );
        for (p1, p2) in out1.pixels().zip(out2.pixels()) {
            assert_eq!(p1, p2, "output must be pixel-identical across runs");
        }
    }
}
