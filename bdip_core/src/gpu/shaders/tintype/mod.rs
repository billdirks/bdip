use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TintypeParams {
    /// Effect strength: 0.0 = identity (original image), 1.0 = full tintype look.
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for TintypeParams {
    const ID: &'static str = "tintype";
    const DISPLAY_NAME: &'static str = "Tintype";
    const DESCRIPTION: &'static str = "Simulates the dark pewter-toned, high-contrast look of Civil War era tintype \
         photography with strong radial vignette and coarse iron-plate grit texture.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend strength of the effect; 0 is the original image, 1 is full effect.",
    }]);
    const PASSES: &'static [PassDef] = &[
        // Pass 0: desaturation, high contrast, warm pewter tint.
        PassDef {
            label: "tintype_tone",
            wgsl_source: include_str!("tintype_pass0.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("toned"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        // Pass 1: strong radial vignette applied to the toned image.
        PassDef {
            label: "tintype_vignette",
            wgsl_source: include_str!("tintype_pass1.wgsl"),
            inputs: &[PassInput::Scratch("toned")],
            output: PassOutput::Scratch("vignetted"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        // Pass 2: coarse procedural grit overlay and final blend with source.
        PassDef {
            label: "tintype_grit",
            wgsl_source: include_str!("tintype_pass2.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("vignetted")],
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

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<TintypeParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_tintype_registry_entry_exists() {
        assert!(registry_by_id("tintype").is_some());
    }

    #[test]
    fn test_tintype_registry_metadata() {
        let reg = registry_by_id("tintype").unwrap();
        assert_eq!(reg.meta.display_name, "Tintype");
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
            3,
            "Tintype must have exactly 3 passes"
        );
    }

    #[test]
    fn test_tintype_make_uniform_known_value() {
        let reg = registry_by_id("tintype").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&TintypeParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_tintype_zero_strength_is_identity() {
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
                shader_id: "tintype",
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
    fn test_tintype_full_strength_desaturates() {
        // At strength=1.0 a fully saturated colour image must appear largely
        // desaturated. A pure red input (R=max, G=0, B=0) must produce an output
        // where R, G, B are significantly closer to each other than in the source.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 65535, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tintype",
                values: vec![1.0],
            }],
        );
        // Source R-G spread is 65535. After near-complete desaturation the spread
        // must collapse substantially — require within 8000 of each other.
        let pixel = out.get_pixel(2, 2);
        let rg_diff = (pixel[0] as i32 - pixel[1] as i32).abs();
        assert!(
            rg_diff < 8000,
            "full-strength tintype must largely desaturate: R={}, G={}, diff={}",
            pixel[0],
            pixel[1],
            rg_diff
        );
    }

    #[test]
    fn test_tintype_full_strength_has_warm_pewter_tint() {
        // The tintype tint produces warm dark-pewter tone: R > B on a neutral grey.
        // This is the opposite of Daguerreotype's cool silver-blue (B > R).
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a bright neutral grey to minimise vignette contribution at center.
        let img = make_solid_image(32, 32, 40000, 40000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tintype",
                values: vec![1.0],
            }],
        );
        // Sample center pixel to avoid the heavy vignette at edges.
        let pixel = out.get_pixel(15, 15);
        assert!(
            pixel[0] > pixel[2],
            "full-strength tintype must have warm pewter tint (R > B): R={}, B={}",
            pixel[0],
            pixel[2]
        );
    }

    #[test]
    fn test_tintype_full_strength_vignette_darkens_corners() {
        // At full strength, corner pixels must be substantially darker than center
        // pixels due to the aggressive vignette (starts at 0.30 vs Daguerreotype 0.38).
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 40000, 40000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tintype",
                values: vec![1.0],
            }],
        );
        let center = out.get_pixel(15, 15)[0] as i32;
        let corner = out.get_pixel(0, 0)[0] as i32;
        assert!(
            corner < center - 2000,
            "corner must be significantly darker than center: center={center}, corner={corner}"
        );
    }

    #[test]
    fn test_tintype_vignette_stronger_than_daguerreotype() {
        // Tintype vignette (start 0.30) must darken corners more aggressively than
        // Daguerreotype (start 0.38) on an identical input.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 40000, 40000, 40000);

        let tintype_out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tintype",
                values: vec![1.0],
            }],
        );
        let daguerreotype_out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "daguerreotype",
                values: vec![1.0],
            }],
        );

        let tintype_corner = tintype_out.get_pixel(0, 0)[0] as i32;
        let dag_corner = daguerreotype_out.get_pixel(0, 0)[0] as i32;
        assert!(
            tintype_corner <= dag_corner,
            "tintype vignette must darken corners at least as much as daguerreotype: \
             tintype={tintype_corner}, daguerreotype={dag_corner}"
        );
    }

    #[test]
    fn test_tintype_full_strength_increases_contrast() {
        // A bright near-white pixel (55000) must remain above 45000 after the
        // S-curve contrast boost, confirming highlights are not crushed to black.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a larger image and sample the center so vignette doesn't affect the result.
        let img = make_solid_image(32, 32, 55000, 55000, 55000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tintype",
                values: vec![1.0],
            }],
        );
        let pixel = out.get_pixel(15, 15);
        assert!(
            pixel[0] > 45000,
            "highlight pixel must remain bright with contrast boost: R={}",
            pixel[0]
        );
    }

    #[test]
    fn test_tintype_alpha_preserved() {
        // All three passes must leave the alpha channel unchanged.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tintype",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be preserved across all three passes"
            );
        }
    }

    #[test]
    fn test_tintype_chained_with_brightness() {
        // Tintype at identity (strength=0) followed by a brightness no-op must equal
        // brightness alone — confirms no colour shift leaks out at zero strength.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 20000, 20000, 20000);

        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "tintype",
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
    fn test_tintype_deterministic() {
        // Identical inputs and parameters must produce bit-identical output across two
        // runs — confirms the coordinate-hashed grit is deterministic, not time-seeded.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "tintype",
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
