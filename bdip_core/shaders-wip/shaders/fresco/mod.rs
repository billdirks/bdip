use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension, ParamKind, PassDef, PassInput,
    PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Fresco shader.
///
/// - `strength`:    Blend weight of the fresco effect. 0.0 = identity (no effect).
/// - `matte`:       How strongly colors are desaturated toward an earthy matte palette.
///   0.0 = original saturation, 1.0 = fully matte/desaturated.
/// - `texture_scale`: UV scale for the plaster grain texture; higher values zoom in.
/// - `_padding`:    Pad struct to 16 bytes for WebGPU uniform alignment.
///
/// # Identity design
///
/// `strength` defaults to 0.0, which blends the fresco composite at weight 0 —
/// producing a pure passthrough regardless of the other slider values.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrescoParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full fresco look.
    pub strength: f32,
    /// Matte desaturation strength: 0.0 = no change, 1.0 = fully matte palette.
    pub matte: f32,
    /// UV scale for the plaster grain overlay (1.0 = natural texture size).
    pub texture_scale: f32,
    pub _padding: f32,
}

impl TransformShader for FrescoParams {
    const ID: &'static str = "fresco";
    const DISPLAY_NAME: &'static str = "Fresco";
    const DESCRIPTION: &'static str = "Simulates a Renaissance fresco painting: matte earthy \
         tones, soft contrast reduction, and a plaster grain texture overlaid on the image.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend intensity of the fresco effect. \
                 0.0 leaves the image completely unchanged (identity).",
        },
        SliderDef {
            name: "Matte",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Desaturation toward an earthy matte palette. \
                 0.0 = original colors, 1.0 = fully matte.",
        },
        SliderDef {
            name: "Texture Scale",
            min: 0.5,
            max: 4.0,
            default: 1.0,
            description: "UV scale multiplier for the plaster grain texture. \
                 Higher values zoom in, lower values tile more finely.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "fresco",
        wgsl_source: include_str!("fresco.wgsl"),
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
            strength: values[0],
            matte: values[1],
            texture_scale: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<FrescoParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ── Registry / metadata ──────────────────────────────────────────────────

    #[test]
    fn test_fresco_registry_entry_exists() {
        assert!(registry_by_id("fresco").is_some());
    }

    #[test]
    fn test_fresco_registry_metadata() {
        let reg = registry_by_id("fresco").unwrap();
        assert_eq!(reg.meta.display_name, "Fresco");
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend intensity of the fresco effect. \
                         0.0 leaves the image completely unchanged (identity).",
                },
                SliderDef {
                    name: "Matte",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Desaturation toward an earthy matte palette. \
                         0.0 = original colors, 1.0 = fully matte.",
                },
                SliderDef {
                    name: "Texture Scale",
                    min: 0.5,
                    max: 4.0,
                    default: 1.0,
                    description: "UV scale multiplier for the plaster grain texture. \
                         Higher values zoom in, lower values tile more finely.",
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
    fn test_fresco_make_uniform_known_value() {
        let reg = registry_by_id("fresco").unwrap();
        let bytes = (reg.make_uniform)(&[0.7, 0.4, 2.0]);
        let expected = bytemuck::bytes_of(&FrescoParams {
            strength: 0.7,
            matte: 0.4,
            texture_scale: 2.0,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: output must equal the source pixel-for-pixel.
    #[test]
    fn test_fresco_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 25000, 18000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fresco",
                values: vec![0.0, 0.5, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 25000).abs() <= 128,
                "R: expected ~25000 at strength=0, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 18000).abs() <= 128,
                "G: expected ~18000 at strength=0, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 40000).abs() <= 128,
                "B: expected ~40000 at strength=0, got {}",
                pixel[2]
            );
        }
    }

    /// Alpha must pass through unchanged regardless of strength.
    #[test]
    fn test_fresco_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fresco",
                values: vec![1.0, 0.5, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    /// Full strength with full matte must desaturate a colored image.
    /// A vivid blue input should have its R and B channels pulled closer together.
    #[test]
    fn test_fresco_full_matte_reduces_saturation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Vivid blue: high B, low R, mid G.
        let img = make_solid_image(4, 4, 5000, 20000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fresco",
                values: vec![1.0, 1.0, 1.0],
            }],
        );
        // After full matte, R should be higher and B lower than the source —
        // channels pulled toward luminance, reducing the R-B spread.
        let src_spread = 60000i32 - 5000i32;
        for pixel in out.pixels() {
            let out_spread = (pixel[2] as i32 - pixel[0] as i32).abs();
            assert!(
                out_spread < src_spread,
                "matte=1.0 must reduce channel spread; src_spread={src_spread}, \
                 out_spread={out_spread}"
            );
        }
    }

    /// Different texture_scale values must produce different grain patterns.
    #[test]
    fn test_fresco_texture_scale_changes_pattern() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let out_a = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fresco",
                values: vec![1.0, 0.0, 1.0],
            }],
        );
        let out_b = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fresco",
                values: vec![1.0, 0.0, 2.0],
            }],
        );
        let any_different = out_a
            .pixels()
            .zip(out_b.pixels())
            .any(|(a, b)| (a[0] as i32 - b[0] as i32).unsigned_abs() > 64);
        assert!(
            any_different,
            "different texture_scale values must produce different grain patterns"
        );
    }

    /// Chaining with brightness at identity must not corrupt the output.
    #[test]
    fn test_fresco_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "fresco",
                    values: vec![0.5, 0.5, 1.0],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 after chaining");
        }
    }

    /// Two runs with identical inputs must produce bit-identical outputs.
    #[test]
    fn test_fresco_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "fresco",
            values: vec![0.8, 0.5, 1.5],
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
            assert_eq!(p1, p2, "outputs must be pixel-identical across runs");
        }
    }
}
