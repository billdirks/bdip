use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension, ParamKind, PassDef, PassInput,
    PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Dust and Scratches shader.
///
/// Five floats are used (four for the uniform alignment rule):
/// - `strength`:        Overall blend weight. 0.0 = identity (no effect).
/// - `scratch_density`: Controls how many vertical scratch lines appear.
/// - `dust_amount`:     Controls the density of small dust-particle specks.
/// - `_padding`:        Pad to 16 bytes for WebGPU uniform alignment.
///
/// # Identity design
///
/// `strength` defaults to 0.0, which mixes the composite overlay at weight 0 —
/// producing a pure passthrough regardless of the other slider values.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustAndScratchesParams {
    /// Overall blend factor: 0.0 = source unchanged (identity), 1.0 = full effect.
    pub strength: f32,
    /// Frequency of vertical scratch lines. Range [0.0, 1.0]. 0.0 = no scratches.
    pub scratch_density: f32,
    /// Frequency of dust specks. Range [0.0, 1.0]. 0.0 = no dust.
    pub dust_amount: f32,
    pub _padding: f32,
}

impl TransformShader for DustAndScratchesParams {
    const ID: &'static str = "dust_and_scratches";
    const DISPLAY_NAME: &'static str = "Dust and Scratches";
    const DESCRIPTION: &'static str = "Simulates aged film damage: procedural vertical scratch lines and random \
         dust specks are composited over the image using blue-noise randomisation.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend intensity of the dust-and-scratches overlay. \
                 0.0 leaves the image completely unchanged (identity).",
        },
        SliderDef {
            name: "Scratch Density",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Relative frequency of vertical scratch lines. \
                 0.0 = no scratches, 1.0 = maximum scratch coverage.",
        },
        SliderDef {
            name: "Dust Amount",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Density of random dust-particle specks. \
                 0.0 = no dust, 1.0 = maximum dust.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "dust_and_scratches",
        wgsl_source: include_str!("dust_and_scratches.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[AuxTextureDef {
            name: "blue_noise_128",
            dimension: AuxTextureDimension::D2,
            filter: AuxSamplerFilter::Nearest,
        }],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            scratch_density: values[1],
            dust_amount: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    DustAndScratchesParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ── Registry / metadata ──────────────────────────────────────────────────

    #[test]
    fn test_dust_and_scratches_registry_entry_exists() {
        assert!(registry_by_id("dust_and_scratches").is_some());
    }

    #[test]
    fn test_dust_and_scratches_registry_metadata() {
        let reg = registry_by_id("dust_and_scratches").unwrap();
        assert_eq!(reg.meta.display_name, "Dust and Scratches");
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend intensity of the dust-and-scratches overlay. \
                         0.0 leaves the image completely unchanged (identity).",
                },
                SliderDef {
                    name: "Scratch Density",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Relative frequency of vertical scratch lines. \
                         0.0 = no scratches, 1.0 = maximum scratch coverage.",
                },
                SliderDef {
                    name: "Dust Amount",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Density of random dust-particle specks. \
                         0.0 = no dust, 1.0 = maximum dust.",
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
    fn test_dust_and_scratches_make_uniform_known_value() {
        let reg = registry_by_id("dust_and_scratches").unwrap();
        let bytes = (reg.make_uniform)(&[0.8, 0.6, 0.4]);
        let expected = bytemuck::bytes_of(&DustAndScratchesParams {
            strength: 0.8,
            scratch_density: 0.6,
            dust_amount: 0.4,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: output must equal the source pixel-for-pixel.
    #[test]
    fn test_dust_and_scratches_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 25000, 18000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "dust_and_scratches",
                values: vec![0.0, 0.5, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 25000).abs() <= 64,
                "R: expected ~25000 at strength=0, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 18000).abs() <= 64,
                "G: expected ~18000 at strength=0, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 40000).abs() <= 64,
                "B: expected ~40000 at strength=0, got {}",
                pixel[2]
            );
        }
    }

    /// Alpha must pass through unchanged regardless of strength.
    #[test]
    fn test_dust_and_scratches_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "dust_and_scratches",
                values: vec![1.0, 0.5, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    /// Full strength must darken at least some pixels on a white image,
    /// demonstrating that scratches/dust are actually applied.
    #[test]
    fn test_dust_and_scratches_full_strength_modifies_image() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Large enough image to guarantee some scratch/dust pixels fire.
        let img = make_solid_image(128, 128, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "dust_and_scratches",
                values: vec![1.0, 1.0, 1.0],
            }],
        );
        // Dust and scratches darken pixels, so expect some pixels below full white.
        let any_darkened = out.pixels().any(|p| p[0] < 60000);
        assert!(
            any_darkened,
            "full strength must darken at least some pixels"
        );
    }

    /// Higher scratch density must produce more darkened pixels than lower density
    /// on an otherwise identical white image.
    #[test]
    fn test_dust_and_scratches_higher_scratch_density_darkens_more_pixels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(128, 128, 65535, 65535, 65535);

        let out_low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "dust_and_scratches",
                values: vec![1.0, 0.05, 0.0],
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "dust_and_scratches",
                values: vec![1.0, 0.95, 0.0],
            }],
        );

        let dark_low = out_low.pixels().filter(|p| p[0] < 60000).count();
        let dark_high = out_high.pixels().filter(|p| p[0] < 60000).count();

        assert!(
            dark_high >= dark_low,
            "higher scratch_density must darken at least as many pixels as lower: \
             low={dark_low}, high={dark_high}"
        );
    }

    /// Higher dust amount must produce more darkened pixels than lower dust amount
    /// on an otherwise identical white image.
    #[test]
    fn test_dust_and_scratches_higher_dust_amount_darkens_more_pixels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(128, 128, 65535, 65535, 65535);

        let out_low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "dust_and_scratches",
                values: vec![1.0, 0.0, 0.05],
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "dust_and_scratches",
                values: vec![1.0, 0.0, 0.95],
            }],
        );

        let dark_low = out_low.pixels().filter(|p| p[0] < 60000).count();
        let dark_high = out_high.pixels().filter(|p| p[0] < 60000).count();

        assert!(
            dark_high >= dark_low,
            "higher dust_amount must darken at least as many pixels as lower: \
             low={dark_low}, high={dark_high}"
        );
    }

    /// Chaining with brightness at its identity value must not corrupt the output.
    #[test]
    fn test_dust_and_scratches_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "dust_and_scratches",
                    values: vec![0.5, 0.5, 0.5],
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
    fn test_dust_and_scratches_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "dust_and_scratches",
            values: vec![0.8, 0.6, 0.4],
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
