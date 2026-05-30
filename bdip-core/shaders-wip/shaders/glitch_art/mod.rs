use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Uniform layout shared by both passes.
///
/// - `strength`: controls both the bit-crush depth and the maximum slice-displacement
///   magnitude. At 0.0 both passes are identity operations.
/// - `seed`: offsets the pseudo-random hash seed so users can select different glitch
///   patterns without changing the overall intensity.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlitchArtParams {
    pub strength: f32,
    pub seed: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

impl TransformShader for GlitchArtParams {
    const ID: &'static str = "glitch_art";
    const DISPLAY_NAME: &'static str = "Glitch Art";
    const DESCRIPTION: &'static str = "Simulates digital signal corruption via bit-crushing and random horizontal \
         scanline displacement.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Overall glitch intensity. 0 = no effect; 1 = maximum bit-crush \
                          and scanline displacement.",
        },
        SliderDef {
            name: "Seed",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Pseudo-random seed for the scanline displacement pattern. \
                          Different values produce different arrangements of displaced rows.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "bit_crush",
            wgsl_source: include_str!("glitch_art_bit_crush.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("crushed"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "glitch",
            wgsl_source: include_str!("glitch_art_glitch.wgsl"),
            inputs: &[PassInput::Scratch("crushed")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            seed: values[1],
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    GlitchArtParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_glitch_art_registry_entry_exists() {
        assert!(registry_by_id("glitch_art").is_some());
    }

    #[test]
    fn test_glitch_art_registry_metadata() {
        let reg = registry_by_id("glitch_art").unwrap();
        assert_eq!(reg.meta.display_name, "Glitch Art");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Overall glitch intensity. 0 = no effect; 1 = maximum \
                                  bit-crush and scanline displacement.",
                },
                SliderDef {
                    name: "Seed",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Pseudo-random seed for the scanline displacement pattern. \
                                  Different values produce different arrangements of displaced \
                                  rows.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "glitch_art must have exactly 2 passes"
        );
    }

    #[test]
    fn test_glitch_art_make_uniform_known_value() {
        let reg = registry_by_id("glitch_art").unwrap();
        let bytes = (reg.make_uniform)(&[0.5, 0.3]);
        let expected = bytemuck::bytes_of(&GlitchArtParams {
            strength: 0.5,
            seed: 0.3,
            _pad0: 0.0,
            _pad1: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_glitch_art_zero_strength_is_identity() {
        // At strength=0 the bit-crush pass uses 256 levels (8-bit, no visible change)
        // and the glitch pass applies zero displacement. A solid-color image must be
        // returned unchanged within rounding tolerance.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "glitch_art",
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
    fn test_glitch_art_alpha_preserved_at_zero_strength() {
        // The alpha channel must pass through both passes unchanged.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "glitch_art",
                values: vec![0.0, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    #[test]
    fn test_glitch_art_alpha_preserved_at_max_strength() {
        // Alpha must be unchanged even when both effects are at full strength.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "glitch_art",
                values: vec![1.0, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved at max strength");
        }
    }

    #[test]
    fn test_glitch_art_bit_crush_quantises_at_max_strength() {
        // At strength=1 the bit-crush exponent is 2 (4 levels). With step=0.25,
        // the only valid linear-light output values are {0.0, 0.25, 0.5, 0.75, 1.0}.
        //
        // The pipeline stores values in linear light internally and applies sRGB
        // gamma-encoding at presentation time, so the u16 values read back from
        // the CPU correspond to sRGB-encoded versions of those linear levels:
        //   linear 0.0  → sRGB 0.0           → u16 =     0
        //   linear 0.25 → sRGB ≈ 0.5372       → u16 ≈ 35229
        //   linear 0.5  → sRGB ≈ 0.7353       → u16 ≈ 48183
        //   linear 0.75 → sRGB ≈ 0.8806       → u16 ≈ 57729
        //   linear 1.0  → sRGB 1.0            → u16 = 65535
        //
        // A tolerance of ±300 u16 accounts for f16 rounding across both passes
        // and the u32→u16 cast in the presentation shader.
        //
        // Because the glitch pass on a solid-colour image cannot change the colour
        // (every displaced row samples the same colour), the output colour must be
        // one of these quantised levels.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "glitch_art",
                values: vec![1.0, 0.0],
            }],
        );
        // sRGB-encoded u16 for linear levels {0.0, 0.25, 0.5, 0.75, 1.0}.
        let valid_levels: [i32; 5] = [0, 35229, 48183, 57729, 65535];
        for pixel in out.pixels() {
            let r = pixel[0] as i32;
            let nearest_dist = valid_levels
                .iter()
                .map(|&l| (r - l).unsigned_abs())
                .min()
                .unwrap();
            assert!(
                nearest_dist <= 300,
                "R channel {r} is not near any 4-level quantisation point (in sRGB u16 space); \
                 nearest distance = {nearest_dist}"
            );
        }
    }

    #[test]
    fn test_glitch_art_nonzero_strength_perturbs_solid_image() {
        // A striped (not solid) image: alternating columns of different colours.
        // At high strength the glitch pass displaces rows, so some pixels will
        // sample from a different column than their own, producing colour changes.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v: u16 = if x % 2 == 0 { 10000 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "glitch_art",
                values: vec![1.0, 0.5],
            }],
        );

        // At least some output pixel must differ noticeably from both source values.
        // After bit-crushing to 4 levels and slice displacement, the output will
        // contain values that are quantised versions of both source values.
        // We just check that the output is not identical to a simple pixel-wise
        // copy of the input (i.e. some displacement occurred on at least one row).
        let any_mismatch = out.enumerate_pixels().any(|(x, _y, p)| {
            let expected: u16 = if x % 2 == 0 { 10000 } else { 55000 };
            // Allow ±3000 for bit-crush quantisation, but flag large deviations
            // caused by cross-column displacement.
            (p[0] as i32 - expected as i32).unsigned_abs() > 3000
        });
        assert!(
            any_mismatch,
            "at strength=1 some pixels must be displaced from their source column"
        );
    }

    #[test]
    fn test_glitch_art_different_seeds_produce_different_patterns() {
        // Two runs with the same strength but different seed values must produce
        // at least one pixel that differs. Uses a striped image so displacement
        // is detectable as a color change.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v: u16 = if x % 2 == 0 { 10000 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_a = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "glitch_art",
                values: vec![1.0, 0.1],
            }],
        );
        let out_b = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "glitch_art",
                values: vec![1.0, 0.9],
            }],
        );

        let any_different = out_a
            .pixels()
            .zip(out_b.pixels())
            .any(|(a, b)| (a[0] as i32 - b[0] as i32).unsigned_abs() > 64);
        assert!(
            any_different,
            "different seed values must produce different displacement patterns"
        );
    }

    #[test]
    fn test_glitch_art_deterministic() {
        // Running with identical params twice must produce bit-identical output.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "glitch_art",
            values: vec![0.8, 0.5],
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

    #[test]
    fn test_glitch_art_chains_with_brightness() {
        // Chaining glitch_art with brightness must not panic and must preserve alpha.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "glitch_art",
                    values: vec![0.5, 0.0],
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
}
