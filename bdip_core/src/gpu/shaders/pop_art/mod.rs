use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PopArtParams {
    pub strength: f32,
    pub levels: f32,
    pub dot_scale: f32,
    pub _padding: f32,
}

impl TransformShader for PopArtParams {
    const ID: &'static str = "pop_art";
    const DISPLAY_NAME: &'static str = "Pop Art";
    const DESCRIPTION: &'static str =
        "Applies a bold, flat-color pop art style with quantized hues and a halftone dot overlay.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend factor between the original image and the pop art result.",
        },
        SliderDef {
            name: "Levels",
            min: 2.0,
            max: 8.0,
            default: 4.0,
            description: "Number of tonal levels in the quantization step.",
        },
        SliderDef {
            name: "Dot Scale",
            min: 4.0,
            max: 32.0,
            default: 12.0,
            description: "Halftone dot cell size in pixels; larger values produce more visible dots.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "quantize",
            wgsl_source: include_str!("quantize.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("quantize"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "colorize",
            wgsl_source: include_str!("colorize.wgsl"),
            inputs: &[PassInput::Scratch("quantize")],
            output: PassOutput::Scratch("colorize"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "combine",
            wgsl_source: include_str!("combine.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("colorize")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            levels: values[1],
            dot_scale: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<PopArtParams>());

#[cfg(test)]
mod tests {
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_pop_art_registry_entry_exists() {
        assert!(registry_by_id("pop_art").is_some());
    }

    #[test]
    fn test_pop_art_registry_metadata() {
        let reg = registry_by_id("pop_art").unwrap();
        assert_eq!(reg.meta.display_name, "Pop Art");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend factor between the original image and the pop art result.",
                },
                SliderDef {
                    name: "Levels",
                    min: 2.0,
                    max: 8.0,
                    default: 4.0,
                    description: "Number of tonal levels in the quantization step.",
                },
                SliderDef {
                    name: "Dot Scale",
                    min: 4.0,
                    max: 32.0,
                    default: 12.0,
                    description: "Halftone dot cell size in pixels; larger values produce more visible dots.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            3,
            "Pop Art must have exactly 3 passes"
        );
    }

    #[test]
    fn test_pop_art_make_uniform_known_value() {
        let reg = registry_by_id("pop_art").unwrap();
        let bytes = (reg.make_uniform)(&[0.5, 4.0, 12.0]);
        let expected = bytemuck::bytes_of(&super::PopArtParams {
            strength: 0.5,
            levels: 4.0,
            dot_scale: 12.0,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_pop_art_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let identity = roundtrip(&mut renderer, &engine, &img, &[]);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pop_art",
                values: vec![0.0, 4.0, 12.0],
            }],
        );
        for (p_ref, p_out) in identity.pixels().zip(out.pixels()) {
            assert_eq!(
                p_ref, p_out,
                "strength=0 must not alter pixels vs identity roundtrip"
            );
        }
    }

    #[test]
    fn test_pop_art_full_strength_reduces_unique_color_values() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 32);
        let total = 32u32 * 32u32;
        for (i, pixel) in img.pixels_mut().enumerate() {
            let v = (1000 + (i as u32 * 63535 / total)) as u16;
            *pixel = image::Rgba([v, v, v, 65535]);
        }

        let unique_in: std::collections::HashSet<[u16; 3]> =
            img.pixels().map(|p| [p[0], p[1], p[2]]).collect();

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pop_art",
                values: vec![1.0, 4.0, 12.0],
            }],
        );

        let unique_out: std::collections::HashSet<[u16; 3]> =
            out.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        assert!(
            unique_out.len() < unique_in.len(),
            "pop art with 4 levels must reduce unique color count: {} in, {} out",
            unique_in.len(),
            unique_out.len()
        );
    }

    #[test]
    fn test_pop_art_more_levels_produces_more_unique_values() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 32);
        let total = 32u32 * 32u32;
        for (i, pixel) in img.pixels_mut().enumerate() {
            let v = (1000 + (i as u32 * 63535 / total)) as u16;
            *pixel = image::Rgba([v, v, v, 65535]);
        }

        let out2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pop_art",
                values: vec![1.0, 2.0, 32.0],
            }],
        );
        let out8 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pop_art",
                values: vec![1.0, 8.0, 32.0],
            }],
        );

        let unique2: std::collections::HashSet<[u16; 3]> =
            out2.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        let unique8: std::collections::HashSet<[u16; 3]> =
            out8.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        assert!(
            unique8.len() > unique2.len(),
            "more levels must produce more distinct output colors: \
             2-level unique={}, 8-level unique={}",
            unique2.len(),
            unique8.len()
        );
    }

    #[test]
    fn test_pop_art_larger_dot_scale_changes_pattern() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(64, 64);
        let total = 64u32 * 64u32;
        for (i, pixel) in img.pixels_mut().enumerate() {
            let v = (16384 + (i as u32 * 32767 / total)) as u16;
            *pixel = image::Rgba([v, v, v, 65535]);
        }

        let out_small = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pop_art",
                values: vec![1.0, 4.0, 4.0],
            }],
        );
        let out_large = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pop_art",
                values: vec![1.0, 4.0, 24.0],
            }],
        );

        let any_different = out_small
            .pixels()
            .zip(out_large.pixels())
            .any(|(a, b)| a[0] != b[0] || a[1] != b[1] || a[2] != b[2]);
        assert!(
            any_different,
            "different dot_scale values must produce different halftone patterns"
        );
    }

    #[test]
    fn test_pop_art_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "pop_art",
                values: vec![1.0, 4.0, 12.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be preserved");
        }
    }

    #[test]
    fn test_pop_art_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "pop_art",
            values: vec![1.0, 4.0, 12.0],
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
