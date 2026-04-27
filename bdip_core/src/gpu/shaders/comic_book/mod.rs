use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension, ParamKind, PassDef, PassInput,
    PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComicBookParams {
    pub strength: f32,
    pub dot_scale: f32,
    pub edge_threshold: f32,
    pub edge_thickness: f32,
}

impl TransformShader for ComicBookParams {
    const ID: &'static str = "comic_book";
    const DISPLAY_NAME: &'static str = "Comic Book";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
        },
        SliderDef {
            name: "Dot Scale",
            min: 4.0,
            max: 64.0,
            default: 16.0,
        },
        SliderDef {
            name: "Edge Threshold",
            min: 0.0,
            max: 1.0,
            default: 0.10,
        },
        SliderDef {
            name: "Edge Thickness",
            min: 0.01,
            max: 0.5,
            default: 0.15,
        },
    ]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "edges",
            wgsl_source: include_str!("edges.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("edges"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "halftone",
            wgsl_source: include_str!("halftone.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("halftone"),
            output_scale: PassScale::Full,
            aux_textures: &[AuxTextureDef {
                name: "halftone_dots",
                dimension: AuxTextureDimension::D2,
                filter: AuxSamplerFilter::Nearest,
            }],
        },
        PassDef {
            label: "combine",
            wgsl_source: include_str!("combine.wgsl"),
            inputs: &[
                PassInput::Source,
                PassInput::Scratch("halftone"),
                PassInput::Scratch("edges"),
            ],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            dot_scale: values[1],
            edge_threshold: values[2],
            edge_thickness: values[3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    ComicBookParams,
>());

#[cfg(test)]
mod tests {
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_comic_book_registry_entry_exists() {
        assert!(registry_by_id("comic_book").is_some());
    }

    #[test]
    fn test_comic_book_registry_metadata() {
        let reg = registry_by_id("comic_book").unwrap();
        assert_eq!(reg.meta.display_name, "Comic Book");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
                SliderDef {
                    name: "Dot Scale",
                    min: 4.0,
                    max: 64.0,
                    default: 16.0,
                },
                SliderDef {
                    name: "Edge Threshold",
                    min: 0.0,
                    max: 1.0,
                    default: 0.10,
                },
                SliderDef {
                    name: "Edge Thickness",
                    min: 0.01,
                    max: 0.5,
                    default: 0.15,
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            3,
            "Comic Book must have exactly 3 passes"
        );
    }

    #[test]
    fn test_comic_book_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let identity = roundtrip(&mut renderer, &engine, &img, &[]);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "comic_book",
                values: vec![0.0, 16.0, 0.10, 0.15],
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
    fn test_comic_book_full_strength_reduces_unique_values() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 32);
        let total = 32u32 * 32u32;
        for (i, pixel) in img.pixels_mut().enumerate() {
            let v = (1000 + (i as u32 * 63535 / total)) as u16;
            *pixel = image::Rgba([v, v, v, 65535]);
        }

        let unique_in: std::collections::HashSet<u16> = img.pixels().map(|p| p[0]).collect();

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "comic_book",
                values: vec![1.0, 16.0, 0.10, 0.15],
            }],
        );

        let unique_out: std::collections::HashSet<u16> = out.pixels().map(|p| p[0]).collect();
        assert!(
            unique_out.len() < unique_in.len(),
            "halftone must reduce unique color count: {} in, {} out",
            unique_in.len(),
            unique_out.len()
        );
    }

    #[test]
    fn test_comic_book_edges_darken_at_boundaries() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 65535 } else { 0 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "comic_book",
                values: vec![1.0, 16.0, 0.05, 0.15],
            }],
        );

        let in_bright = img.get_pixel(14, 8)[0] as i32;
        let out_bright = out.get_pixel(14, 8)[0] as i32;
        assert!(
            out_bright < in_bright,
            "edge pixel should be darkened by ink outlines: in={in_bright}, out={out_bright}"
        );
    }

    #[test]
    fn test_comic_book_halftone_pass_uses_aux_texture() {
        let reg = registry_by_id("comic_book").unwrap();
        let halftone_pass = reg.meta.passes.iter().find(|p| p.label == "halftone");
        assert!(
            halftone_pass.is_some(),
            "comic_book must have a 'halftone' pass"
        );
        let has_halftone_dots = halftone_pass
            .unwrap()
            .aux_textures
            .iter()
            .any(|a| a.name == "halftone_dots");
        assert!(
            has_halftone_dots,
            "halftone pass must declare 'halftone_dots' in its aux_textures"
        );
    }

    #[test]
    fn test_comic_book_dot_scale_changes_pattern() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(64, 64);
        let total = 64u32 * 64u32;
        for (i, pixel) in img.pixels_mut().enumerate() {
            let v = (1000 + (i as u32 * 63535 / total)) as u16;
            *pixel = image::Rgba([v, v, v, 65535]);
        }

        let out_small = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "comic_book",
                values: vec![1.0, 8.0, 0.10, 0.15],
            }],
        );
        let out_large = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "comic_book",
                values: vec![1.0, 32.0, 0.10, 0.15],
            }],
        );

        let any_different = out_small
            .pixels()
            .zip(out_large.pixels())
            .any(|(a, b)| a[0] != b[0]);
        assert!(
            any_different,
            "different dot_scale values must produce different halftone patterns"
        );
    }

    #[test]
    fn test_comic_book_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "comic_book",
                values: vec![1.0, 16.0, 0.10, 0.15],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be preserved");
        }
    }

    #[test]
    fn test_comic_book_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "comic_book",
            values: vec![1.0, 16.0, 0.10, 0.15],
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
