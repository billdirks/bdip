use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension, ParamKind, PassDef, PassInput,
    PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CatGhostParams {
    pub size: f32,
    pub strength: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for CatGhostParams {
    const ID: &'static str = "cat_ghost";
    const DISPLAY_NAME: &'static str = "Cat Ghost";
    const DESCRIPTION: &'static str =
        "Tiles a transparent cat image over the source as a centered repeating overlay.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Size",
            min: 50.0,
            max: 2000.0,
            default: 200.0,
            description: "Width of each cat tile in pixels; height scales proportionally \
                          to preserve the 1129×1498 aspect ratio.",
        },
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Opacity of the cat overlay; 0 leaves the image unchanged, \
                          1 composites at full opacity.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "cat_ghost",
        wgsl_source: include_str!("cat_ghost.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[AuxTextureDef {
            name: "twilight_cat",
            dimension: AuxTextureDimension::D2,
            filter: AuxSamplerFilter::Linear,
        }],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            size: values[0],
            strength: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<CatGhostParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_cat_ghost_registry_entry_exists() {
        assert!(registry_by_id("cat_ghost").is_some());
    }

    #[test]
    fn test_cat_ghost_registry_metadata() {
        let reg = registry_by_id("cat_ghost").unwrap();
        assert_eq!(reg.meta.display_name, "Cat Ghost");
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Size",
                    min: 50.0,
                    max: 2000.0,
                    default: 200.0,
                    description: "Width of each cat tile in pixels; height scales proportionally \
                                  to preserve the 1129×1498 aspect ratio.",
                },
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Opacity of the cat overlay; 0 leaves the image unchanged, \
                                  1 composites at full opacity.",
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
    fn test_cat_ghost_make_uniform_known_value() {
        let reg = registry_by_id("cat_ghost").unwrap();
        let bytes = (reg.make_uniform)(&[300.0, 0.75]);
        let expected = bytemuck::bytes_of(&CatGhostParams {
            size: 300.0,
            strength: 0.75,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_cat_ghost_zero_strength_is_identity() {
        // strength=0.0 is the identity value: output must match input within ±128.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cat_ghost",
                values: vec![200.0, 0.0],
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
    fn test_cat_ghost_alpha_preserved() {
        // Source alpha must pass through unchanged at any strength value.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cat_ghost",
                values: vec![200.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    #[test]
    fn test_cat_ghost_full_strength_changes_output() {
        // At strength=1.0 over a white image, at least some pixels must differ from white.
        // The cat has an alpha channel; transparent regions stay white, opaque regions change.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a large tile so we cover more of the cat image (including opaque areas).
        let img = make_solid_image(256, 256, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cat_ghost",
                values: vec![256.0, 1.0],
            }],
        );
        let any_changed = out
            .pixels()
            .any(|p| (p[0] as i32 - 65535).abs() > 128 || (p[1] as i32 - 65535).abs() > 128);
        assert!(
            any_changed,
            "full-strength cat ghost over white must change at least some pixels"
        );
    }

    #[test]
    fn test_cat_ghost_chains_with_brightness() {
        // Chaining must not panic, and source alpha must be preserved end-to-end.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "cat_ghost",
                    values: vec![200.0, 0.5],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved through the chain");
        }
    }
}
