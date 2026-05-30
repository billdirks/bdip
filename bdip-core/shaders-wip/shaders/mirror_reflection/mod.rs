use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Mirror Reflection shader.
///
/// `mode` selects which axes are flipped using a bitmask encoded as a float:
///   - `0.0` — no mirror (identity)
///   - `1.0` — horizontal mirror (flip left/right)
///   - `2.0` — vertical mirror (flip top/bottom)
///   - `3.0` — both axes
///
/// `blend` controls how much of the mirrored image replaces the original.
/// At `0.0` the output is the unmodified source; at `1.0` it is fully mirrored.
/// Integer values of `mode` are the meaningful control points; the slider range
/// supports these four values.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MirrorReflectionParams {
    pub mode: f32,
    pub blend: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

impl TransformShader for MirrorReflectionParams {
    const ID: &'static str = "mirror_reflection";
    const DISPLAY_NAME: &'static str = "Mirror Reflection";
    const DESCRIPTION: &'static str =
        "Flips the image along horizontal, vertical, or both axes with an adjustable blend.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Mode",
            min: 0.0,
            max: 3.0,
            default: 0.0,
            description: "Mirror axis: 0 = none (identity), 1 = horizontal (left/right), \
                          2 = vertical (top/bottom), 3 = both axes.",
        },
        SliderDef {
            name: "Blend",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original (0.0) and the mirrored image (1.0).",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "mirror_reflection",
        wgsl_source: include_str!("mirror_reflection.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            mode: values[0],
            blend: values[1],
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    MirrorReflectionParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ── Registry / metadata tests ────────────────────────────────────────────

    #[test]
    fn test_mirror_reflection_registry_entry_exists() {
        assert!(registry_by_id("mirror_reflection").is_some());
    }

    #[test]
    fn test_mirror_reflection_registry_metadata() {
        let reg = registry_by_id("mirror_reflection").unwrap();
        assert_eq!(reg.meta.display_name, "Mirror Reflection");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Mode",
                    min: 0.0,
                    max: 3.0,
                    default: 0.0,
                    description: "Mirror axis: 0 = none (identity), 1 = horizontal (left/right), \
                                  2 = vertical (top/bottom), 3 = both axes.",
                },
                SliderDef {
                    name: "Blend",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original (0.0) and the mirrored image (1.0).",
                },
            ])
        );
    }

    #[test]
    fn test_mirror_reflection_make_uniform_known_value() {
        let reg = registry_by_id("mirror_reflection").unwrap();
        let bytes = (reg.make_uniform)(&[1.0, 0.75]);
        let expected = bytemuck::bytes_of(&MirrorReflectionParams {
            mode: 1.0,
            blend: 0.75,
            _pad0: 0.0,
            _pad1: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// Default parameters (mode=0, blend=0) must be an identity transformation.
    #[test]
    fn test_mirror_reflection_identity_at_defaults() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 10000, 30000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "mirror_reflection",
                values: vec![0.0, 0.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 10000).abs() <= 64,
                "R: expected ~10000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 30000).abs() <= 64,
                "G: expected ~30000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 50000).abs() <= 64,
                "B: expected ~50000, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// blend=0.0 with any mode is still a no-op because the blend factor is zero.
    #[test]
    fn test_mirror_reflection_zero_blend_is_identity_regardless_of_mode() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 40000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            // mode=3 (both axes) but blend=0.0 → should return original
            &[Transform {
                shader_id: "mirror_reflection",
                values: vec![3.0, 0.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 20000).abs() <= 64,
                "R: expected ~20000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 40000).abs() <= 64,
                "G: expected ~40000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 60000).abs() <= 64,
                "B: expected ~60000, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// On a uniform (solid-colour) image, any mirror + full blend must produce
    /// the same uniform colour, since all pixels are identical.
    #[test]
    fn test_mirror_reflection_solid_image_unchanged_under_any_mode() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        for mode in [1.0_f32, 2.0, 3.0] {
            let img = make_solid_image(4, 4, 32767, 16000, 8000);
            let out = roundtrip(
                &mut renderer,
                &engine,
                &img,
                &[Transform {
                    shader_id: "mirror_reflection",
                    values: vec![mode, 1.0],
                }],
            );

            for pixel in out.pixels() {
                assert!(
                    (pixel[0] as i32 - 32767).abs() <= 64,
                    "mode={mode}: R: expected ~32767, got {}",
                    pixel[0]
                );
                assert!(
                    (pixel[1] as i32 - 16000).abs() <= 64,
                    "mode={mode}: G: expected ~16000, got {}",
                    pixel[1]
                );
                assert!(
                    (pixel[2] as i32 - 8000).abs() <= 64,
                    "mode={mode}: B: expected ~8000, got {}",
                    pixel[2]
                );
                assert_eq!(pixel[3], 65535, "alpha must be preserved");
            }
        }
    }

    /// Alpha channel must be preserved for all mode/blend combinations.
    #[test]
    fn test_mirror_reflection_alpha_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 50000, 50000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "mirror_reflection",
                values: vec![3.0, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535");
        }
    }

    /// Chaining mirror_reflection with brightness must not panic and must preserve
    /// alpha.
    #[test]
    fn test_mirror_reflection_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "mirror_reflection",
                    values: vec![1.0, 1.0],
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
