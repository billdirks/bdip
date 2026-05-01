use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Duo-tone shader.
///
/// Six component values define two colors (shadow and highlight) in linear-light
/// RGB. The shader maps each pixel's Rec.709 luminance through a lerp between
/// these two colors. Defaults of shadow=(0,0,0) and highlight=(1,1,1) produce a
/// grayscale conversion, which is the only mathematically well-defined identity
/// for a two-color lerp: dark tones map to black, bright tones map to white.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DuoToneParams {
    pub shadow_r: f32,
    pub shadow_g: f32,
    pub shadow_b: f32,
    pub _padding0: f32,
    pub highlight_r: f32,
    pub highlight_g: f32,
    pub highlight_b: f32,
    pub _padding1: f32,
}

impl TransformShader for DuoToneParams {
    const ID: &'static str = "duo_tone";
    const DISPLAY_NAME: &'static str = "Duo-tone";
    const DESCRIPTION: &'static str = "Maps image luminance to two colors: dark tones take the shadow color, \
         bright tones take the highlight color, with smooth interpolation between them.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Shadow R",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Red component of the color applied to shadow (dark) tones.",
        },
        SliderDef {
            name: "Shadow G",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Green component of the color applied to shadow (dark) tones.",
        },
        SliderDef {
            name: "Shadow B",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blue component of the color applied to shadow (dark) tones.",
        },
        SliderDef {
            name: "Highlight R",
            min: 0.0,
            max: 1.0,
            default: 1.0,
            description: "Red component of the color applied to highlight (bright) tones.",
        },
        SliderDef {
            name: "Highlight G",
            min: 0.0,
            max: 1.0,
            default: 1.0,
            description: "Green component of the color applied to highlight (bright) tones.",
        },
        SliderDef {
            name: "Highlight B",
            min: 0.0,
            max: 1.0,
            default: 1.0,
            description: "Blue component of the color applied to highlight (bright) tones.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "duo_tone",
        wgsl_source: include_str!("duo_tone.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            shadow_r: values[0],
            shadow_g: values[1],
            shadow_b: values[2],
            _padding0: 0.0,
            highlight_r: values[3],
            highlight_g: values[4],
            highlight_b: values[5],
            _padding1: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<DuoToneParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ── Registry / metadata ──────────────────────────────────────────────────

    #[test]
    fn test_duo_tone_registry_entry_exists() {
        assert!(registry_by_id("duo_tone").is_some());
    }

    #[test]
    fn test_duo_tone_registry_metadata() {
        let reg = registry_by_id("duo_tone").unwrap();
        assert_eq!(reg.meta.display_name, "Duo-tone");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Shadow R",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Red component of the color applied to shadow (dark) tones.",
                },
                SliderDef {
                    name: "Shadow G",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Green component of the color applied to shadow (dark) tones.",
                },
                SliderDef {
                    name: "Shadow B",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blue component of the color applied to shadow (dark) tones.",
                },
                SliderDef {
                    name: "Highlight R",
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                    description: "Red component of the color applied to highlight (bright) tones.",
                },
                SliderDef {
                    name: "Highlight G",
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                    description: "Green component of the color applied to highlight (bright) tones.",
                },
                SliderDef {
                    name: "Highlight B",
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                    description: "Blue component of the color applied to highlight (bright) tones.",
                },
            ])
        );
    }

    #[test]
    fn test_duo_tone_passes_count() {
        let reg = registry_by_id("duo_tone").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_duo_tone_make_uniform_known_value() {
        let reg = registry_by_id("duo_tone").unwrap();
        let bytes = (reg.make_uniform)(&[0.1, 0.2, 0.3, 0.8, 0.9, 1.0]);
        let expected = bytemuck::bytes_of(&DuoToneParams {
            shadow_r: 0.1,
            shadow_g: 0.2,
            shadow_b: 0.3,
            _padding0: 0.0,
            highlight_r: 0.8,
            highlight_g: 0.9,
            highlight_b: 1.0,
            _padding1: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// Default parameters (shadow=black, highlight=white) produce a grayscale
    /// image. A neutral grey input must come out grey (all channels equal) and
    /// match the expected Rec.709 luminance of the source.
    #[test]
    fn test_duo_tone_default_grey_input_stays_grey() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Neutral grey: all channels equal, so luminance == any channel value.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "duo_tone",
                values: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }],
        );

        for pixel in out.pixels() {
            // Output must be achromatic (all channels equal).
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "R vs G: grey input must produce grey output: R={} G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[0] as i32 - pixel[2] as i32).abs() <= 64,
                "R vs B: grey input must produce grey output: R={} B={}",
                pixel[0],
                pixel[2]
            );
        }
    }

    /// A pure-black input must map exactly to the shadow color.
    #[test]
    fn test_duo_tone_black_input_maps_to_shadow_color() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Black: luminance = 0, so output = shadow color.
        // Shadow = (0.0, 0.0, 1.0) linear → pure blue.
        let img = make_solid_image(2, 2, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "duo_tone",
                // shadow=(0,0,1) blue, highlight=(1,0,0) red
                values: vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            }],
        );

        // linear 0.0 → sRGB 0 → u16 0; linear 1.0 → sRGB 1.0 → u16 65535.
        for pixel in out.pixels() {
            assert!(
                pixel[0] <= 64,
                "R must be near 0 (shadow R=0): {}",
                pixel[0]
            );
            assert!(
                pixel[1] <= 64,
                "G must be near 0 (shadow G=0): {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 65535).abs() <= 64,
                "B must be near 65535 (shadow B=1): {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    /// A pure-white input must map exactly to the highlight color.
    #[test]
    fn test_duo_tone_white_input_maps_to_highlight_color() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // White: luminance = 1, so output = highlight color.
        // Highlight = (1.0, 0.5, 0.0) linear → orange-ish.
        let img = make_solid_image(2, 2, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "duo_tone",
                // shadow=(0,0,1), highlight=(1,0.5,0)
                values: vec![0.0, 0.0, 1.0, 1.0, 0.5, 0.0],
            }],
        );

        // highlight_r=1.0 → u16 ≈ 65535; highlight_g=0.5 linear → sRGB ≈ 0.735 → u16 ≈ 48185;
        // highlight_b=0.0 → u16 = 0.
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 64,
                "R must be near 65535 (highlight R=1.0): {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 48185).abs() <= 256,
                "G must be near 48185 (highlight G=0.5 linear): {}",
                pixel[1]
            );
            assert!(
                pixel[2] <= 64,
                "B must be near 0 (highlight B=0.0): {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    /// Alpha channel must not be modified regardless of the color parameters.
    #[test]
    fn test_duo_tone_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 30000, 20000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "duo_tone",
                values: vec![0.0, 0.2, 0.8, 1.0, 0.5, 0.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by duo_tone");
        }
    }

    /// Chaining duo_tone with brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_duo_tone_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 30000, 40000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "duo_tone",
                values: vec![0.0, 0.1, 0.5, 0.8, 0.4, 0.0],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "duo_tone",
                    values: vec![0.0, 0.1, 0.5, 0.8, 0.4, 0.0],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );

        for (a, b) in standalone.pixels().zip(chained.pixels()) {
            assert!((a[0] as i32 - b[0] as i32).abs() <= 64, "R chain mismatch");
            assert!((a[1] as i32 - b[1] as i32).abs() <= 64, "G chain mismatch");
            assert!((a[2] as i32 - b[2] as i32).abs() <= 64, "B chain mismatch");
        }
    }

    /// When shadow and highlight are set to the same color, all pixels must
    /// output that color regardless of luminance.
    #[test]
    fn test_duo_tone_same_shadow_and_highlight_produces_flat_color() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 10000, 30000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "duo_tone",
                // shadow == highlight == (0.5, 0.25, 0.75) linear
                values: vec![0.5, 0.25, 0.75, 0.5, 0.25, 0.75],
            }],
        );

        // All pixels must converge to the same output color.
        let first = out.get_pixel(0, 0);
        for pixel in out.pixels() {
            assert_eq!(
                pixel[0], first[0],
                "R must be uniform: {} vs {}",
                pixel[0], first[0]
            );
            assert_eq!(
                pixel[1], first[1],
                "G must be uniform: {} vs {}",
                pixel[1], first[1]
            );
            assert_eq!(
                pixel[2], first[2],
                "B must be uniform: {} vs {}",
                pixel[2], first[2]
            );
        }
    }
}
