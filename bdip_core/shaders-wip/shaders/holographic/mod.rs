use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Holographic foil effect.
///
/// Layout (4 × f32 = 16 bytes, aligned for WebGPU uniform buffers):
///   intensity      — overall blend strength of the holographic overlay [0, 1]
///   frequency      — rainbow colour-shift frequency (cycles across the image) [0.5, 20]
///   scale          — spatial scale of the iridescent pattern [0.5, 4]
///   blend_strength — screen-blend weight applied before the additive mix [0, 1]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HolographicParams {
    pub intensity: f32,
    pub frequency: f32,
    pub scale: f32,
    pub blend_strength: f32,
}

impl TransformShader for HolographicParams {
    const ID: &'static str = "holographic";
    const DISPLAY_NAME: &'static str = "Holographic";
    const DESCRIPTION: &'static str = "Overlays iridescent rainbow spectral colours onto the image, simulating holographic \
         sticker or foil, generated entirely from UV coordinates and sine waves.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Intensity",
            min: 0.0,
            max: 1.0,
            default: 0.0, // Identity: no holographic overlay applied.
            description: "Overall blend strength of the holographic foil overlay. \
                          0.0 leaves the image unchanged.",
        },
        SliderDef {
            name: "Frequency",
            min: 0.5,
            max: 20.0,
            default: 6.0, // Neutral mid-range rainbow cycling speed.
            description: "Number of full rainbow spectrum cycles across the image width. \
                          Higher values produce finer, more densely packed colour bands.",
        },
        SliderDef {
            name: "Rainbow Scale",
            min: 0.5,
            max: 4.0,
            default: 1.0, // Neutral scale: pattern fills the frame naturally.
            description: "Spatial scale of the iridescent pattern. Values above 1 zoom in \
                          on the pattern; values below 1 zoom out.",
        },
        SliderDef {
            name: "Blend Strength",
            min: 0.0,
            max: 1.0,
            default: 0.5, // Balanced screen-blend contribution.
            description: "Weight of the screen-blend layer relative to the additive layer. \
                          Higher values produce a brighter, more luminous foil look.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "holographic",
        wgsl_source: include_str!("holographic.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            intensity: values[0],
            frequency: values[1],
            scale: values[2],
            blend_strength: values[3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    HolographicParams,
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
    fn test_holographic_registry_entry_exists() {
        assert!(registry_by_id("holographic").is_some());
    }

    #[test]
    fn test_holographic_registry_metadata() {
        let reg = registry_by_id("holographic").unwrap();
        assert_eq!(reg.meta.display_name, "Holographic");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Intensity",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Overall blend strength of the holographic foil overlay. \
                                  0.0 leaves the image unchanged.",
                },
                SliderDef {
                    name: "Frequency",
                    min: 0.5,
                    max: 20.0,
                    default: 6.0,
                    description: "Number of full rainbow spectrum cycles across the image width. \
                                  Higher values produce finer, more densely packed colour bands.",
                },
                SliderDef {
                    name: "Rainbow Scale",
                    min: 0.5,
                    max: 4.0,
                    default: 1.0,
                    description: "Spatial scale of the iridescent pattern. Values above 1 zoom in \
                                  on the pattern; values below 1 zoom out.",
                },
                SliderDef {
                    name: "Blend Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Weight of the screen-blend layer relative to the additive layer. \
                                  Higher values produce a brighter, more luminous foil look.",
                },
            ])
        );
    }

    #[test]
    fn test_holographic_passes_count() {
        let reg = registry_by_id("holographic").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_holographic_make_uniform_known_value() {
        let reg = registry_by_id("holographic").unwrap();
        let bytes = (reg.make_uniform)(&[0.8, 10.0, 2.0, 0.6]);
        let expected = bytemuck::bytes_of(&HolographicParams {
            intensity: 0.8,
            frequency: 10.0,
            scale: 2.0,
            blend_strength: 0.6,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// intensity=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_holographic_identity_at_zero_intensity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "holographic",
                values: vec![0.0, 6.0, 1.0, 0.5],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 20000).abs() <= 64,
                "R mismatch: {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 15000).abs() <= 64,
                "G mismatch: {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 30000).abs() <= 64,
                "B mismatch: {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    /// At full intensity the overlay must alter a dark image by adding spectral colour.
    #[test]
    fn test_holographic_full_intensity_alters_dark_image() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Dark input; use an 8×8 image so UV variation produces colour variation
        // across multiple pixels at different positions.
        let img = make_solid_image(8, 8, 200, 200, 200);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "holographic",
                values: vec![1.0, 6.0, 1.0, 0.5],
            }],
        );

        let any_changed = out.pixels().any(|p| p[0] > 500 || p[1] > 500 || p[2] > 500);
        assert!(
            any_changed,
            "Full-intensity holographic overlay must alter a dark image"
        );
    }

    /// Higher intensity must produce a stronger result than lower intensity on a dark image.
    #[test]
    fn test_holographic_higher_intensity_produces_stronger_effect() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(8, 8, 500, 500, 500);

        let low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "holographic",
                values: vec![0.2, 6.0, 1.0, 0.5],
            }],
        );
        let high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "holographic",
                values: vec![1.0, 6.0, 1.0, 0.5],
            }],
        );

        let sum_low: u64 = low
            .pixels()
            .map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64)
            .sum();
        let sum_high: u64 = high
            .pixels()
            .map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64)
            .sum();

        assert!(
            sum_high > sum_low,
            "Higher intensity must produce a brighter output: low={sum_low} high={sum_high}"
        );
    }

    /// The overlay must produce colour variation across pixels, not a uniform tint.
    #[test]
    fn test_holographic_produces_color_variation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Neutral grey input; use a 16×16 image for UV diversity.
        let img = make_solid_image(16, 16, 100, 100, 100);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "holographic",
                values: vec![1.0, 6.0, 1.0, 0.5],
            }],
        );

        let distinct: std::collections::HashSet<(u16, u16, u16)> =
            out.pixels().map(|p| (p[0], p[1], p[2])).collect();

        assert!(
            distinct.len() > 1,
            "Holographic must produce colour variation; got {} distinct colours",
            distinct.len()
        );
    }

    /// Alpha channel must not be modified regardless of intensity.
    #[test]
    fn test_holographic_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "holographic",
                values: vec![1.0, 6.0, 1.0, 0.5],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the holographic result.
    #[test]
    fn test_holographic_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 15000, 10000, 25000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "holographic",
                values: vec![0.5, 6.0, 1.0, 0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "holographic",
                    values: vec![0.5, 6.0, 1.0, 0.5],
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
}
