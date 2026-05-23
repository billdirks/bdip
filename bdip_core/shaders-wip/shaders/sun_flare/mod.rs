use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Sun Flare effect.
///
/// Layout (8 × f32 = 32 bytes, aligned for WebGPU uniform buffers):
///   flare_x, flare_y   — normalised position of the light source in [0, 1]
///   intensity          — brightness multiplier for the flare
///   size               — scale factor for the entire flare complex
///   tint_r, tint_g, tint_b — linear-RGB color tint applied to the flare
///   _padding           — pad to 32 bytes (multiple of 16)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SunFlareParams {
    pub flare_x: f32,
    pub flare_y: f32,
    pub intensity: f32,
    pub size: f32,
    pub tint_r: f32,
    pub tint_g: f32,
    pub tint_b: f32,
    pub _padding: f32,
}

impl TransformShader for SunFlareParams {
    const ID: &'static str = "sun_flare";
    const DISPLAY_NAME: &'static str = "Sun Flare";
    const DESCRIPTION: &'static str = "Adds a procedural sun/lens-flare with radial streaks, a bright primary spot, \
         and secondary lens artifacts along the axis from the image centre to the light source.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Position X",
            min: 0.0,
            max: 1.0,
            default: 0.5, // Centre of image horizontally.
            description: "Horizontal position of the flare source in normalised [0, 1] coordinates.",
        },
        SliderDef {
            name: "Position Y",
            min: 0.0,
            max: 1.0,
            default: 0.5, // Centre of image vertically.
            description: "Vertical position of the flare source in normalised [0, 1] coordinates.",
        },
        SliderDef {
            name: "Intensity",
            min: 0.0,
            max: 1.0,
            default: 0.0, // Identity: no flare contribution added.
            description: "Overall brightness of the flare. 0.0 leaves the image unchanged.",
        },
        SliderDef {
            name: "Size",
            min: 0.1,
            max: 2.0,
            default: 1.0, // Neutral scale.
            description: "Scale factor for the entire flare complex, including streaks and artifacts.",
        },
        SliderDef {
            name: "Tint Red",
            min: 0.0,
            max: 1.0,
            default: 1.0, // Neutral white tint.
            description: "Red component of the color tint applied to the flare.",
        },
        SliderDef {
            name: "Tint Green",
            min: 0.0,
            max: 1.0,
            default: 0.9, // Slightly warm default (white-ish sun).
            description: "Green component of the color tint applied to the flare.",
        },
        SliderDef {
            name: "Tint Blue",
            min: 0.0,
            max: 1.0,
            default: 0.7, // Slightly warm/golden default.
            description: "Blue component of the color tint applied to the flare.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "sun_flare",
        wgsl_source: include_str!("sun_flare.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            flare_x: values[0],
            flare_y: values[1],
            intensity: values[2],
            size: values[3],
            tint_r: values[4],
            tint_g: values[5],
            tint_b: values[6],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<SunFlareParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ── Registry / metadata ──────────────────────────────────────────────────

    #[test]
    fn test_sun_flare_registry_entry_exists() {
        assert!(registry_by_id("sun_flare").is_some());
    }

    #[test]
    fn test_sun_flare_registry_metadata() {
        let reg = registry_by_id("sun_flare").unwrap();
        assert_eq!(reg.meta.display_name, "Sun Flare");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Position X",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Horizontal position of the flare source in normalised [0, 1] coordinates.",
                },
                SliderDef {
                    name: "Position Y",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Vertical position of the flare source in normalised [0, 1] coordinates.",
                },
                SliderDef {
                    name: "Intensity",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Overall brightness of the flare. 0.0 leaves the image unchanged.",
                },
                SliderDef {
                    name: "Size",
                    min: 0.1,
                    max: 2.0,
                    default: 1.0,
                    description: "Scale factor for the entire flare complex, including streaks and artifacts.",
                },
                SliderDef {
                    name: "Tint Red",
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                    description: "Red component of the color tint applied to the flare.",
                },
                SliderDef {
                    name: "Tint Green",
                    min: 0.0,
                    max: 1.0,
                    default: 0.9,
                    description: "Green component of the color tint applied to the flare.",
                },
                SliderDef {
                    name: "Tint Blue",
                    min: 0.0,
                    max: 1.0,
                    default: 0.7,
                    description: "Blue component of the color tint applied to the flare.",
                },
            ])
        );
    }

    #[test]
    fn test_sun_flare_passes_count() {
        let reg = registry_by_id("sun_flare").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_sun_flare_make_uniform_known_value() {
        let reg = registry_by_id("sun_flare").unwrap();
        let bytes = (reg.make_uniform)(&[0.3, 0.7, 0.8, 1.2, 1.0, 0.9, 0.7]);
        let expected = bytemuck::bytes_of(&SunFlareParams {
            flare_x: 0.3,
            flare_y: 0.7,
            intensity: 0.8,
            size: 1.2,
            tint_r: 1.0,
            tint_g: 0.9,
            tint_b: 0.7,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// intensity=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_sun_flare_identity_at_zero_intensity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sun_flare",
                values: vec![0.5, 0.5, 0.0, 1.0, 1.0, 0.9, 0.7],
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

    /// At full intensity the flare must brighten a dark image by adding light.
    #[test]
    fn test_sun_flare_full_intensity_brightens_dark_image() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Very dark input; flare centered in the image so the bright spot is visible.
        let img = make_solid_image(16, 16, 200, 200, 200);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sun_flare",
                values: vec![0.5, 0.5, 1.0, 1.0, 1.0, 0.9, 0.7],
            }],
        );

        let any_brighter = out
            .pixels()
            .any(|p| p[0] > 1000 || p[1] > 1000 || p[2] > 1000);
        assert!(
            any_brighter,
            "Full-intensity sun flare must brighten a dark image"
        );
    }

    /// Increasing intensity must produce a brighter result than lower intensity
    /// on a dark image, since the contribution is always non-negative.
    #[test]
    fn test_sun_flare_higher_intensity_is_brighter() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(16, 16, 500, 500, 500);

        let low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sun_flare",
                values: vec![0.5, 0.5, 0.3, 1.0, 1.0, 0.9, 0.7],
            }],
        );

        let high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sun_flare",
                values: vec![0.5, 0.5, 1.0, 1.0, 1.0, 0.9, 0.7],
            }],
        );

        let sum_r_low: u64 = low.pixels().map(|p| p[0] as u64).sum();
        let sum_r_high: u64 = high.pixels().map(|p| p[0] as u64).sum();
        assert!(
            sum_r_high > sum_r_low,
            "Higher intensity must produce brighter output: low={sum_r_low} high={sum_r_high}"
        );
    }

    /// The flare must produce spatial variation across pixels, not a flat overlay.
    #[test]
    fn test_sun_flare_produces_spatial_variation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(16, 16, 100, 100, 100);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sun_flare",
                values: vec![0.5, 0.5, 1.0, 1.0, 1.0, 0.9, 0.7],
            }],
        );

        let distinct: std::collections::HashSet<(u16, u16, u16)> =
            out.pixels().map(|p| (p[0], p[1], p[2])).collect();

        assert!(
            distinct.len() > 1,
            "Sun flare must produce spatial variation; got {} distinct values",
            distinct.len()
        );
    }

    /// Color tint must bias the flare toward the specified channel.
    /// A pure-red tint (1, 0, 0) must produce higher red output than blue on a dark image.
    #[test]
    fn test_sun_flare_red_tint_dominates_blue() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(16, 16, 100, 100, 100);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sun_flare",
                values: vec![0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0],
            }],
        );

        let (sum_r, sum_b): (u64, u64) = out
            .pixels()
            .fold((0u64, 0u64), |(r, b), p| (r + p[0] as u64, b + p[2] as u64));

        assert!(
            sum_r > sum_b,
            "Red tint must dominate blue channel: R={sum_r} B={sum_b}"
        );
    }

    /// Alpha channel must not be modified regardless of intensity.
    #[test]
    fn test_sun_flare_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sun_flare",
                values: vec![0.5, 0.5, 1.0, 1.0, 1.0, 0.9, 0.7],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the sun-flare result.
    #[test]
    fn test_sun_flare_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(8, 8, 15000, 10000, 25000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sun_flare",
                values: vec![0.5, 0.5, 0.5, 1.0, 1.0, 0.9, 0.7],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "sun_flare",
                    values: vec![0.5, 0.5, 0.5, 1.0, 1.0, 0.9, 0.7],
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
