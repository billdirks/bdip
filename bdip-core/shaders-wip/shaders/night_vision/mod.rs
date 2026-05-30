use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Night Vision shader.
///
/// Five floats pack into 16 bytes (one padding float for WebGPU uniform alignment):
/// - `green_tint`:      Intensity of the phosphor green colorisation.
/// - `noise_amount`:    Amplitude of the high-frequency noise typical of NV equipment.
/// - `scanline_intensity`: Contrast depth of the CRT-style horizontal scanlines.
/// - `amplification`:   Brightness multiplier simulating light amplification.
///   1.0 leaves luminance unchanged; higher values lift dark areas.
///
/// # Identity design
///
/// All parameters default to values that produce a no-op:
/// - `green_tint` = 0.0:  no colour shift applied.
/// - `noise_amount` = 0.0:  no noise added.
/// - `scanline_intensity` = 0.0:  no scanlines drawn.
/// - `amplification` = 1.0:  luminance multiplied by 1 (unchanged).
///
/// Because all four active fields fit in 16 bytes exactly, no padding float is needed.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NightVisionParams {
    /// Strength of the green phosphor tint. 0.0 = no tint; 1.0 = full green.
    pub green_tint: f32,
    /// Amplitude of high-frequency noise. 0.0 = no noise; 1.0 = heavy noise.
    pub noise_amount: f32,
    /// Depth of the horizontal CRT scanlines. 0.0 = no lines; 1.0 = maximum contrast.
    pub scanline_intensity: f32,
    /// Light-amplification multiplier. 1.0 = no amplification; values above 1.0 lift darks.
    pub amplification: f32,
}

impl TransformShader for NightVisionParams {
    const ID: &'static str = "night_vision";
    const DISPLAY_NAME: &'static str = "Night Vision";
    const DESCRIPTION: &'static str = "Simulates night-vision goggles: green phosphor tint, light amplification, \
         CRT scanlines, and high-frequency sensor noise.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Green Tint",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Intensity of the characteristic phosphor green colorisation. \
                 0.0 = no tint (identity); 1.0 = fully green.",
        },
        SliderDef {
            name: "Noise Amount",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Amplitude of the high-frequency sensor noise typical of night-vision \
                 equipment. 0.0 = no noise (identity); 1.0 = heavy grain.",
        },
        SliderDef {
            name: "Scanline Intensity",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Contrast depth of the horizontal CRT-style scanlines. \
                 0.0 = no scanlines (identity); 1.0 = maximum line contrast.",
        },
        SliderDef {
            name: "Amplification",
            min: 1.0,
            max: 8.0,
            default: 1.0,
            description: "Light-amplification multiplier. \
                 1.0 = no amplification (identity); higher values lift dark areas.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "night_vision",
        wgsl_source: include_str!("night_vision.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            green_tint: values[0],
            noise_amount: values[1],
            scanline_intensity: values[2],
            amplification: values[3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    NightVisionParams,
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
    fn test_night_vision_registry_entry_exists() {
        assert!(registry_by_id("night_vision").is_some());
    }

    #[test]
    fn test_night_vision_registry_metadata() {
        let reg = registry_by_id("night_vision").unwrap();
        assert_eq!(reg.meta.display_name, "Night Vision");
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Green Tint",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Intensity of the characteristic phosphor green colorisation. \
                         0.0 = no tint (identity); 1.0 = fully green.",
                },
                SliderDef {
                    name: "Noise Amount",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Amplitude of the high-frequency sensor noise typical of night-vision \
                         equipment. 0.0 = no noise (identity); 1.0 = heavy grain.",
                },
                SliderDef {
                    name: "Scanline Intensity",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Contrast depth of the horizontal CRT-style scanlines. \
                         0.0 = no scanlines (identity); 1.0 = maximum line contrast.",
                },
                SliderDef {
                    name: "Amplification",
                    min: 1.0,
                    max: 8.0,
                    default: 1.0,
                    description: "Light-amplification multiplier. \
                         1.0 = no amplification (identity); higher values lift dark areas.",
                },
            ])
        );
    }

    #[test]
    fn test_night_vision_make_uniform_known_value() {
        let reg = registry_by_id("night_vision").unwrap();
        let bytes = (reg.make_uniform)(&[0.8, 0.5, 0.3, 2.0]);
        let expected = bytemuck::bytes_of(&NightVisionParams {
            green_tint: 0.8,
            noise_amount: 0.5,
            scanline_intensity: 0.3,
            amplification: 2.0,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// All parameters at identity values must leave the image pixel-for-pixel unchanged.
    #[test]
    fn test_night_vision_default_params_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 20000, 35000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "night_vision",
                values: vec![0.0, 0.0, 0.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 20000).abs() <= 64,
                "R: expected ~20000 at identity, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 35000).abs() <= 64,
                "G: expected ~35000 at identity, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 50000).abs() <= 64,
                "B: expected ~50000 at identity, got {}",
                pixel[2]
            );
        }
    }

    /// Alpha channel must pass through unchanged regardless of active parameters.
    #[test]
    fn test_night_vision_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "night_vision",
                values: vec![1.0, 1.0, 1.0, 4.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    /// Green tint must shift the green channel higher than red/blue on a grey image.
    #[test]
    fn test_night_vision_green_tint_raises_green_channel() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Neutral grey so any channel asymmetry is due to the tint alone.
        let img = make_solid_image(4, 4, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "night_vision",
                values: vec![1.0, 0.0, 0.0, 1.0],
            }],
        );
        // Full green tint: green must be strictly higher than red or blue.
        for pixel in out.pixels() {
            assert!(
                pixel[1] > pixel[0],
                "green must exceed red with full tint: G={}, R={}",
                pixel[1],
                pixel[0]
            );
            assert!(
                pixel[1] > pixel[2],
                "green must exceed blue with full tint: G={}, B={}",
                pixel[1],
                pixel[2]
            );
        }
    }

    /// Amplification above 1.0 must raise the luminance of a dark image.
    #[test]
    fn test_night_vision_amplification_brightens_dark_image() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Very dark image — sRGB u16 ~3277 (≈5% white).
        let img = make_solid_image(4, 4, 3277, 3277, 3277);
        let out_no_amp = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "night_vision",
                values: vec![0.0, 0.0, 0.0, 1.0],
            }],
        );
        let out_amp = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "night_vision",
                values: vec![0.0, 0.0, 0.0, 4.0],
            }],
        );
        // Amplified output must be visibly brighter.
        for (p_no_amp, p_amp) in out_no_amp.pixels().zip(out_amp.pixels()) {
            assert!(
                p_amp[0] > p_no_amp[0],
                "amplification must brighten R: baseline={}, amplified={}",
                p_no_amp[0],
                p_amp[0]
            );
        }
    }

    /// Scanlines must introduce variation across rows on an otherwise uniform image.
    #[test]
    fn test_night_vision_scanlines_create_row_variation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a taller image to capture multiple scanline periods.
        let img = make_solid_image(4, 32, 40000, 40000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "night_vision",
                values: vec![0.0, 0.0, 1.0, 1.0],
            }],
        );
        // With scanlines active there must be at least two distinct green-channel
        // values across the rows.
        let row_values: Vec<u16> = (0..32).map(|y| out.get_pixel(0, y)[1]).collect();
        let first = row_values[0];
        let has_variation = row_values.iter().any(|&v| v != first);
        assert!(
            has_variation,
            "scanlines must produce row-level luminance variation"
        );
    }

    /// Noise must introduce per-pixel variation on a flat colour image.
    #[test]
    fn test_night_vision_noise_creates_pixel_variation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Large enough to reliably contain multiple different hash results.
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "night_vision",
                values: vec![0.0, 1.0, 0.0, 1.0],
            }],
        );
        let first = out.get_pixel(0, 0)[1];
        let has_variation = out.pixels().any(|p| p[1] != first);
        assert!(
            has_variation,
            "noise must produce per-pixel luminance variation"
        );
    }

    /// Chaining with brightness at its identity value must not corrupt the output.
    #[test]
    fn test_night_vision_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "night_vision",
                    values: vec![0.5, 0.3, 0.2, 2.0],
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
    fn test_night_vision_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "night_vision",
            values: vec![0.8, 0.6, 0.4, 3.0],
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
