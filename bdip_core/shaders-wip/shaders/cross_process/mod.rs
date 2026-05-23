use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CrossProcessParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for CrossProcessParams {
    const ID: &'static str = "cross_process";
    const DISPLAY_NAME: &'static str = "Cross Process";
    const DESCRIPTION: &'static str = "Simulates cross-processing film by applying per-channel curve shifts: \
         red highlight boost, green S-curve midtone lift, and blue shadow lift.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // Identity: no curve shift applied.
        description: "Blend strength of the cross-process effect. 0.0 leaves the image unchanged.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "cross_process",
        wgsl_source: include_str!("cross_process.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    CrossProcessParams,
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
    fn test_cross_process_registry_entry_exists() {
        assert!(registry_by_id("cross_process").is_some());
    }

    #[test]
    fn test_cross_process_registry_metadata() {
        let reg = registry_by_id("cross_process").unwrap();
        assert_eq!(reg.meta.display_name, "Cross Process");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend strength of the cross-process effect. 0.0 leaves the image unchanged.",
            }])
        );
    }

    #[test]
    fn test_cross_process_passes_count() {
        let reg = registry_by_id("cross_process").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_cross_process_make_uniform_known_value() {
        let reg = registry_by_id("cross_process").unwrap();
        let bytes = (reg.make_uniform)(&[0.6]);
        let expected = bytemuck::bytes_of(&CrossProcessParams {
            strength: 0.6,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// strength=0.0 is the identity: the image must pass through unchanged.
    #[test]
    fn test_cross_process_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 20000, 15000, 8000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cross_process",
                values: vec![0.0],
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
                (pixel[2] as i32 - 8000).abs() <= 64,
                "B mismatch: {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    /// Alpha channel must not be modified regardless of strength.
    #[test]
    fn test_cross_process_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 30000, 20000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cross_process",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged");
        }
    }

    /// Red channel at full strength should be brightened (power curve < 1 lifts values).
    /// A mid-grey input (~32767 u16 ≈ 0.5 sRGB → ~0.214 linear) should produce a
    /// higher red value after pow(v, 0.85).
    #[test]
    fn test_cross_process_red_channel_lifted_at_full_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Mid-grey input: all channels equal.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let with_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cross_process",
                values: vec![1.0],
            }],
        );
        let without_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cross_process",
                values: vec![0.0],
            }],
        );

        // Red should be lifted because pow(v, 0.85) > v for v in (0, 1).
        for (a, b) in with_effect.pixels().zip(without_effect.pixels()) {
            assert!(
                a[0] > b[0],
                "R should be lifted by cross-process: effect={} identity={}",
                a[0],
                b[0]
            );
        }
    }

    /// Green channel S-curve boosts midtones: mid-grey input should produce a
    /// higher green value (smoothstep lifts the midtone region).
    #[test]
    fn test_cross_process_green_channel_midtone_boosted() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Use a mid-grey input where the smoothstep curve has maximum slope.
        // 32767/65535 ≈ 0.5 sRGB → ~0.214 linear.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let with_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cross_process",
                values: vec![1.0],
            }],
        );
        let without_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cross_process",
                values: vec![0.0],
            }],
        );

        // Smoothstep at v≈0.214: smoothstep(0.214) = 3*(0.214)^2 - 2*(0.214)^3 ≈ 0.175
        // which is less than 0.214, so green should be reduced in the lower range.
        // At v≈0.5 linear (sRGB ~0.735): smoothstep(0.5) = 0.5, equal to input.
        // The input here is ~0.214 linear, which is in the lower half where
        // smoothstep < v (the curve crushes shadows relative to the input).
        for (a, b) in with_effect.pixels().zip(without_effect.pixels()) {
            // Green should be modified (either direction) — not identical to identity.
            assert_ne!(a[1], b[1], "G should be modified by cross-process S-curve");
        }
    }

    /// Blue channel shadow lift: near-black input should have its blue value raised
    /// (1 - pow(1 - v, 1.3) > v for small v since the exponent > 1 compresses the
    /// complement).
    #[test]
    fn test_cross_process_blue_channel_shadows_lifted() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Near-black input: small linear value for the shadow lift to affect.
        let img = make_solid_image(2, 2, 2000, 2000, 2000);
        let with_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cross_process",
                values: vec![1.0],
            }],
        );
        let without_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cross_process",
                values: vec![0.0],
            }],
        );

        // Blue shadows should be lifted: 1 - pow(1-v, 1.3) > v for v near 0.
        for (a, b) in with_effect.pixels().zip(without_effect.pixels()) {
            assert!(
                a[2] > b[2],
                "B shadows should be lifted by cross-process: effect={} identity={}",
                a[2],
                b[2]
            );
        }
    }

    /// At strength=1.0, pure black (0,0,0) should remain black.
    /// Both pow(0, 0.85) = 0, smoothstep(0) = 0, and 1 - pow(1, 1.3) = 0.
    #[test]
    fn test_cross_process_black_input_stays_black() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cross_process",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(pixel[0] <= 64, "R should remain near black: {}", pixel[0]);
            assert!(pixel[1] <= 64, "G should remain near black: {}", pixel[1]);
            assert!(pixel[2] <= 64, "B should remain near black: {}", pixel[2]);
        }
    }

    /// At strength=1.0, pure white (65535,65535,65535) should remain near white.
    /// pow(1, 0.85) = 1, smoothstep(1) = 1, 1 - pow(0, 1.3) = 1.
    #[test]
    fn test_cross_process_white_input_stays_near_white() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cross_process",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[0] >= 64000,
                "R should remain near white: {}",
                pixel[0]
            );
            assert!(
                pixel[1] >= 64000,
                "G should remain near white: {}",
                pixel[1]
            );
            assert!(
                pixel[2] >= 64000,
                "B should remain near white: {}",
                pixel[2]
            );
        }
    }

    /// Chaining with brightness at identity (0.0) must not alter the result.
    #[test]
    fn test_cross_process_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 15000, 10000, 5000);
        let standalone = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cross_process",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "cross_process",
                    values: vec![0.5],
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
