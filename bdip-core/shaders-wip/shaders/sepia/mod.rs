use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, PassScale, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SepiaParams {
    pub _unused: [f32; 4],
}

impl TransformShader for SepiaParams {
    const ID: &'static str = "sepia";
    const DISPLAY_NAME: &'static str = "Sepia";
    const DESCRIPTION: &'static str =
        "Applies a sepia tone using the W3C standard color matrix in linear light.";
    const PARAM: ParamKind = ParamKind::Toggle;
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "sepia",
        wgsl_source: include_str!("sepia.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(_: &[f32]) -> Self {
        Self { _unused: [0.0; 4] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<SepiaParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_sepia_registry_entry_exists() {
        assert!(registry_by_id("sepia").is_some());
    }

    #[test]
    fn test_sepia_registry_metadata() {
        let reg = registry_by_id("sepia").unwrap();
        assert_eq!(reg.meta.display_name, "Sepia");
        assert_eq!(reg.meta.param, ParamKind::Toggle);
    }

    #[test]
    fn test_sepia_passes_count() {
        let reg = registry_by_id("sepia").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_sepia_make_uniform_known_value() {
        let reg = registry_by_id("sepia").unwrap();
        let bytes = (reg.make_uniform)(&[]);
        let expected = bytemuck::bytes_of(&SepiaParams { _unused: [0.0; 4] });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_sepia_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sepia",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by sepia");
        }
    }

    #[test]
    fn test_sepia_black_stays_black() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Black input (all zeros) → matrix multiplication yields all zeros.
        let img = make_solid_image(2, 2, 0, 0, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sepia",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(
                pixel[0], 0,
                "R: black input should stay 0, got {}",
                pixel[0]
            );
            assert_eq!(
                pixel[1], 0,
                "G: black input should stay 0, got {}",
                pixel[1]
            );
            assert_eq!(
                pixel[2], 0,
                "B: black input should stay 0, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_sepia_output_channels_ordered_correctly() {
        // For any neutral-grey input, the sepia matrix outputs R > G > B,
        // which is the characteristic warm-brown tone.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Mid-grey: equal R, G, B.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sepia",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[0] >= pixel[1],
                "R should be >= G for grey input (warm tone): R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                pixel[1] >= pixel[2],
                "G should be >= B for grey input (warm tone): G={}, B={}",
                pixel[1],
                pixel[2]
            );
        }
    }

    #[test]
    fn test_sepia_white_input_output_in_expected_range() {
        // White sRGB input (65535 u16) is linearized by ingest to 1.0 linear.
        // The sepia matrix produces these linear-light outputs:
        //   R = 0.393 + 0.769 + 0.189 = 1.351 (exceeds 1.0 — headroom preserved)
        //   G = 0.349 + 0.686 + 0.168 = 1.203
        //   B = 0.272 + 0.534 + 0.131 = 0.937
        // The present shader re-encodes to sRGB before writing u16, so:
        //   R and G (>1.0) saturate to 65535.
        //   B: sRGB(0.937) ≈ 0.9718 → u16 ≈ 63686.
        // A tolerance of 512 absorbs f16 quantization in the Rgba16Float intermediate.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 65535, 65535, 65535);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "sepia",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 64,
                "R: white should saturate near 65535, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 65535).abs() <= 64,
                "G: white should saturate near 65535, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 63686).abs() <= 512,
                "B: white input expected ~63686 (sRGB-encoded 0.937), got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_sepia_chained_with_brightness() {
        // Applying brightness before sepia must not break the warm-tone property
        // (R >= G >= B for grey-ish inputs).
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "brightness",
                    values: vec![0.1],
                },
                Transform {
                    shader_id: "sepia",
                    values: vec![],
                },
            ],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[0] >= pixel[1],
                "R should be >= G after brightness+sepia: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                pixel[1] >= pixel[2],
                "G should be >= B after brightness+sepia: G={}, B={}",
                pixel[1],
                pixel[2]
            );
        }
    }
}
