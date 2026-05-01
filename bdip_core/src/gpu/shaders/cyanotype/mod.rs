use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, PassScale, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CyanotypeParams {
    pub _unused: [f32; 4],
}

impl TransformShader for CyanotypeParams {
    const ID: &'static str = "cyanotype";
    const DISPLAY_NAME: &'static str = "Cyanotype";
    const DESCRIPTION: &'static str = "Simulates the historical cyanotype photographic printing process: converts \
         to grayscale then tints with a Prussian blue/cyan palette.";
    const PARAM: ParamKind = ParamKind::Toggle;
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "cyanotype",
        wgsl_source: include_str!("cyanotype.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(_: &[f32]) -> Self {
        Self { _unused: [0.0; 4] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    CyanotypeParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_cyanotype_registry_entry_exists() {
        assert!(registry_by_id("cyanotype").is_some());
    }

    #[test]
    fn test_cyanotype_registry_metadata() {
        let reg = registry_by_id("cyanotype").unwrap();
        assert_eq!(reg.meta.display_name, "Cyanotype");
        assert_eq!(reg.meta.param, ParamKind::Toggle);
    }

    #[test]
    fn test_cyanotype_passes_count() {
        let reg = registry_by_id("cyanotype").unwrap();
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_cyanotype_make_uniform_known_value() {
        let reg = registry_by_id("cyanotype").unwrap();
        let bytes = (reg.make_uniform)(&[]);
        let expected = bytemuck::bytes_of(&CyanotypeParams { _unused: [0.0; 4] });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_cyanotype_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cyanotype",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by cyanotype");
        }
    }

    #[test]
    fn test_cyanotype_black_maps_to_deep_blue() {
        // Black input (luma = 0.0) must map to the shadow colour: (0.0, 0.05, 0.20) linear.
        // After sRGB re-encoding and u16 quantisation:
        //   R = 0.0      → u16 = 0
        //   G = 0.05     → sRGB ≈ 0.2485 → u16 ≈ 16291
        //   B = 0.20     → sRGB ≈ 0.4836 → u16 ≈ 31686
        // Tolerance of 512 covers f16 quantisation in the Rgba16Float intermediate.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 0, 0, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cyanotype",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[0] <= 512,
                "R: black input should map near 0 (deep blue shadow), got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 16291).abs() <= 512,
                "G: black input should map near 16291, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 31686).abs() <= 512,
                "B: black input should map near 31686, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_cyanotype_white_maps_to_pale_blue_white() {
        // White input (luma = 1.0) must map to the highlight colour: (0.85, 0.93, 1.0) linear.
        // After sRGB re-encoding and u16 quantisation:
        //   R = 0.85     → sRGB ≈ 0.9247 → u16 ≈ 60609
        //   G = 0.93     → sRGB ≈ 0.9637 → u16 ≈ 63165
        //   B = 1.0      → sRGB = 1.0    → u16 = 65535
        // Tolerance of 512 covers f16 quantisation.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 65535, 65535, 65535);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cyanotype",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 60609).abs() <= 512,
                "R: white input should map near 60609, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 63165).abs() <= 512,
                "G: white input should map near 63165, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 65535).abs() <= 64,
                "B: white input should map to 65535, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_cyanotype_blue_dominant_for_grey_input() {
        // For any neutral-grey input, the cyanotype tint must satisfy B >= G >= R,
        // which is the characteristic prussian-blue tone.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "cyanotype",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[2] >= pixel[1],
                "B should be >= G for grey input (blue tone): B={}, G={}",
                pixel[2],
                pixel[1]
            );
            assert!(
                pixel[1] >= pixel[0],
                "G should be >= R for grey input (blue tone): G={}, R={}",
                pixel[1],
                pixel[0]
            );
        }
    }

    #[test]
    fn test_cyanotype_chained_with_brightness() {
        // Applying brightness before cyanotype must not break the blue-dominant property
        // (B >= G >= R for grey-ish inputs).
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
                    shader_id: "cyanotype",
                    values: vec![],
                },
            ],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[2] >= pixel[1],
                "B should be >= G after brightness+cyanotype: B={}, G={}",
                pixel[2],
                pixel[1]
            );
            assert!(
                pixel[1] >= pixel[0],
                "G should be >= R after brightness+cyanotype: G={}, R={}",
                pixel[1],
                pixel[0]
            );
        }
    }
}
