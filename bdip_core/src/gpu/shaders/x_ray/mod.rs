use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, PassScale, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct XRayParams {
    pub _unused: [f32; 4],
}

impl TransformShader for XRayParams {
    const ID: &'static str = "x_ray";
    const DISPLAY_NAME: &'static str = "X-Ray";
    const DESCRIPTION: &'static str = "Simulates an X-ray appearance: inverts color, converts to grayscale, then applies high contrast.";
    const PARAM: ParamKind = ParamKind::Toggle;
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "x_ray",
        wgsl_source: include_str!("x_ray.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(_: &[f32]) -> Self {
        Self { _unused: [0.0; 4] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<XRayParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_x_ray_registry_entry_exists() {
        assert!(registry_by_id("x_ray").is_some());
    }

    #[test]
    fn test_x_ray_registry_metadata() {
        let reg = registry_by_id("x_ray").unwrap();
        assert_eq!(reg.meta.display_name, "X-Ray");
        assert_eq!(reg.meta.param, ParamKind::Toggle);
        assert_eq!(reg.meta.passes.len(), 1);
    }

    #[test]
    fn test_x_ray_make_uniform_known_value() {
        let reg = registry_by_id("x_ray").unwrap();
        let bytes = (reg.make_uniform)(&[]);
        let expected = bytemuck::bytes_of(&XRayParams { _unused: [0.0; 4] });
        assert_eq!(bytes, expected);
    }

    /// Pure black input: invert → white (1.0), grayscale → 1.0, contrast → 1.0 → white output.
    #[test]
    fn test_x_ray_black_produces_white() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 0, 0, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "x_ray",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 128,
                "R: black input should produce white, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 65535).abs() <= 128,
                "G: black input should produce white, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 65535).abs() <= 128,
                "B: black input should produce white, got {}",
                pixel[2]
            );
        }
    }

    /// Pure white input: invert → black (0.0), grayscale → 0.0, contrast → 0.0 → black output.
    #[test]
    fn test_x_ray_white_produces_black() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 65535, 65535, 65535);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "x_ray",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[0] <= 128,
                "R: white input should produce black, got {}",
                pixel[0]
            );
            assert!(
                pixel[1] <= 128,
                "G: white input should produce black, got {}",
                pixel[1]
            );
            assert!(
                pixel[2] <= 128,
                "B: white input should produce black, got {}",
                pixel[2]
            );
        }
    }

    /// Output channels must be equal (grayscale), regardless of colored input.
    #[test]
    fn test_x_ray_produces_grayscale_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Colored input with distinct channels.
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "x_ray",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "R and G must be equal (grayscale): R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[1] as i32 - pixel[2] as i32).abs() <= 64,
                "G and B must be equal (grayscale): G={}, B={}",
                pixel[1],
                pixel[2]
            );
        }
    }

    /// Alpha channel must pass through unchanged.
    #[test]
    fn test_x_ray_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "x_ray",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by x_ray");
        }
    }

    /// High-contrast step: a mid-grey input must produce an output darker than the
    /// post-invert luminance, verifying the squaring curve compresses midtones.
    ///
    /// Pipeline trace for a sRGB-encoded neutral grey (32767/65535 ≈ 0.500 sRGB):
    ///   ingest:    srgb_to_linear(0.500) ≈ 0.214 linear
    ///   invert:    1.0 − 0.214 = 0.786 linear
    ///   grayscale: 0.786 (equal channels, so unchanged)
    ///   contrast:  0.786² ≈ 0.618 linear
    ///   present:   linear_to_srgb(0.618) ≈ 0.808 → ~52970 u16
    ///
    /// Without the squaring step the post-invert luminance would encode as ~59300.
    /// The contrast step must push the output below that un-squared value.
    #[test]
    fn test_x_ray_high_contrast_darkens_midtones() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Neutral grey — all sRGB channels equal, so invert+grayscale leaves a
        // single luminance value of ~0.786 linear. Squaring that to ~0.618 linear
        // and re-encoding to sRGB gives ~52970, well below the un-squared ~59300.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "x_ray",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            // Output must be below the un-squared post-invert luminance (~59300).
            // Expected ~52970; allow ±1000 tolerance.
            assert!(
                pixel[0] < 58000,
                "high contrast step must darken midtones vs un-squared invert; got R={}",
                pixel[0]
            );
            // Output must not be trivially dark (confirms squaring did not over-crush).
            assert!(
                pixel[0] > 45000,
                "high contrast output should remain a light grey, not black; got R={}",
                pixel[0]
            );
        }
    }

    /// Chaining x_ray after brightness shift: output must still be grayscale.
    #[test]
    fn test_x_ray_chained_after_brightness_produces_grayscale() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);
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
                    shader_id: "x_ray",
                    values: vec![],
                },
            ],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "R and G must be equal after brightness+x_ray: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[1] as i32 - pixel[2] as i32).abs() <= 64,
                "G and B must be equal after brightness+x_ray: G={}, B={}",
                pixel[1],
                pixel[2]
            );
        }
    }
}
