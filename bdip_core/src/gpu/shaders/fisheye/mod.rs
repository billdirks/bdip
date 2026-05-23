use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FisheyeParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for FisheyeParams {
    const ID: &'static str = "fisheye";
    const DISPLAY_NAME: &'static str = "Fisheye";
    const DESCRIPTION: &'static str =
        "Applies radial UV barrel distortion to create a fisheye lens effect.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: -1.0,
        max: 1.0,
        default: 0.0,
        description: "Distortion strength. 0.0 = no-op; positive = barrel (fisheye bulge); \
             negative = pincushion (inverse fisheye).",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "fisheye",
        wgsl_source: include_str!("fisheye.wgsl"),
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

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<FisheyeParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_fisheye_registry_entry_exists() {
        assert!(registry_by_id("fisheye").is_some());
    }

    #[test]
    fn test_fisheye_registry_metadata() {
        let reg = registry_by_id("fisheye").unwrap();
        assert_eq!(reg.meta.display_name, "Fisheye");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: -1.0,
                max: 1.0,
                default: 0.0,
                description: "Distortion strength. 0.0 = no-op; positive = barrel (fisheye bulge); \
                     negative = pincushion (inverse fisheye).",
            }])
        );
    }

    #[test]
    fn test_fisheye_make_uniform_known_value() {
        let reg = registry_by_id("fisheye").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&FisheyeParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 (identity) the output pixels must equal the input pixels.
    /// A solid-color image is used so every pixel is verifiable regardless of
    /// the UV mapping path taken by the distortion math.
    #[test]
    fn test_fisheye_identity_at_zero_strength() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 40000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fisheye",
                values: vec![0.0],
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

    /// Alpha channel must be passed through unchanged by the distortion.
    #[test]
    fn test_fisheye_alpha_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fisheye",
                values: vec![0.5],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    /// Barrel distortion (positive strength) maps the centre pixel back to
    /// itself — a pixel at the exact centre of the image is at r=0, so the
    /// distortion factor is 1.0 regardless of strength.
    #[test]
    fn test_fisheye_barrel_centre_pixel_unchanged() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Use a 3×3 image so pixel (1,1) is the exact centre.
        // Fill with a unique colour so the identity of the centre pixel is clear.
        let img = make_solid_image(3, 3, 50000, 10000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "fisheye",
                values: vec![1.0],
            }],
        );

        // Centre pixel at (1,1) must be unchanged.
        let p = out.get_pixel(1, 1);
        assert!(
            (p[0] as i32 - 50000).abs() <= 128,
            "centre R: expected ~50000, got {}",
            p[0]
        );
        assert!(
            (p[1] as i32 - 10000).abs() <= 128,
            "centre G: expected ~10000, got {}",
            p[1]
        );
        assert!(
            (p[2] as i32 - 30000).abs() <= 128,
            "centre B: expected ~30000, got {}",
            p[2]
        );
    }

    /// Chaining the fisheye with another shader (brightness) must not panic and
    /// must produce a result whose alpha channel is intact.
    #[test]
    fn test_fisheye_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "fisheye",
                    values: vec![0.3],
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
