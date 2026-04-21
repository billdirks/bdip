use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, SliderDef, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ContrastParams {
    pub value: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for ContrastParams {
    const ID: &'static str = "contrast";
    const DISPLAY_NAME: &'static str = "Contrast";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Amount",
        min: -1.0,
        max: 1.0,
        default: 0.0,
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "contrast",
        wgsl_source: include_str!("contrast.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            value: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ContrastParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_contrast_registry_entry_exists() {
        assert!(registry_by_id("contrast").is_some());
    }

    #[test]
    fn test_contrast_registry_metadata() {
        let reg = registry_by_id("contrast").unwrap();
        assert_eq!(reg.meta.display_name, "Contrast");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Amount",
                min: -1.0,
                max: 1.0,
                default: 0.0,
            }])
        );
    }

    #[test]
    fn test_contrast_make_uniform_known_value() {
        let reg = registry_by_id("contrast").unwrap();
        let bytes = (reg.make_uniform)(&[0.5]);
        let expected = bytemuck::bytes_of(&ContrastParams {
            value: 0.5,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_contrast_zero_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Use a non-neutral color so the shader has meaningful values to act on.
        let img = make_solid_image(2, 2, 10794, 25700, 51400);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "contrast",
                values: vec![0.0],
            }],
        );

        // contrast_offset=0 → scale=1.0 → identity; only f16 rounding applies.
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 10794).abs() <= 64,
                "R: expected ~10794, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 25700).abs() <= 64,
                "G: expected ~25700, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 51400).abs() <= 64,
                "B: expected ~51400, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_contrast_max_positive_clamps_below_midpoint_to_black() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 50% gray sRGB (≈0.214 linear) is below the 0.5 linear midpoint.
        // contrast=1.0 → scale=2.0 → (0.214 - 0.5)*2.0 + 0.5 = -0.072 → clamped to 0.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "contrast",
                values: vec![1.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[0], 0, "R: below-midpoint pixel should clamp to 0");
            assert_eq!(pixel[1], 0, "G: below-midpoint pixel should clamp to 0");
            assert_eq!(pixel[2], 0, "B: below-midpoint pixel should clamp to 0");
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_contrast_max_positive_pushes_above_midpoint_brighter() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 51400/65535 ≈ 0.784 sRGB → ≈0.577 linear (above 0.5 midpoint).
        // contrast=1.0 → (0.577 - 0.5)*2.0 + 0.5 = 0.655 linear → sRGB ≈ 0.829 → u16 ≈ 54366.
        let img = make_solid_image(2, 2, 51400, 51400, 51400);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "contrast",
                values: vec![1.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[0] > 51400,
                "R: above-midpoint pixel should brighten with positive contrast, got {}",
                pixel[0]
            );
            assert!(
                (pixel[0] as i32 - 54366).abs() <= 128,
                "R: expected ~54366, got {}",
                pixel[0]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_contrast_max_negative_flattens_to_neutral_gray() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // contrast=-1.0 → scale=0.0 → all channels become 0.5 linear regardless of input.
        // 0.5 linear → sRGB ≈ 0.735 → u16 ≈ 48184.
        let img = make_solid_image(2, 2, 0, 0, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "contrast",
                values: vec![-1.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 48184).abs() <= 128,
                "R: expected neutral gray ~48184, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 48184).abs() <= 128,
                "G: expected neutral gray ~48184, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 48184).abs() <= 128,
                "B: expected neutral gray ~48184, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_contrast_preserves_alpha() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "contrast",
                values: vec![1.0],
            }],
        );

        for pixel in out_img.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be unchanged by contrast");
        }
    }
}
