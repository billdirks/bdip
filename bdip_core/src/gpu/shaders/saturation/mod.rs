use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, SliderDef, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SaturationParams {
    pub value: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for SaturationParams {
    const ID: &'static str = "saturation";
    const DISPLAY_NAME: &'static str = "Saturation";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Amount",
        min: -1.0,
        max: 1.0,
        default: 0.0,
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "saturation",
        wgsl_source: include_str!("saturation.wgsl"),
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

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    SaturationParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_saturation_registry_entry_exists() {
        assert!(registry_by_id("saturation").is_some());
    }

    #[test]
    fn test_saturation_registry_metadata() {
        let reg = registry_by_id("saturation").unwrap();
        assert_eq!(reg.meta.display_name, "Saturation");
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
    fn test_saturation_make_uniform_known_value() {
        let reg = registry_by_id("saturation").unwrap();
        let bytes = (reg.make_uniform)(&[0.5]);
        let expected = bytemuck::bytes_of(&SaturationParams {
            value: 0.5,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_saturation_zero_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Use a non-gray, non-neutral color so saturation has values to act on.
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "saturation",
                values: vec![0.0],
            }],
        );

        // saturation_offset=0 → scale=1.0 → identity; only f16 rounding applies.
        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: expected ~32767, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 16384).abs() <= 64,
                "G: expected ~16384, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 8192).abs() <= 64,
                "B: expected ~8192, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_saturation_negative_one_produces_grayscale() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Pure red: R=65535, G=0, B=0 in sRGB. After desaturation, all channels
        // equal Rec.709 luminance of the linear values: lum = 0.2126*1.0 = 0.2126.
        let img = make_solid_image(2, 2, 65535, 0, 0);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "saturation",
                values: vec![-1.0],
            }],
        );

        for pixel in out_img.pixels() {
            // All three channels should be equal — the defining property of grayscale.
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "R and G should be equal after full desaturation: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[1] as i32 - pixel[2] as i32).abs() <= 64,
                "G and B should be equal after full desaturation: G={}, B={}",
                pixel[1],
                pixel[2]
            );
            // The gray value should reflect luminance, not zero or the original red.
            assert!(
                pixel[0] > 0 && pixel[0] < 65535,
                "Desaturated value should be a mid-tone, got {}",
                pixel[0]
            );
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_saturation_positive_increases_color() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Input: warm color where R > G > B.
        // After positive saturation: R should increase (it's above luminance),
        // G and B should decrease (they're below luminance).
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "saturation",
                values: vec![0.5],
            }],
        );

        for pixel in out_img.pixels() {
            assert!(
                pixel[0] > 32767,
                "R should increase with positive saturation: got {}",
                pixel[0]
            );
            assert!(
                pixel[1] < 16384,
                "G should decrease with positive saturation: got {}",
                pixel[1]
            );
            assert!(
                pixel[2] < 8192,
                "B should decrease with positive saturation: got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }
}
