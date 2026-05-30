use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstamaticParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for InstamaticParams {
    const ID: &'static str = "instamatic";
    const DISPLAY_NAME: &'static str = "Instamatic";
    const DESCRIPTION: &'static str = "Simulates the color rendering of cheap instant cameras — faded, warm with a \
         yellow-green midtone cast, lifted shadows, and a subtle vignette.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend strength of the Instamatic effect; 0 is unchanged, 1 is the full look.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "instamatic",
        wgsl_source: include_str!("instamatic.wgsl"),
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
    InstamaticParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_instamatic_registry_entry_exists() {
        assert!(registry_by_id("instamatic").is_some());
    }

    #[test]
    fn test_instamatic_registry_metadata() {
        let reg = registry_by_id("instamatic").unwrap();
        assert_eq!(reg.meta.display_name, "Instamatic");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Blend strength of the Instamatic effect; 0 is unchanged, 1 is the \
                              full look.",
            }])
        );
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
    }

    #[test]
    fn test_instamatic_make_uniform_known_value() {
        let reg = registry_by_id("instamatic").unwrap();
        let bytes = (reg.make_uniform)(&[0.6]);
        let expected = bytemuck::bytes_of(&InstamaticParams {
            strength: 0.6,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_instamatic_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "instamatic",
                values: vec![0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 64,
                "R: strength=0 must return original, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 16384).abs() <= 64,
                "G: strength=0 must return original, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 8192).abs() <= 64,
                "B: strength=0 must return original, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_instamatic_full_strength_warms_image() {
        // A neutral-grey input at full strength must be warmer (R > B) due to
        // the warm channel balance and yellow-green midtone cast.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Mid-grey sRGB value (~0.5 sRGB → ~0.214 linear) hits the midtone range.
        let img = make_solid_image(8, 8, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "instamatic",
                values: vec![1.0],
            }],
        );
        let mean_r: f64 = out.pixels().map(|p| p[0] as f64).sum::<f64>() / 64.0;
        let mean_b: f64 = out.pixels().map(|p| p[2] as f64).sum::<f64>() / 64.0;
        assert!(
            mean_r > mean_b,
            "full strength on grey must produce warmer (R>B) output, R={mean_r:.0} B={mean_b:.0}"
        );
    }

    #[test]
    fn test_instamatic_full_strength_lifts_shadows() {
        // Pure black input at full strength must be lifted above 0 by the shadow
        // lift component.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "instamatic",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0,
                "pure black must be lifted at full strength, got pixel {:?}",
                pixel
            );
        }
    }

    #[test]
    fn test_instamatic_full_strength_compresses_highlights() {
        // Pure white input at full strength must be compressed below the original
        // white level due to highlight fading.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a central pixel to avoid the vignette darkening the comparison.
        let img = make_solid_image(64, 64, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "instamatic",
                values: vec![1.0],
            }],
        );
        // Check a central pixel far from the vignette edge.
        let center_pixel = out.get_pixel(32, 32);
        assert!(
            center_pixel[0] < 65535,
            "highlights must be compressed at full strength, R={}",
            center_pixel[0]
        );
    }

    #[test]
    fn test_instamatic_vignette_darkens_corners() {
        // At full strength the corners of a uniform white image must be darker
        // than the centre due to the vignette.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(64, 64, 65535, 65535, 65535);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "instamatic",
                values: vec![1.0],
            }],
        );
        let center = out.get_pixel(32, 32);
        let corner = out.get_pixel(0, 0);
        assert!(
            center[0] > corner[0],
            "centre must be brighter than corner: centre R={} corner R={}",
            center[0],
            corner[0]
        );
    }

    #[test]
    fn test_instamatic_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "instamatic",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    #[test]
    fn test_instamatic_chaining_with_brightness() {
        // Verify that the shader output can be fed into another shader without
        // errors, confirming correct texture format and pipeline wiring.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "instamatic",
                    values: vec![0.5],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.1],
                },
            ],
        );
        let any_nonzero = out.pixels().any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0);
        assert!(any_nonzero, "chained output must contain non-zero pixels");
    }
}
