use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HighKeyParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for HighKeyParams {
    const ID: &'static str = "high_key";
    const DISPLAY_NAME: &'static str = "High Key";
    const DESCRIPTION: &'static str =
        "Simulates high-key lighting: boosts exposure and lifts shadows toward white.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Intensity of the high-key effect. 0 is no change; 1 is fully high-key.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "high_key",
        wgsl_source: include_str!("high_key.wgsl"),
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

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<HighKeyParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    fn make_engine_and_renderer() -> (GpuEngine, Renderer) {
        let engine = GpuEngine::new().unwrap();
        let renderer = Renderer::new(&engine);
        (engine, renderer)
    }

    #[test]
    fn test_high_key_registry_entry_exists() {
        assert!(registry_by_id("high_key").is_some());
    }

    #[test]
    fn test_high_key_registry_metadata() {
        let reg = registry_by_id("high_key").unwrap();
        assert_eq!(reg.meta.display_name, "High Key");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Intensity of the high-key effect. 0 is no change; 1 is fully high-key.",
            }])
        );
    }

    #[test]
    fn test_high_key_make_uniform_known_value() {
        let reg = registry_by_id("high_key").unwrap();
        let bytes = (reg.make_uniform)(&[0.5]);
        let expected = bytemuck::bytes_of(&HighKeyParams {
            strength: 0.5,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_high_key_identity_at_zero_strength() {
        // strength=0 must produce identity: scale=2^0=1, floor=0.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 10794, 25700, 51400);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "high_key",
                values: vec![0.0],
            }],
        );

        // sRGB→linear→sRGB round-trip with f16 precision; allow ±64 LSB.
        for pixel in out.pixels() {
            assert!((pixel[0] as i32 - 10794).abs() <= 64);
            assert!((pixel[1] as i32 - 25700).abs() <= 64);
            assert!((pixel[2] as i32 - 51400).abs() <= 64);
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_high_key_brightens_image_at_full_strength() {
        // At strength=1 mid-gray (32767 u16, ≈0.500 sRGB, ≈0.214 linear) must
        // get significantly brighter than the input.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "high_key",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                pixel[0] > 55000,
                "Expected R to be bright at full strength, got {}",
                pixel[0]
            );
            assert!(
                pixel[1] > 55000,
                "Expected G to be bright at full strength, got {}",
                pixel[1]
            );
            assert!(
                pixel[2] > 55000,
                "Expected B to be bright at full strength, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_high_key_lifts_blacks() {
        // Pure black input (0, 0, 0 linear) should be lifted above zero at non-zero strength.
        // The shadow-lift floor = strength * 0.3 * (1 - 0) = strength * 0.3.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "high_key",
                values: vec![1.0],
            }],
        );

        // At strength=1: linear_out = 0 * 4 + 0.3 * 1 = 0.3 linear
        // linear_to_srgb(0.3) ≈ 0.585 sRGB → u16 ≈ 38,340
        for pixel in out.pixels() {
            assert!(
                pixel[0] > 30000,
                "Expected black to be lifted, got R={}",
                pixel[0]
            );
            assert!(
                pixel[1] > 30000,
                "Expected black to be lifted, got G={}",
                pixel[1]
            );
            assert!(
                pixel[2] > 30000,
                "Expected black to be lifted, got B={}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_high_key_alpha_preserved() {
        // Alpha channel must pass through unmodified regardless of strength.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "high_key",
                values: vec![0.5],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "Alpha should be preserved");
        }
    }

    #[test]
    fn test_high_key_chaining_with_brightness() {
        // Chain high_key into brightness to verify in-VRAM handoff works correctly.
        let (engine, mut renderer) = make_engine_and_renderer();
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "high_key",
                    values: vec![0.5],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );

        // brightness at 0 is identity — output must still be brighter than input.
        for pixel in out.pixels() {
            assert!(
                pixel[0] > 32767,
                "Chained output should be brighter than input, got R={}",
                pixel[0]
            );
        }
    }
}
