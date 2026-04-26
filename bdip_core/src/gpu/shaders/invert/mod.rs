use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, PassScale, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InvertParams {
    pub _unused: [f32; 4],
}

impl TransformShader for InvertParams {
    const ID: &'static str = "invert";
    const DISPLAY_NAME: &'static str = "Invert";
    const PARAM: ParamKind = ParamKind::Toggle;
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "invert",
        wgsl_source: include_str!("invert.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(_: &[f32]) -> Self {
        Self { _unused: [0.0; 4] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<InvertParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_invert_registry_entry_exists() {
        assert!(registry_by_id("invert").is_some());
    }

    #[test]
    fn test_invert_registry_metadata() {
        let reg = registry_by_id("invert").unwrap();
        assert_eq!(reg.meta.display_name, "Invert");
        assert_eq!(reg.meta.param, ParamKind::Toggle);
    }

    #[test]
    fn test_invert_make_uniform_known_value() {
        let reg = registry_by_id("invert").unwrap();
        let bytes = (reg.make_uniform)(&[]);
        let expected = bytemuck::bytes_of(&InvertParams { _unused: [0.0; 4] });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_invert_shader() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Note: linear-light invert means 1.0 - linear_value.
        let img = make_solid_image(2, 2, 0, 65535, 32767);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "invert",
                values: vec![],
            }],
        );

        for pixel in out_img.pixels() {
            // R: 0 → inverted → 65535
            assert!(
                (pixel[0] as i32 - 65535).abs() <= 100,
                "R: expected ~65535, got {}",
                pixel[0]
            );
            // G: 65535 → inverted → 0
            assert!(pixel[1] <= 100, "G: expected ~0, got {}", pixel[1]);
            // Alpha preserved
            assert_eq!(pixel[3], 65535);
        }
    }

    #[test]
    fn test_double_invert_restores_original() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 10794, 25700, 51400);
        let out_img = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "invert",
                    values: vec![],
                },
                Transform {
                    shader_id: "invert",
                    values: vec![],
                },
            ],
        );

        for pixel in out_img.pixels() {
            assert!(
                (pixel[0] as i32 - 10794).abs() <= 128,
                "R: expected ~10794, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 25700).abs() <= 128,
                "G: expected ~25700, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 51400).abs() <= 128,
                "B: expected ~51400, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535);
        }
    }
}
