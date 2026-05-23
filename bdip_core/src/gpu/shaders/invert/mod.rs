use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, PassScale, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InvertParams {
    pub _unused: [f32; 4],
}

impl TransformShader for InvertParams {
    const ID: &'static str = "invert";
    const DISPLAY_NAME: &'static str = "Invert";
    const DESCRIPTION: &'static str = "Inverts all color channels in linear light (1 − value).";
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

    fn run_invert(
        engine: &GpuEngine,
        renderer: &mut Renderer,
        r: u16,
        g: u16,
        b: u16,
    ) -> crate::Rgba16Image {
        let img = make_solid_image(2, 2, r, g, b);
        roundtrip(
            renderer,
            engine,
            &img,
            &[Transform {
                shader_id: "invert",
                values: vec![],
            }],
        )
    }

    #[test]
    fn test_invert_registry_entry_exists() {
        assert!(registry_by_id("invert").is_some());
    }

    #[test]
    fn test_invert_display_name() {
        let reg = registry_by_id("invert").unwrap();
        assert_eq!(reg.meta.display_name, "Invert");
    }

    #[test]
    fn test_invert_param_kind() {
        let reg = registry_by_id("invert").unwrap();
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
    fn test_invert_black_channel_becomes_white() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let out = run_invert(&engine, &mut renderer, 0, 32767, 32767);
        let r = out.get_pixel(0, 0)[0];
        assert!(
            (r as i32 - 65535).abs() <= 100,
            "R: expected ~65535, got {r}"
        );
    }

    #[test]
    fn test_invert_white_channel_becomes_black() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let out = run_invert(&engine, &mut renderer, 32767, 65535, 32767);
        let g = out.get_pixel(0, 0)[1];
        assert!(g <= 100, "G: expected ~0, got {g}");
    }

    #[test]
    fn test_invert_midtone_channel_inverts() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let out = run_invert(&engine, &mut renderer, 32767, 32767, 32767);
        let b = out.get_pixel(0, 0)[2];
        // 32767 ≈ 0.5 sRGB ≈ 0.214 linear; inverted → 0.786 linear ≈ 0.899 sRGB ≈ 58922 u16.
        assert!(
            (b as i32 - 58922).abs() <= 300,
            "B: expected ~58922, got {b}"
        );
    }

    #[test]
    fn test_invert_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let out = run_invert(&engine, &mut renderer, 0, 0, 0);
        assert_eq!(out.get_pixel(0, 0)[3], 65535, "alpha must be preserved");
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
