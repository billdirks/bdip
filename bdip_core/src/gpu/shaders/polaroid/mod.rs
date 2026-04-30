use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension, ParamKind, PassDef, PassInput,
    PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PolaroidParams {
    pub grade: f32,
    pub border: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for PolaroidParams {
    const ID: &'static str = "polaroid";
    const DISPLAY_NAME: &'static str = "Polaroid";
    const DESCRIPTION: &'static str =
        "Applies Polaroid film color science and a classic white border frame.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Grade",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Intensity of the Polaroid color grade (warm mids, faded blacks); \
                          0 leaves the image ungraded.",
        },
        SliderDef {
            name: "Border",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Opacity of the white Polaroid border frame; \
                          0 hides the border, 1 shows it at full white.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "polaroid_grade",
            wgsl_source: include_str!("polaroid_grade.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("grade"),
            output_scale: PassScale::Full,
            aux_textures: &[AuxTextureDef {
                name: "polaroid_lut_64",
                dimension: AuxTextureDimension::D3,
                filter: AuxSamplerFilter::Linear,
            }],
        },
        PassDef {
            label: "polaroid_border",
            wgsl_source: include_str!("polaroid_border.wgsl"),
            inputs: &[PassInput::Scratch("grade")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            grade: values[0],
            border: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<PolaroidParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{
        AuxTextureDimension, ParamKind, SliderDef, Transform, registry_by_id,
    };
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_polaroid_registry_entry_exists() {
        assert!(registry_by_id("polaroid").is_some());
    }

    #[test]
    fn test_polaroid_registry_metadata() {
        let reg = registry_by_id("polaroid").unwrap();
        assert_eq!(reg.meta.display_name, "Polaroid");
        assert_eq!(reg.meta.passes.len(), 2, "must have exactly 2 passes");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Grade",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Intensity of the Polaroid color grade (warm mids, faded blacks); \
                                  0 leaves the image ungraded.",
                },
                SliderDef {
                    name: "Border",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Opacity of the white Polaroid border frame; \
                                  0 hides the border, 1 shows it at full white.",
                },
            ])
        );
    }

    #[test]
    fn test_polaroid_make_uniform_known_value() {
        let reg = registry_by_id("polaroid").unwrap();
        let bytes = (reg.make_uniform)(&[0.75, 0.5]);
        let expected = bytemuck::bytes_of(&PolaroidParams {
            grade: 0.75,
            border: 0.5,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_polaroid_grade_pass_uses_lut_aux_texture() {
        let reg = registry_by_id("polaroid").unwrap();
        let grade_pass = &reg.meta.passes[0];
        assert_eq!(
            grade_pass.aux_textures.len(),
            1,
            "grade pass must declare exactly one aux"
        );
        assert_eq!(grade_pass.aux_textures[0].name, "polaroid_lut_64");
        assert_eq!(
            grade_pass.aux_textures[0].dimension,
            AuxTextureDimension::D3
        );
        assert_eq!(grade_pass.aux_textures[0].filter, AuxSamplerFilter::Linear);
    }

    #[test]
    fn test_polaroid_border_pass_has_no_aux_texture() {
        let reg = registry_by_id("polaroid").unwrap();
        let border_pass = &reg.meta.passes[1];
        assert_eq!(
            border_pass.aux_textures.len(),
            0,
            "border pass must declare no aux textures"
        );
    }

    #[test]
    fn test_polaroid_grade_zero_border_zero_is_passthrough() {
        // grade=0, border=0: grade pass blends 0% (noop), border pass applies 0% white.
        // Full pipeline should be within the sRGB↔linear roundtrip tolerance.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polaroid",
                values: vec![0.0, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32767).abs() <= 128,
                "R: grade=0 border=0 must be within ±128 of input, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 32767).abs() <= 128,
                "G: grade=0 border=0 must be within ±128 of input, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 32767).abs() <= 128,
                "B: grade=0 border=0 must be within ±128 of input, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_polaroid_grade_warms_midtones() {
        // The Polaroid grade boosts red and cuts blue in midtones (≈0.5 sRGB gray).
        // A neutral mid-gray input should come out with R > input and B < input.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polaroid",
                values: vec![1.0, 0.0],
            }],
        );
        let pixel = out.get_pixel(1, 1);
        assert!(
            pixel[0] > 32767,
            "red channel should be boosted by warm cast, got {}",
            pixel[0]
        );
        assert!(
            pixel[2] < 32767,
            "blue channel should be reduced by warm cast, got {}",
            pixel[2]
        );
    }

    #[test]
    fn test_polaroid_border_one_makes_corner_white() {
        // With border=1.0, pixels in the border area (e.g. corner 0,0) become white.
        // A 4×4 image: pixel (0,0) has uv=(0,0), which is outside the inner photo area.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 0, 0, 0);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polaroid",
                values: vec![1.0, 1.0],
            }],
        );
        let corner = out.get_pixel(0, 0);
        // Allow ±1 for f16/sRGB encoding rounding at the 1.0 boundary.
        assert!(
            corner[0] >= 65534,
            "corner R must be white with border=1, got {}",
            corner[0]
        );
        assert!(
            corner[1] >= 65534,
            "corner G must be white with border=1, got {}",
            corner[1]
        );
        assert!(
            corner[2] >= 65534,
            "corner B must be white with border=1, got {}",
            corner[2]
        );
    }

    #[test]
    fn test_polaroid_border_zero_leaves_no_white() {
        // With border=0, even pixels in the border region are not forced to white.
        // Use mid-gray input so the graded output is clearly not white.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polaroid",
                values: vec![1.0, 0.0],
            }],
        );
        let corner = out.get_pixel(0, 0);
        assert!(
            corner[0] < 60000,
            "corner R must not be white with border=0, got {}",
            corner[0]
        );
        assert!(
            corner[2] < 60000,
            "corner B must not be white with border=0, got {}",
            corner[2]
        );
    }

    #[test]
    fn test_polaroid_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polaroid",
                values: vec![1.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    #[test]
    fn test_polaroid_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 16384, 8192);
        let params = vec![1.0f32, 1.0];
        let out_1 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polaroid",
                values: params.clone(),
            }],
        );
        let out_2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "polaroid",
                values: params,
            }],
        );
        for (a, b) in out_1.pixels().zip(out_2.pixels()) {
            assert_eq!(a, b, "identical params must produce identical output");
        }
    }
}
