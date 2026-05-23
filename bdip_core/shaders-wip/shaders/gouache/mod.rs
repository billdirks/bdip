use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GouacheParams {
    pub strength: f32,
    pub _padding: [f32; 3], // pad to 16 bytes for WebGPU uniform alignment
}

impl TransformShader for GouacheParams {
    const ID: &'static str = "gouache";
    const DISPLAY_NAME: &'static str = "Gouache";
    const DESCRIPTION: &'static str =
        "Simulates opaque gouache paint: smooths fine detail and boosts colour saturation.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0, // identity: no smoothing, no saturation boost
        description: "Intensity of the gouache effect. 0 leaves the image unchanged; \
                      1 applies full smoothing and maximum saturation boost.",
    }]);
    const PASSES: &'static [PassDef] = &[
        // Pass 1: horizontal Gaussian blur — begins the detail-flattening step.
        PassDef {
            label: "blur_h",
            wgsl_source: include_str!("gouache_blur_h.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("h"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        // Pass 2: vertical Gaussian blur — completes the 2D separable smooth.
        PassDef {
            label: "blur_v",
            wgsl_source: include_str!("gouache_blur_v.wgsl"),
            inputs: &[PassInput::Scratch("h")],
            output: PassOutput::Scratch("blurred"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        // Pass 3: blend source with smoothed result and boost saturation.
        PassDef {
            label: "color",
            wgsl_source: include_str!("gouache_color.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("blurred")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<GouacheParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_gouache_registry_entry_exists() {
        assert!(registry_by_id("gouache").is_some());
    }

    #[test]
    fn test_gouache_registry_metadata() {
        let reg = registry_by_id("gouache").unwrap();
        assert_eq!(reg.meta.display_name, "Gouache");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Intensity of the gouache effect. 0 leaves the image unchanged; \
                              1 applies full smoothing and maximum saturation boost.",
            }])
        );
        assert_eq!(
            reg.meta.passes.len(),
            3,
            "Gouache must have exactly 3 passes"
        );
    }

    #[test]
    fn test_gouache_make_uniform_known_value() {
        let reg = registry_by_id("gouache").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&GouacheParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_gouache_zero_strength_is_identity() {
        // At strength=0 the blur sigma is 0, so both blur passes copy pixels
        // unchanged, and the color pass reduces to no-op blending and no
        // saturation boost. The output must match the source within f16 rounding.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "gouache",
                values: vec![0.0],
            }],
        );
        for pixel in out.pixels() {
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
        }
    }

    #[test]
    fn test_gouache_alpha_preserved() {
        // Neither the blur passes nor the color pass should alter alpha.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "gouache",
                values: vec![0.8],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved at strength=0.8");
        }
    }

    #[test]
    fn test_gouache_positive_strength_boosts_saturation_on_chromatic_input() {
        // A chromatic input with R > G > B should have its dominant channel (R)
        // pushed further from luma after the color pass, confirming the saturation
        // boost is applied. A neutral (gray) pixel has nothing to boost, so we
        // compare against a colour pixel.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Strong red tint: R clearly above neutral, B clearly below.
        let img = make_solid_image(4, 4, 50000, 32767, 10000);

        let out_no_effect = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "gouache",
                values: vec![0.0],
            }],
        );
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "gouache",
                values: vec![1.0],
            }],
        );

        // With positive strength the R channel (above luma) should be pushed higher.
        let r_baseline = out_no_effect.get_pixel(0, 0)[0] as i32;
        let r_boosted = out_full.get_pixel(0, 0)[0] as i32;
        assert!(
            r_boosted >= r_baseline,
            "R channel should be pushed higher by saturation boost: baseline={r_baseline}, boosted={r_boosted}"
        );

        // B channel (below luma) should be pushed lower.
        let b_baseline = out_no_effect.get_pixel(0, 0)[2] as i32;
        let b_boosted = out_full.get_pixel(0, 0)[2] as i32;
        assert!(
            b_boosted <= b_baseline,
            "B channel should be pushed lower by saturation boost: baseline={b_baseline}, boosted={b_boosted}"
        );
    }

    #[test]
    fn test_gouache_smoothing_reduces_edge_contrast() {
        // A step image (left dark, right light) should have a softer transition at
        // high strength than at zero, because the blur flattens the edge.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 10000 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_zero = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "gouache",
                values: vec![0.0],
            }],
        );
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "gouache",
                values: vec![1.0],
            }],
        );

        // The channel difference across the boundary should be smaller at strength=1.
        let diff_zero =
            (out_zero.get_pixel(17, 8)[0] as i32 - out_zero.get_pixel(14, 8)[0] as i32).abs();
        let diff_full =
            (out_full.get_pixel(17, 8)[0] as i32 - out_full.get_pixel(14, 8)[0] as i32).abs();

        assert!(
            diff_full < diff_zero,
            "edge contrast must be reduced by smoothing: diff_full={diff_full}, diff_zero={diff_zero}"
        );
    }

    #[test]
    fn test_gouache_gray_image_unchanged_by_saturation_boost() {
        // A neutral gray image has no chrominance, so the saturation boost has
        // nothing to amplify. All output channels should remain equal (gray).
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "gouache",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - pixel[1] as i32).abs() <= 64,
                "R and G should remain equal on gray input: R={}, G={}",
                pixel[0],
                pixel[1]
            );
            assert!(
                (pixel[1] as i32 - pixel[2] as i32).abs() <= 64,
                "G and B should remain equal on gray input: G={}, B={}",
                pixel[1],
                pixel[2]
            );
        }
    }

    #[test]
    fn test_gouache_deterministic() {
        // Identical inputs and parameters must produce bit-identical output.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 16384, 8192);
        let transform = Transform {
            shader_id: "gouache",
            values: vec![0.6],
        };
        let out1 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            std::slice::from_ref(&transform),
        );
        let out2 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            std::slice::from_ref(&transform),
        );
        for (p1, p2) in out1.pixels().zip(out2.pixels()) {
            assert_eq!(p1, p2, "outputs must be pixel-identical across runs");
        }
    }

    #[test]
    fn test_gouache_chained_with_brightness() {
        // Verify Gouache chains correctly with another shader in the pipeline.
        // If the chain runs without panicking and produces non-zero output, the
        // bind-group wiring across shader boundaries is working correctly.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "gouache",
                    values: vec![0.5],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.1],
                },
            ],
        );
        // The output must be non-zero (pipeline produced meaningful data).
        let any_nonzero = out.pixels().any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0);
        assert!(any_nonzero, "chained output must contain non-zero pixels");
    }
}
