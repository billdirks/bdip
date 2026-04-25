use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClarityParams {
    pub amount: f32,        // u_Clarity ∈ [-1.0, 1.0]
    pub _padding: [f32; 3], // pad to 16 bytes
}

impl TransformShader for ClarityParams {
    const ID: &'static str = "clarity";
    const DISPLAY_NAME: &'static str = "Clarity";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Amount",
        min: -1.0,
        max: 1.0,
        default: 0.0,
    }]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "blur_h",
            wgsl_source: include_str!("blur_h.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("h"),
            output_scale: PassScale::Full,
        },
        PassDef {
            label: "blur_v",
            wgsl_source: include_str!("blur_v.wgsl"),
            inputs: &[PassInput::Scratch("h")],
            output: PassOutput::Scratch("v"),
            output_scale: PassScale::Full,
        },
        PassDef {
            label: "combine",
            wgsl_source: include_str!("combine.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("v")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            amount: values[0],
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ClarityParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_clarity_registry_entry_exists() {
        assert!(registry_by_id("clarity").is_some());
    }

    #[test]
    fn test_clarity_registry_metadata() {
        let reg = registry_by_id("clarity").unwrap();
        assert_eq!(reg.meta.display_name, "Clarity");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Amount",
                min: -1.0,
                max: 1.0,
                default: 0.0,
            }])
        );
        assert_eq!(
            reg.meta.passes.len(),
            3,
            "Clarity must have exactly 3 passes"
        );
    }

    #[test]
    fn test_clarity_make_uniform_known_value() {
        let reg = registry_by_id("clarity").unwrap();
        let bytes = (reg.make_uniform)(&[0.5]);
        let expected = bytemuck::bytes_of(&ClarityParams {
            amount: 0.5,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_clarity_zero_amount_is_identity() {
        // At amount=0 the combine formula reduces to C_in + 0 = C_in.
        // The blur roundtrip introduces tiny f16 rounding, so ±64 u16 is the
        // established tolerance for GPU-roundtrip tests in this codebase.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // 32767 ≈ 0.5 sRGB ≈ mid-gray — sits at the midtone-weight peak.
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "clarity",
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
                (pixel[1] as i32 - 32767).abs() <= 64,
                "G: expected ~32767, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 32767).abs() <= 64,
                "B: expected ~32767, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn test_clarity_positive_amount_increases_contrast_on_edge() {
        // A step image (left half dark-gray, right half light-gray) has a strong
        // horizontal high-pass signal near the boundary. With amount=0.5, the
        // combine pass should push pixels on each side of the edge further from
        // the overall mean than they were at amount=0.0.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32×16 step: left half at 20000, right half at 45000 (sRGB u16).
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 20000 } else { 45000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_0 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "clarity",
                values: vec![0.0],
            }],
        );
        let out_pos = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "clarity",
                values: vec![0.5],
            }],
        );

        // Compare a pixel just inside the dark side near the edge (x=14, y=8)
        // and one just inside the bright side (x=17, y=8).
        let dark_0 = out_0.get_pixel(14, 8)[0] as i32;
        let dark_pos = out_pos.get_pixel(14, 8)[0] as i32;
        let bright_0 = out_0.get_pixel(17, 8)[0] as i32;
        let bright_pos = out_pos.get_pixel(17, 8)[0] as i32;

        // Dark side: positive clarity pulls dark edge pixels darker.
        assert!(
            dark_pos < dark_0,
            "dark-side pixel should be darker at amount=0.5: {} vs {}",
            dark_pos,
            dark_0
        );
        // Bright side: positive clarity pushes bright edge pixels brighter.
        assert!(
            bright_pos > bright_0,
            "bright-side pixel should be brighter at amount=0.5: {} vs {}",
            bright_pos,
            bright_0
        );
    }

    #[test]
    fn test_clarity_negative_amount_softens_edge() {
        // Negative amount inverts the high-pass addition, blending the blurred
        // signal back in and softening edges. The inter-band difference must be
        // smaller at amount=-0.5 than at amount=0.0.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 20000 } else { 45000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_0 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "clarity",
                values: vec![0.0],
            }],
        );
        let out_neg = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "clarity",
                values: vec![-0.5],
            }],
        );

        // At the boundary (x=14..17) the inter-pixel difference should be smaller
        // with negative amount (softer transition).
        let diff_0 = (out_0.get_pixel(17, 8)[0] as i32 - out_0.get_pixel(14, 8)[0] as i32).abs();
        let diff_neg =
            (out_neg.get_pixel(17, 8)[0] as i32 - out_neg.get_pixel(14, 8)[0] as i32).abs();

        assert!(
            diff_neg < diff_0,
            "edge transition must be softer at amount=-0.5: diff_neg={diff_neg}, diff_0={diff_0}"
        );
    }

    #[test]
    fn test_clarity_alpha_preserved() {
        // The combine pass copies alpha from the source; neither blur pass must
        // alter the alpha channel.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "clarity",
                values: vec![0.5],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    #[test]
    fn test_clarity_deterministic() {
        // Running Clarity twice with identical inputs must produce bit-identical output.
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "clarity",
            values: vec![0.5],
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
}
