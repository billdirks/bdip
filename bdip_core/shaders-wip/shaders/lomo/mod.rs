use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LomoParams {
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for LomoParams {
    const ID: &'static str = "lomo";
    const DISPLAY_NAME: &'static str = "Lomo";
    const DESCRIPTION: &'static str =
        "Simulates the Lomography camera aesthetic: vivid colors with a strong radial vignette.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Overall effect intensity. 0 is an identity pass-through; 1 applies full lomo look.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "lomo",
        wgsl_source: include_str!("lomo.wgsl"),
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

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<LomoParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_lomo_registry_entry_exists() {
        assert!(registry_by_id("lomo").is_some());
    }

    #[test]
    fn test_lomo_registry_metadata() {
        let reg = registry_by_id("lomo").unwrap();
        assert_eq!(reg.meta.display_name, "Lomo");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Overall effect intensity. 0 is an identity pass-through; 1 applies full lomo look.",
            }])
        );
    }

    #[test]
    fn test_lomo_make_uniform_known_value() {
        let reg = registry_by_id("lomo").unwrap();
        let bytes = (reg.make_uniform)(&[0.7]);
        let expected = bytemuck::bytes_of(&LomoParams {
            strength: 0.7,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    /// strength=0 must be a perfect pass-through (identity).
    #[test]
    fn test_lomo_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Colorful, non-neutral pixel so any saturation change would be visible.
        let img = make_solid_image(2, 2, 32767, 16384, 8192);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "lomo",
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

    /// Full strength on a neutral gray center pixel must preserve luminance
    /// (saturation boost of a gray has no effect) and must not apply vignette
    /// when the pixel is exactly at the image center.
    ///
    /// Note: A 2×2 image has no single-pixel center; the closest UV to 0.5 is
    /// (0.25, 0.25) and (0.75, 0.75). The vignette distance at those points is
    /// ~0.354, which is just above the 0.35 smoothstep start — so very slight
    /// dimming occurs. The test uses a loose tolerance to accommodate this.
    #[test]
    fn test_lomo_full_strength_center_pixels_are_bright() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Neutral gray — saturation boost of a gray is identity.
        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "lomo",
                values: vec![1.0],
            }],
        );

        // Center pixels at ~0.354 distance — inside vignette falloff start (0.35).
        // Allow up to ~5 % dimming (≈3276 counts) for the marginal smoothstep ramp.
        for pixel in out.pixels() {
            assert!(
                pixel[0] > 30000,
                "R: center pixel should remain bright, got {}",
                pixel[0]
            );
        }
    }

    /// Full strength on a colored image must boost saturation: the dominant
    /// channel (R) must increase relative to the identity output, since it lies
    /// above the Rec.709 luminance.
    #[test]
    fn test_lomo_full_strength_increases_saturation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Warm orange-ish color (R > G > B). In a 2×2 image, pixels are near the
        // center so the vignette contributes little; the saturation boost dominates.
        let img = make_solid_image(2, 2, 40000, 20000, 10000);

        let identity = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "lomo",
                values: vec![0.0],
            }],
        );
        let lomo = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "lomo",
                values: vec![1.0],
            }],
        );

        // With full saturation boost the dominant channel R must be higher than the
        // identity result; the weaker channels B must be lower.
        for (id_px, lomo_px) in identity.pixels().zip(lomo.pixels()) {
            assert!(
                lomo_px[0] >= id_px[0],
                "R should increase with lomo saturation boost: identity={}, lomo={}",
                id_px[0],
                lomo_px[0]
            );
            assert!(
                lomo_px[2] <= id_px[2],
                "B should decrease with lomo saturation boost: identity={}, lomo={}",
                id_px[2],
                lomo_px[2]
            );
        }
    }

    /// Alpha must be unchanged regardless of strength value.
    #[test]
    fn test_lomo_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "lomo",
                values: vec![1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// Lomo chained with an identity brightness transform must produce the same
    /// result as lomo alone, verifying the output texture is passed correctly
    /// through the pipeline.
    #[test]
    fn test_lomo_chained_with_brightness_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(2, 2, 32767, 16384, 8192);

        let lomo_only = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "lomo",
                values: vec![0.5],
            }],
        );
        let chained = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "lomo",
                    values: vec![0.5],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );

        for (a, b) in lomo_only.pixels().zip(chained.pixels()) {
            assert!(
                (a[0] as i32 - b[0] as i32).abs() <= 64,
                "R mismatch: lomo_only={}, chained={}",
                a[0],
                b[0]
            );
            assert!(
                (a[1] as i32 - b[1] as i32).abs() <= 64,
                "G mismatch: lomo_only={}, chained={}",
                a[1],
                b[1]
            );
            assert!(
                (a[2] as i32 - b[2] as i32).abs() <= 64,
                "B mismatch: lomo_only={}, chained={}",
                a[2],
                b[2]
            );
        }
    }
}
