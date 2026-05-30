use std::f32::consts::TAU;

use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SwirlParams {
    /// Maximum rotation angle (radians) applied at the centre. 0.0 = identity.
    pub angle: f32,
    /// Distance from the centre (normalised half-diagonal units) where the
    /// rotation reaches zero. Effective range (0.0, 2.0].
    pub radius: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for SwirlParams {
    const ID: &'static str = "swirl";
    const DISPLAY_NAME: &'static str = "Swirl";
    const DESCRIPTION: &'static str = "Rotates pixels around the image centre by an angle that falls off with \
         distance, creating a swirl/spiral distortion.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Angle",
            min: -TAU,
            max: TAU,
            default: 0.0,
            description: "Maximum rotation at the centre in radians. 0.0 = no-op; positive \
                          = counter-clockwise; negative = clockwise.",
        },
        SliderDef {
            name: "Radius",
            min: 0.01,
            max: 2.0,
            default: 1.0,
            description: "Distance from the centre (in normalised half-diagonal units) at \
                          which the rotation reaches zero. Smaller values confine the swirl \
                          to a tighter region.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "swirl",
        wgsl_source: include_str!("swirl.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            angle: values[0],
            radius: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<SwirlParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_swirl_registry_entry_exists() {
        assert!(registry_by_id("swirl").is_some());
    }

    #[test]
    fn test_swirl_registry_metadata() {
        let reg = registry_by_id("swirl").unwrap();
        assert_eq!(reg.meta.display_name, "Swirl");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Angle",
                    min: -TAU,
                    max: TAU,
                    default: 0.0,
                    description: "Maximum rotation at the centre in radians. 0.0 = no-op; \
                                  positive = counter-clockwise; negative = clockwise.",
                },
                SliderDef {
                    name: "Radius",
                    min: 0.01,
                    max: 2.0,
                    default: 1.0,
                    description: "Distance from the centre (in normalised half-diagonal units) \
                                  at which the rotation reaches zero. Smaller values confine \
                                  the swirl to a tighter region.",
                },
            ])
        );
    }

    #[test]
    fn test_swirl_make_uniform_known_value() {
        let reg = registry_by_id("swirl").unwrap();
        let bytes = (reg.make_uniform)(&[1.5, 0.8]);
        let expected = bytemuck::bytes_of(&SwirlParams {
            angle: 1.5,
            radius: 0.8,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    /// At angle=0.0 (identity) the output pixels must match the input pixels for
    /// every pixel in a solid-colour image, verifying the no-op fast path.
    #[test]
    fn test_swirl_identity_at_zero_angle() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 40000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "swirl",
                values: vec![0.0, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 20000).abs() <= 64,
                "R: expected ~20000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 40000).abs() <= 64,
                "G: expected ~40000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 60000).abs() <= 64,
                "B: expected ~60000, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// A solid-colour image under any swirl angle must produce the same solid
    /// colour in every pixel (all source pixels are identical, so rotation
    /// cannot change the result), while leaving alpha intact.
    #[test]
    fn test_swirl_solid_image_unchanged_at_any_angle() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "swirl",
                values: vec![3.0, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 30000).abs() <= 64,
                "R: expected ~30000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 30000).abs() <= 64,
                "G: expected ~30000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 30000).abs() <= 64,
                "B: expected ~30000, got {}",
                pixel[2]
            );
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// The alpha channel must pass through unchanged regardless of swirl strength.
    #[test]
    fn test_swirl_alpha_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "swirl",
                values: vec![2.0, 1.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    /// The centre pixel of an odd-dimensioned image is at r=0, so the full
    /// swirl angle is applied there. For a solid-colour image the sampled
    /// source pixel is always the same colour regardless of rotation direction,
    /// so the centre pixel output colour must equal the input.
    #[test]
    fn test_swirl_centre_pixel_colour_preserved_in_solid_image() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(3, 3, 50000, 10000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "swirl",
                values: vec![std::f32::consts::PI, 1.0],
            }],
        );

        let p = out.get_pixel(1, 1);
        assert!(
            (p[0] as i32 - 50000).abs() <= 128,
            "centre R: expected ~50000, got {}",
            p[0]
        );
        assert!(
            (p[1] as i32 - 10000).abs() <= 128,
            "centre G: expected ~10000, got {}",
            p[1]
        );
        assert!(
            (p[2] as i32 - 30000).abs() <= 128,
            "centre B: expected ~30000, got {}",
            p[2]
        );
    }

    /// Chaining swirl with brightness must not panic and must leave the alpha
    /// channel intact across the full pipeline.
    #[test]
    fn test_swirl_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "swirl",
                    values: vec![1.0, 0.5],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 after chaining");
        }
    }

    /// Negative angle (clockwise) must produce a valid result without panicking.
    /// The out-of-bounds fill path uses opaque black (alpha=1.0 / u16=65535), so
    /// alpha must be 65535 for every output pixel regardless of whether the
    /// rotated UV mapped within the image or was filled.
    #[test]
    fn test_swirl_negative_angle_alpha_intact() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 40000, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "swirl",
                values: vec![-2.0, 1.0],
            }],
        );

        // Both in-bounds and out-of-bounds fill pixels must have alpha=65535.
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }
}
