use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Magnifying Glass UV distortion effect.
///
/// The uniform layout is:
///   - zoom:     zoom factor inside the lens circle (1.0 = identity / no magnification)
///   - radius:   lens circle radius as a fraction of the shorter image dimension [0.0, 1.0]
///   - center_x: horizontal centre of the lens [0.0, 1.0]; 0.5 = image centre
///   - center_y: vertical centre of the lens [0.0, 1.0]; 0.5 = image centre
///
/// All four fields fill the 16-byte WebGPU uniform minimum without padding.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MagnifyingGlassParams {
    pub zoom: f32,
    pub radius: f32,
    pub center_x: f32,
    pub center_y: f32,
}

impl TransformShader for MagnifyingGlassParams {
    const ID: &'static str = "magnifying_glass";
    const DISPLAY_NAME: &'static str = "Magnifying Glass";
    const DESCRIPTION: &'static str =
        "Magnifies a circular region of the image using UV coordinate scaling.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Zoom",
            min: 1.0,
            max: 4.0,
            default: 1.0,
            description: "Magnification factor inside the lens. 1.0 = no-op; higher values \
                          zoom in further.",
        },
        SliderDef {
            name: "Radius",
            min: 0.0,
            max: 1.0,
            default: 0.25,
            description: "Radius of the magnified circle as a fraction of the shorter image \
                          dimension. 0.0 = no visible lens.",
        },
        SliderDef {
            name: "Center X",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Horizontal centre of the magnified circle. 0.0 = left edge, \
                          1.0 = right edge.",
        },
        SliderDef {
            name: "Center Y",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Vertical centre of the magnified circle. 0.0 = top edge, \
                          1.0 = bottom edge.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "magnifying_glass",
        wgsl_source: include_str!("magnifying_glass.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            zoom: values[0],
            radius: values[1],
            center_x: values[2],
            center_y: values[3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    MagnifyingGlassParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // -------------------------------------------------------------------------
    // Registry tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_magnifying_glass_registry_entry_exists() {
        assert!(registry_by_id("magnifying_glass").is_some());
    }

    #[test]
    fn test_magnifying_glass_registry_metadata() {
        let reg = registry_by_id("magnifying_glass").unwrap();
        assert_eq!(reg.meta.display_name, "Magnifying Glass");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Zoom",
                    min: 1.0,
                    max: 4.0,
                    default: 1.0,
                    description: "Magnification factor inside the lens. 1.0 = no-op; higher \
                                  values zoom in further.",
                },
                SliderDef {
                    name: "Radius",
                    min: 0.0,
                    max: 1.0,
                    default: 0.25,
                    description: "Radius of the magnified circle as a fraction of the shorter \
                                  image dimension. 0.0 = no visible lens.",
                },
                SliderDef {
                    name: "Center X",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Horizontal centre of the magnified circle. 0.0 = left edge, \
                                  1.0 = right edge.",
                },
                SliderDef {
                    name: "Center Y",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Vertical centre of the magnified circle. 0.0 = top edge, \
                                  1.0 = bottom edge.",
                },
            ])
        );
    }

    #[test]
    fn test_magnifying_glass_make_uniform_known_value() {
        let reg = registry_by_id("magnifying_glass").unwrap();
        let bytes = (reg.make_uniform)(&[2.0, 0.3, 0.5, 0.5]);
        let expected = bytemuck::bytes_of(&MagnifyingGlassParams {
            zoom: 2.0,
            radius: 0.3,
            center_x: 0.5,
            center_y: 0.5,
        });
        assert_eq!(bytes, expected);
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip tests
    // -------------------------------------------------------------------------

    /// At zoom=1.0 (identity), every pixel must be unchanged regardless of radius
    /// or centre position, because scaling by 1.0 maps each UV back to itself.
    #[test]
    fn test_magnifying_glass_identity_at_zoom_one() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 40000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "magnifying_glass",
                values: vec![1.0, 0.5, 0.5, 0.5],
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

    /// A solid-colour image under any zoom factor must produce the same colour
    /// everywhere, because every source pixel is identical so the UV mapping
    /// cannot change the result.
    #[test]
    fn test_magnifying_glass_solid_image_unchanged_at_any_zoom() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "magnifying_glass",
                values: vec![3.0, 0.5, 0.5, 0.5],
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

    /// Alpha must pass through unchanged at maximum zoom strength.
    #[test]
    fn test_magnifying_glass_alpha_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "magnifying_glass",
                values: vec![4.0, 0.5, 0.5, 0.5],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    /// At radius=0.0 the lens circle has zero size, so no pixel falls inside it
    /// and the entire image is passed through unchanged.
    #[test]
    fn test_magnifying_glass_zero_radius_is_passthrough() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 15000, 35000, 55000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "magnifying_glass",
                values: vec![4.0, 0.0, 0.5, 0.5],
            }],
        );

        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 15000).abs() <= 64,
                "R: expected ~15000, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 35000).abs() <= 64,
                "G: expected ~35000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 55000).abs() <= 64,
                "B: expected ~55000, got {}",
                pixel[2]
            );
        }
    }

    /// The pixel at the exact lens centre is at distance 0 from the centre, so
    /// the scaled UV maps back to the centre position itself — the centre pixel
    /// must equal the input regardless of zoom or radius.
    #[test]
    fn test_magnifying_glass_centre_pixel_unchanged() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 3×3 so pixel (1,1) is the exact centre (centre UV = 0.5).
        let img = make_solid_image(3, 3, 50000, 10000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "magnifying_glass",
                values: vec![4.0, 0.5, 0.5, 0.5],
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

    /// Chaining the magnifying glass with brightness must not panic and must
    /// leave the alpha channel intact across the full pipeline.
    #[test]
    fn test_magnifying_glass_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "magnifying_glass",
                    values: vec![2.0, 0.4, 0.5, 0.5],
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
}
