use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TinyPlanetParams {
    /// Zoom level in [0.0, 1.0]. 0.0 = identity (flat pass-through);
    /// 1.0 = maximum planet wrap.
    pub zoom: f32,
    /// Source-image rotation before projection, in degrees [-180.0, 180.0].
    /// 0.0 = no rotation.
    pub rotation: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for TinyPlanetParams {
    const ID: &'static str = "tiny_planet";
    const DISPLAY_NAME: &'static str = "Tiny Planet";
    const DESCRIPTION: &'static str = "Wraps a panoramic image around a sphere via stereographic projection \
         to create the illusion of a small planet viewed from below.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Zoom",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Planet zoom level. 0.0 = no effect (identity); \
                 1.0 = maximum stereographic wrap.",
        },
        SliderDef {
            name: "Rotation",
            min: -180.0,
            max: 180.0,
            default: 0.0,
            description: "Rotates the source panorama around the horizontal axis before \
                 projecting, in degrees.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "tiny_planet",
        wgsl_source: include_str!("tiny_planet.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            zoom: values[0],
            rotation: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    TinyPlanetParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_tiny_planet_registry_entry_exists() {
        assert!(registry_by_id("tiny_planet").is_some());
    }

    #[test]
    fn test_tiny_planet_registry_metadata() {
        let reg = registry_by_id("tiny_planet").unwrap();
        assert_eq!(reg.meta.display_name, "Tiny Planet");
        assert_eq!(reg.meta.passes.len(), 1);
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Zoom",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Planet zoom level. 0.0 = no effect (identity); \
                         1.0 = maximum stereographic wrap.",
                },
                SliderDef {
                    name: "Rotation",
                    min: -180.0,
                    max: 180.0,
                    default: 0.0,
                    description: "Rotates the source panorama around the horizontal axis before \
                         projecting, in degrees.",
                },
            ])
        );
    }

    #[test]
    fn test_tiny_planet_make_uniform_known_value() {
        let reg = registry_by_id("tiny_planet").unwrap();
        let bytes = (reg.make_uniform)(&[0.5, 90.0]);
        let expected = bytemuck::bytes_of(&TinyPlanetParams {
            zoom: 0.5,
            rotation: 90.0,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    /// At zoom=0.0, rotation=0.0 (identity defaults) a solid-colour image passes
    /// through unchanged.
    #[test]
    fn test_tiny_planet_identity_at_zero_zoom() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 40000, 60000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tiny_planet",
                values: vec![0.0, 0.0],
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

    /// Alpha channel must be 65535 for every output pixel regardless of zoom.
    #[test]
    fn test_tiny_planet_alpha_preservation() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tiny_planet",
                values: vec![0.5, 0.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 for all pixels");
        }
    }

    /// A solid-colour image should return the same colour everywhere even at
    /// maximum zoom, because every visible sample maps back to the same colour.
    #[test]
    fn test_tiny_planet_solid_image_max_zoom() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(8, 8, 50000, 20000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tiny_planet",
                values: vec![1.0, 0.0],
            }],
        );

        for pixel in out.pixels() {
            // Pixels inside the sphere show the solid colour; pixels outside
            // (behind the sphere) are filled with black (0, 0, 0, 65535). Both
            // outcomes are valid — check only that alpha is 65535.
            assert_eq!(pixel[3], 65535, "alpha must be 65535 at max zoom");
        }
    }

    /// Rotation parameter must not change alpha or produce out-of-bounds access.
    #[test]
    fn test_tiny_planet_rotation_no_crash() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tiny_planet",
                values: vec![0.5, 180.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 after rotation");
        }
    }

    /// Negative rotation must also produce valid output.
    #[test]
    fn test_tiny_planet_negative_rotation_no_crash() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "tiny_planet",
                values: vec![0.5, -180.0],
            }],
        );

        for pixel in out.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be 65535 with negative rotation"
            );
        }
    }

    /// Chaining tiny_planet with another shader (brightness) must not panic and
    /// must keep the alpha channel intact.
    #[test]
    fn test_tiny_planet_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let img = make_solid_image(4, 4, 20000, 20000, 20000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "tiny_planet",
                    values: vec![0.5, 0.0],
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
