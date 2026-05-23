use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Burned Edges shader.
///
/// Four floats fill one 16-byte WebGPU uniform slot exactly — no padding needed.
///
/// # Identity design
///
/// `intensity` defaults to 0.0, which blends the burn overlay at weight 0 —
/// a pure passthrough regardless of the other slider values.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BurnedEdgesParams {
    /// Blend weight of the burn overlay. 0.0 = identity (no effect), 1.0 = full burn.
    pub intensity: f32,
    /// How far the burn extends inward from each edge, in normalised image
    /// coordinates. 0.0 = no burn, 0.5 = burn reaches halfway to center.
    pub radius: f32,
    /// Width of the transition zone between unburned and fully burned regions,
    /// expressed as a fraction of `radius`. 0.0 = hard edge, 1.0 = fully feathered.
    pub softness: f32,
    /// Warm charred tint amount. 0.0 = pure black char, 1.0 = warm brown/amber char.
    pub tint: f32,
}

impl TransformShader for BurnedEdgesParams {
    const ID: &'static str = "burned_edges";
    const DISPLAY_NAME: &'static str = "Burned Edges";
    const DESCRIPTION: &'static str = "Darkens and chars the image edges, simulating a photograph burned or scorched around \
         its perimeter with an organic, uneven flame texture.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Intensity",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend strength of the burn overlay. \
                 0.0 leaves the image completely unchanged (identity).",
        },
        SliderDef {
            name: "Radius",
            min: 0.0,
            max: 0.5,
            default: 0.25,
            description: "How far the burn extends inward from each edge in normalised \
                 image coordinates. 0.5 reaches the center of the image.",
        },
        SliderDef {
            name: "Softness",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Width of the feathered transition from unburned to fully burned, \
                 as a fraction of the radius. 0.0 = hard char line, 1.0 = fully gradual.",
        },
        SliderDef {
            name: "Tint",
            min: 0.0,
            max: 1.0,
            default: 0.5,
            description: "Color of the charred region. 0.0 = pure black, \
                 1.0 = warm brown/amber char tone.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "burned_edges",
        wgsl_source: include_str!("burned_edges.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            intensity: values[0],
            radius: values[1],
            softness: values[2],
            tint: values[3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    BurnedEdgesParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ── Registry / metadata ──────────────────────────────────────────────────

    #[test]
    fn test_burned_edges_registry_entry_exists() {
        assert!(registry_by_id("burned_edges").is_some());
    }

    #[test]
    fn test_burned_edges_registry_metadata() {
        let reg = registry_by_id("burned_edges").unwrap();
        assert_eq!(reg.meta.display_name, "Burned Edges");
        assert_eq!(reg.meta.passes.len(), 1, "must have exactly 1 pass");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Intensity",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend strength of the burn overlay. \
                         0.0 leaves the image completely unchanged (identity).",
                },
                SliderDef {
                    name: "Radius",
                    min: 0.0,
                    max: 0.5,
                    default: 0.25,
                    description: "How far the burn extends inward from each edge in normalised \
                         image coordinates. 0.5 reaches the center of the image.",
                },
                SliderDef {
                    name: "Softness",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Width of the feathered transition from unburned to fully burned, \
                         as a fraction of the radius. 0.0 = hard char line, 1.0 = fully gradual.",
                },
                SliderDef {
                    name: "Tint",
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                    description: "Color of the charred region. 0.0 = pure black, \
                         1.0 = warm brown/amber char tone.",
                },
            ])
        );
    }

    #[test]
    fn test_burned_edges_make_uniform_known_value() {
        let reg = registry_by_id("burned_edges").unwrap();
        let bytes = (reg.make_uniform)(&[0.8, 0.3, 0.6, 0.4]);
        let expected = bytemuck::bytes_of(&BurnedEdgesParams {
            intensity: 0.8,
            radius: 0.3,
            softness: 0.6,
            tint: 0.4,
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ──────────────────────────────────────────────────

    /// intensity=0.0 is the identity: output must equal the source pixel-for-pixel
    /// regardless of other parameter values.
    #[test]
    fn test_burned_edges_zero_intensity_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 25000, 18000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "burned_edges",
                values: vec![0.0, 0.25, 0.5, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 25000).abs() <= 64,
                "R: expected ~25000 at intensity=0, got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 18000).abs() <= 64,
                "G: expected ~18000 at intensity=0, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 40000).abs() <= 64,
                "B: expected ~40000 at intensity=0, got {}",
                pixel[2]
            );
        }
    }

    /// Alpha channel must pass through unchanged regardless of intensity.
    #[test]
    fn test_burned_edges_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(32, 32, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "burned_edges",
                values: vec![1.0, 0.5, 0.5, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha channel must be unchanged");
        }
    }

    /// Full intensity with a large radius must darken the corner pixels of the image.
    /// Corner pixels have the smallest edge distance, so they receive the most burn.
    #[test]
    fn test_burned_edges_full_intensity_darkens_corners() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // 4×4 image: corner pixels are at distance 1/8 from edges, which is well
        // inside a radius of 0.5.
        let img = make_solid_image(4, 4, 50000, 50000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "burned_edges",
                // Large radius covers most of the image; hard edge to avoid noise masking the result.
                values: vec![1.0, 0.5, 0.0, 0.0],
            }],
        );
        // The corner pixel (0,0) must be significantly darker than the input.
        let corner = out.get_pixel(0, 0);
        assert!(
            (corner[0] as i32) < 10000,
            "corner pixel R should be significantly darkened; got {}",
            corner[0]
        );
    }

    /// With radius=0.0 the burn zone has zero extent, so output must equal the source
    /// (modulo noise displacement, which also has zero range when radius=0).
    #[test]
    fn test_burned_edges_zero_radius_no_burn() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32000, 32000, 32000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "burned_edges",
                values: vec![1.0, 0.0, 0.5, 0.5],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 32000).abs() <= 256,
                "R: expected ~32000 with zero radius, got {}",
                pixel[0]
            );
        }
    }

    /// tint=0.0 (pure black char) must produce darker output on a bright image
    /// than tint=1.0 (warm brown char), since the char color at tint=1 is slightly
    /// brighter than pure black.
    #[test]
    fn test_burned_edges_pure_black_tint_darker_than_warm_tint() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a 2×2 image: every pixel is at the corner, deeply inside the burn zone.
        let img = make_solid_image(2, 2, 65535, 65535, 65535);

        let out_black = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "burned_edges",
                values: vec![1.0, 0.5, 0.0, 0.0], // tint=0 (pure black)
            }],
        );
        let out_warm = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "burned_edges",
                values: vec![1.0, 0.5, 0.0, 1.0], // tint=1 (warm brown)
            }],
        );

        // Sum of red channel across all pixels: warm char is slightly brighter.
        let sum_black: u32 = out_black.pixels().map(|p| p[0] as u32).sum();
        let sum_warm: u32 = out_warm.pixels().map(|p| p[0] as u32).sum();
        assert!(
            sum_warm >= sum_black,
            "warm tint (sum={sum_warm}) should produce equal or brighter output than \
             pure black tint (sum={sum_black})"
        );
    }

    /// Chaining with the brightness identity must not corrupt the output.
    #[test]
    fn test_burned_edges_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "burned_edges",
                    values: vec![0.5, 0.25, 0.5, 0.5],
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.0],
                },
            ],
        );
        // Alpha must survive the chain unchanged.
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be 65535 after chaining");
        }
    }

    /// Two runs with identical inputs must produce bit-identical outputs
    /// (the procedural noise is deterministic given stable pixel coordinates).
    #[test]
    fn test_burned_edges_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "burned_edges",
            values: vec![0.8, 0.3, 0.5, 0.6],
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
