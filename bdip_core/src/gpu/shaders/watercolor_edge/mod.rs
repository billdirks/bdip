use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters shared across both Watercolor Edge passes.
///
/// The one meaningful field packs into 4 bytes; three padding floats bring the
/// struct to 16 bytes to satisfy WebGPU's uniform alignment requirement.
///
/// # Identity design
///
/// At `strength = 0.0` the composite pass reduces to `src * 1.0 = src`, which is
/// a true identity — the image passes through unchanged. This satisfies the
/// requirement that default values produce a no-op transformation.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WatercolorEdgeParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full dark-edge effect.
    pub strength: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for WatercolorEdgeParams {
    const ID: &'static str = "watercolor_edge";
    const DISPLAY_NAME: &'static str = "Watercolor Edge";
    const DESCRIPTION: &'static str = "Darkens edges using Sobel detection and dark-color multiplication, \
         simulating the characteristic dark outlines of watercolor paintings.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Strength",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Controls how dark and prominent the detected edges appear. \
                      0.0 leaves the image unchanged (identity); 1.0 applies maximum \
                      edge darkening.",
    }]);

    // Two-pass pipeline:
    //   Pass 1 — edges:     Sobel edge detection on the luma channel → scratch texture.
    //   Pass 2 — composite: multiplies the edge dark mask into the source image.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "edges",
            wgsl_source: include_str!("watercolor_edge_edges.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("edges"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "composite",
            wgsl_source: include_str!("watercolor_edge_composite.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("edges")],
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

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    WatercolorEdgeParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_watercolor_edge_registry_entry_exists() {
        assert!(registry_by_id("watercolor_edge").is_some());
    }

    #[test]
    fn test_watercolor_edge_registry_metadata() {
        let reg = registry_by_id("watercolor_edge").unwrap();
        assert_eq!(reg.meta.display_name, "Watercolor Edge");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[SliderDef {
                name: "Strength",
                min: 0.0,
                max: 1.0,
                default: 0.0,
                description: "Controls how dark and prominent the detected edges appear. \
                              0.0 leaves the image unchanged (identity); 1.0 applies maximum \
                              edge darkening.",
            }])
        );
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "Watercolor Edge must have exactly 2 passes"
        );
    }

    #[test]
    fn test_watercolor_edge_make_uniform_known_value() {
        let reg = registry_by_id("watercolor_edge").unwrap();
        let bytes = (reg.make_uniform)(&[0.75]);
        let expected = bytemuck::bytes_of(&WatercolorEdgeParams {
            strength: 0.75,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 the composite pass reduces to src * 1.0 = src.
    /// The output must equal the source within GPU rounding tolerance.
    #[test]
    fn test_watercolor_edge_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "watercolor_edge",
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
                (pixel[1] as i32 - 20000).abs() <= 64,
                "G: expected ~20000, got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 50000).abs() <= 64,
                "B: expected ~50000, got {}",
                pixel[2]
            );
        }
    }

    /// Alpha channel must pass through unchanged at any strength value.
    #[test]
    fn test_watercolor_edge_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "watercolor_edge",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// A solid-colour image has no edges (Sobel = 0). The dark-mask is 1.0
    /// everywhere, so the output must equal the source at any strength.
    #[test]
    fn test_watercolor_edge_solid_image_unchanged() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 40000, 40000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "watercolor_edge",
                values: vec![1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 40000).abs() <= 64,
                "solid image must be unchanged by edge darkening: got {}",
                pixel[0]
            );
        }
    }

    /// On an image with a sharp edge, pixels near the boundary must be darker
    /// than pixels well inside a flat region when strength > 0.
    #[test]
    fn test_watercolor_edge_darkens_edge_pixels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32×16 step image: left half at 30000, right half at 55000.
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 30000 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "watercolor_edge",
                values: vec![1.0],
            }],
        );

        // A pixel at the edge boundary (x=15) should be darker than a pixel
        // well inside the flat region (x=2), since Sobel is near-zero away from edges.
        let edge_pixel = out.get_pixel(15, 8)[0] as i32;
        let flat_pixel = out.get_pixel(2, 8)[0] as i32;
        assert!(
            edge_pixel < flat_pixel,
            "edge pixel (x=15) must be darker than flat-region pixel (x=2): \
             edge={edge_pixel}, flat={flat_pixel}"
        );
    }

    /// Greater strength must produce darker edges on an image with a sharp boundary.
    #[test]
    fn test_watercolor_edge_higher_strength_darker_edges() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 30000 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "watercolor_edge",
                values: vec![0.3],
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "watercolor_edge",
                values: vec![1.0],
            }],
        );

        // The edge pixel at x=15 must be darker at higher strength.
        let low_edge = out_low.get_pixel(15, 8)[0] as i32;
        let high_edge = out_high.get_pixel(15, 8)[0] as i32;
        assert!(
            high_edge < low_edge,
            "higher strength must produce a darker edge: low={low_edge}, high={high_edge}"
        );
    }

    /// The effect must preserve original colors in flat (no-edge) regions.
    /// At strength=1.0 on a step image, flat pixels far from the edge are
    /// unaffected by the dark mask (mask ≈ 1.0 there).
    #[test]
    fn test_watercolor_edge_flat_region_color_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 30000 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "watercolor_edge",
                values: vec![1.0],
            }],
        );

        // Pixel well inside the flat dark region (x=2) must be close to the original.
        let flat = out.get_pixel(2, 8)[0] as i32;
        assert!(
            (flat - 30000).abs() <= 200,
            "flat-region pixel must be close to original 30000, got {flat}"
        );
    }

    /// Chaining watercolor_edge with brightness must not panic and must preserve alpha.
    #[test]
    fn test_watercolor_edge_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "watercolor_edge",
                    values: vec![0.5],
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

    /// Running Watercolor Edge twice with identical inputs must produce bit-identical output.
    #[test]
    fn test_watercolor_edge_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "watercolor_edge",
            values: vec![0.8],
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
