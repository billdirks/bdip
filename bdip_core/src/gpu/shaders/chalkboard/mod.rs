use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters shared across both Chalkboard passes.
///
/// Packs into 8 bytes; two padding floats bring the struct to 16 bytes to satisfy
/// WebGPU's uniform-buffer alignment requirement.
///
/// # Identity design
///
/// The spec requires that default parameter values produce a no-op transformation.
/// For Chalkboard the artistic effect (dark background + white chalk lines) cannot
/// be literally identity at any non-zero strength. The design follows the pattern
/// established by Pencil Sketch and Stained Glass: a `strength` blend parameter
/// defaults to `0.0`, which passes the source image through unchanged (identity),
/// while `chalk_boost` controls edge brightness when `strength` is non-zero.
/// At `strength = 0.0` the output equals the source regardless of other sliders.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChalkboardParams {
    /// Blend factor: 0.0 = source unchanged (identity), 1.0 = full chalkboard effect.
    pub strength: f32,
    /// Multiplier applied to raw Sobel edge magnitude before clamping.
    /// Higher values make faint edges appear as bright chalk lines. Range [0.1, 10.0].
    pub chalk_boost: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for ChalkboardParams {
    const ID: &'static str = "chalkboard";
    const DISPLAY_NAME: &'static str = "Chalkboard";
    const DESCRIPTION: &'static str = "Renders the image as a chalk drawing on a dark chalkboard \
         using inverted Sobel edge detection and procedural chalk-grain texture.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full chalkboard \
                          effect (1.0). The identity value is 0.0.",
        },
        SliderDef {
            name: "Chalk Boost",
            min: 0.1,
            max: 10.0,
            default: 3.0,
            description: "Sensitivity of edge detection. Higher values brighten faint edges \
                          into visible chalk lines.",
        },
    ]);

    // Two-pass pipeline:
    //   Pass 1 — edges: Sobel edge detection → inverted chalk lines on dark background
    //                   stored in a scratch texture.
    //   Pass 2 — grain: add procedural chalk-grain noise, blend with source via strength.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "edges",
            wgsl_source: include_str!("chalkboard_edges.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("edges"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "grain",
            wgsl_source: include_str!("chalkboard_grain.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("edges")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            chalk_boost: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    ChalkboardParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_chalkboard_registry_entry_exists() {
        assert!(registry_by_id("chalkboard").is_some());
    }

    #[test]
    fn test_chalkboard_registry_metadata() {
        let reg = registry_by_id("chalkboard").unwrap();
        assert_eq!(reg.meta.display_name, "Chalkboard");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full chalkboard \
                                  effect (1.0). The identity value is 0.0.",
                },
                SliderDef {
                    name: "Chalk Boost",
                    min: 0.1,
                    max: 10.0,
                    default: 3.0,
                    description: "Sensitivity of edge detection. Higher values brighten faint edges \
                                  into visible chalk lines.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            2,
            "Chalkboard must have exactly 2 passes"
        );
    }

    #[test]
    fn test_chalkboard_make_uniform_known_value() {
        let reg = registry_by_id("chalkboard").unwrap();
        let bytes = (reg.make_uniform)(&[0.75, 4.0]);
        let expected = bytemuck::bytes_of(&ChalkboardParams {
            strength: 0.75,
            chalk_boost: 4.0,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 the grain pass reduces to mix(src, chalk, 0.0) = src.
    /// The output must equal the source regardless of chalk_boost.
    #[test]
    fn test_chalkboard_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "chalkboard",
                values: vec![0.0, 3.0],
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
    fn test_chalkboard_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "chalkboard",
                values: vec![1.0, 3.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// A uniform (solid-colour) image has no edges; the Sobel magnitude is zero
    /// everywhere. At full strength the output should be near the dark chalkboard
    /// background (dark green/black). With grain the mean brightness must be low.
    #[test]
    fn test_chalkboard_solid_image_produces_dark_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Mid-gray solid image — Sobel returns zero on a constant input.
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "chalkboard",
                values: vec![1.0, 3.0],
            }],
        );
        // Background colour is dark green (~0.07, 0.15, 0.07 linear).
        // In u16 that is approximately R≈4588, G≈9830, B≈4588.
        // Grain adds up to ±grain_scale (≈0.04 linear ≈ 2621 u16).
        // So R must stay well below 16384 (0.25 linear), and G below 32768.
        let mean_r: f64 = out.pixels().map(|p| p[0] as f64).sum::<f64>() / (16.0 * 16.0);
        let mean_g: f64 = out.pixels().map(|p| p[1] as f64).sum::<f64>() / (16.0 * 16.0);
        assert!(
            mean_r < 16000.0,
            "R mean on solid image should be near dark background, got {mean_r:.0}"
        );
        assert!(
            mean_g < 20000.0,
            "G mean on solid image should be near dark background, got {mean_g:.0}"
        );
    }

    /// On an image with a sharp edge the pixels near the edge boundary should be
    /// bright (near white/chalk) compared to the flat chalkboard background areas.
    #[test]
    fn test_chalkboard_edge_pixels_brighter_than_background() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32×16 step image: left half dark, right half bright.
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 5000 } else { 60000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "chalkboard",
                values: vec![1.0, 3.0],
            }],
        );

        // Pixel at the edge boundary (x=15 or x=16) should be brighter than a
        // pixel well inside a flat region (x=2, far from the step).
        let edge_pixel = out.get_pixel(15, 8)[0] as i32;
        let flat_pixel = out.get_pixel(2, 8)[0] as i32;
        assert!(
            edge_pixel > flat_pixel,
            "edge pixel (x=15) must be brighter (chalk line) than flat background (x=2): \
             edge={edge_pixel}, flat={flat_pixel}"
        );
    }

    /// Higher chalk_boost must amplify faint edges into brighter chalk lines, making
    /// the overall output brighter (more chalk) compared to lower boost.
    #[test]
    fn test_chalkboard_higher_chalk_boost_increases_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Shallow gradient: produces a weak, uniform Sobel signal.
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v = (x * 2000) as u16;
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_low = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "chalkboard",
                values: vec![1.0, 1.0],
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "chalkboard",
                values: vec![1.0, 8.0],
            }],
        );

        // Higher boost means more edges become bright chalk lines → higher mean brightness.
        let mean_low: f64 = out_low.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 16.0);
        let mean_high: f64 = out_high.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 16.0);
        assert!(
            mean_high > mean_low,
            "higher chalk_boost must produce a brighter (higher mean R) output: \
             low={mean_low:.0}, high={mean_high:.0}"
        );
    }

    /// Chaining chalkboard with brightness must not panic and must preserve alpha.
    #[test]
    fn test_chalkboard_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "chalkboard",
                    values: vec![0.5, 3.0],
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

    /// Running Chalkboard twice with identical inputs must produce bit-identical output.
    #[test]
    fn test_chalkboard_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "chalkboard",
            values: vec![0.8, 3.0],
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
