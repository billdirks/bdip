use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Line Art shader.
///
/// Two meaningful fields pack into 8 bytes; a `vec2<f32>` pad brings the
/// struct to 16 bytes to satisfy WebGPU's uniform alignment requirement.
///
/// # Identity design
///
/// The spec requires that default parameter values produce a no-op transformation.
/// A pure line-art pass cannot be an identity at any `threshold` value — it always
/// converts the image to edge lines on white. The `strength` blend parameter solves
/// this: at `strength = 0.0` the shader outputs `mix(src, line_art, 0.0) = src`,
/// which is an exact identity regardless of `threshold`. This matches the pattern
/// used by Pencil Sketch and other artistic effects.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineArtParams {
    /// Sensitivity multiplier applied to raw Sobel magnitude before clamping.
    /// Higher values make faint edges visible as dark lines. Range [0.1, 10.0].
    pub threshold: f32,
    /// Blend weight: 0.0 = source unchanged (identity), 1.0 = full line-art effect.
    pub strength: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for LineArtParams {
    const ID: &'static str = "line_art";
    const DISPLAY_NAME: &'static str = "Line Art";
    const DESCRIPTION: &'static str =
        "Converts the image to dark edge lines on a white background using Sobel edge detection.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Threshold",
            min: 0.1,
            max: 10.0,
            default: 2.0,
            description: "Sensitivity of edge detection. Higher values make faint edges \
                          more visible as dark lines.",
        },
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full line-art \
                          effect (1.0). The identity value is 0.0.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "line_art",
        wgsl_source: include_str!("line_art.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            threshold: values[0],
            strength: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<LineArtParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    #[test]
    fn test_line_art_registry_entry_exists() {
        assert!(registry_by_id("line_art").is_some());
    }

    #[test]
    fn test_line_art_registry_metadata() {
        let reg = registry_by_id("line_art").unwrap();
        assert_eq!(reg.meta.display_name, "Line Art");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Threshold",
                    min: 0.1,
                    max: 10.0,
                    default: 2.0,
                    description: "Sensitivity of edge detection. Higher values make faint edges \
                                  more visible as dark lines.",
                },
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full line-art \
                                  effect (1.0). The identity value is 0.0.",
                },
            ])
        );
        assert_eq!(
            reg.meta.passes.len(),
            1,
            "Line Art must have exactly 1 pass"
        );
    }

    #[test]
    fn test_line_art_make_uniform_known_value() {
        let reg = registry_by_id("line_art").unwrap();
        let bytes = (reg.make_uniform)(&[3.0, 0.75]);
        let expected = bytemuck::bytes_of(&LineArtParams {
            threshold: 3.0,
            strength: 0.75,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    /// At strength=0.0 the shader outputs mix(src, line_art, 0.0) = src — a true
    /// identity regardless of the threshold value.
    #[test]
    fn test_line_art_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "line_art",
                values: vec![2.0, 0.0],
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

    /// A uniform (solid-colour) image has no edges; at full strength the Sobel
    /// gradient is zero everywhere, so the output is near-white (1 - 0 = 1).
    #[test]
    fn test_line_art_solid_image_produces_white() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "line_art",
                values: vec![2.0, 1.0],
            }],
        );
        // line_value = 1 - 0 = 1.0 → u16 ≈ 65535. Allow ±200 for f16 rounding.
        for pixel in out.pixels() {
            assert!(
                pixel[0] > 60000,
                "R on solid image: expected near-white (~65535), got {}",
                pixel[0]
            );
        }
    }

    /// On an image with a sharp edge, pixels at the boundary must be noticeably
    /// darker than pixels in a flat (uniform) region after line-art conversion.
    #[test]
    fn test_line_art_edge_pixels_darker_than_flat_pixels() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32×16 step image: left half dark, right half bright.
        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 10000 } else { 55000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "line_art",
                values: vec![2.0, 1.0],
            }],
        );

        // Pixel at x=15 is on the edge; pixel at x=2 is in the flat dark region.
        let edge_pixel = out.get_pixel(15, 8)[0] as i32;
        let flat_pixel = out.get_pixel(2, 8)[0] as i32;
        assert!(
            edge_pixel < flat_pixel,
            "edge pixel (x=15) must be darker than flat-region pixel (x=2): \
             edge={edge_pixel}, flat={flat_pixel}"
        );
    }

    /// Higher threshold must amplify weak edges, producing more (darker) lines
    /// than a low threshold on the same image.
    #[test]
    fn test_line_art_higher_threshold_increases_darkness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Shallow ramp: produces a weak, uniform Sobel signal across all columns.
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
                shader_id: "line_art",
                values: vec![1.0, 1.0],
            }],
        );
        let out_high = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "line_art",
                values: vec![8.0, 1.0],
            }],
        );

        // Higher threshold → stronger edges → lower line_value → darker output.
        let mean_low: f64 = out_low.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 16.0);
        let mean_high: f64 = out_high.pixels().map(|p| p[0] as f64).sum::<f64>() / (32.0 * 16.0);
        assert!(
            mean_high < mean_low,
            "higher threshold must produce a darker (lower mean) output: \
             low={mean_low:.0}, high={mean_high:.0}"
        );
    }

    /// Alpha must pass through unchanged at any parameter combination.
    #[test]
    fn test_line_art_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "line_art",
                values: vec![2.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// Chaining Line Art with Brightness must not panic and must preserve alpha.
    #[test]
    fn test_line_art_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(8, 8, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "line_art",
                    values: vec![2.0, 0.5],
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

    /// Running Line Art twice with identical inputs must produce bit-identical output.
    #[test]
    fn test_line_art_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "line_art",
            values: vec![2.0, 0.8],
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
