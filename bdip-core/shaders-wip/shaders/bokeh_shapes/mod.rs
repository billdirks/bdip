use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Bokeh Shapes effect.
///
/// Layout (16 bytes, matching the WGSL struct):
/// - `radius`: Size of the bokeh blur kernel in pixels. At 0.0 no blur is applied
///   (identity). Larger values produce bigger out-of-focus circles.
/// - `sides`: Number of polygon sides for the bokeh aperture shape. 0 = circle,
///   3 = triangle, 4 = square, 6 = hexagon. Non-integer values are floored by the
///   shader. When `radius` is 0.0 this value has no visual effect.
/// - `strength`: Blend weight between the original image (0.0) and the bokeh-blurred
///   result (1.0). At 0.0 the output is identical to the source (identity).
/// - `_padding`: Fills the struct to 16 bytes for WebGPU uniform alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BokehShapesParams {
    pub radius: f32,
    pub sides: f32,
    pub strength: f32,
    pub _padding: f32,
}

impl TransformShader for BokehShapesParams {
    const ID: &'static str = "bokeh_shapes";
    const DISPLAY_NAME: &'static str = "Bokeh Shapes";
    const DESCRIPTION: &'static str = "Simulates lens bokeh blur with a visible polygon aperture shape \
         (triangle, hexagon, circle, etc.) computed from a hex-distance kernel.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Radius",
            min: 0.0,
            max: 50.0,
            default: 0.0,
            description: "Bokeh kernel radius in pixels. At 0 no blur is applied.",
        },
        SliderDef {
            name: "Sides",
            min: 0.0,
            max: 12.0,
            default: 6.0,
            description: "Number of polygon sides for the aperture shape. \
                         0 = circle, 3 = triangle, 6 = hexagon.",
        },
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between original (0) and blurred result (1). \
                         At 0 the output is unchanged (identity).",
        },
    ]);

    // Four passes — the polygon blur runs at 4× downsampled resolution to keep the
    // per-pixel gather kernel cost tractable at large image sizes:
    //
    //   Pass 1 (down): Box-filter downsample the source 4× → scratch "down".
    //   Pass 2 (blur): Polygon-shaped gather kernel on the downsampled image.
    //     The kernel radius is divided by 4 in the shader to map the user-facing
    //     pixel radius onto the reduced-resolution coordinate system. The result
    //     is written to scratch "blurred_down" (still at 4× reduced size).
    //   Pass 3 (up):  Bilinear upsample of "blurred_down" back to full resolution
    //     → scratch "blurred".
    //   Pass 4 (blend): Mix the original source with the full-resolution blurred
    //     scratch using `strength` as the blend weight. At strength=0 the output
    //     equals the source (identity).
    //
    // Downsampling by 4× reduces the pixel count 16× before the gather loop,
    // making a 50 px user radius correspond to a ≤13 tap radius on the small
    // image — the same cost-reduction strategy used by Clarity and Tilt-Shift.
    //
    // A RADIUS_CAP in the blur shader bounds the loop per-dispatch independently
    // of image size; at 4× down it is set to ceil(50/4) = 13.
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "down",
            wgsl_source: include_str!("bokeh_shapes_down.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("down"),
            output_scale: PassScale::Down(4),
            aux_textures: &[],
        },
        PassDef {
            label: "blur",
            wgsl_source: include_str!("bokeh_shapes_blur.wgsl"),
            inputs: &[PassInput::Scratch("down")],
            output: PassOutput::Scratch("blurred_down"),
            output_scale: PassScale::Down(4),
            aux_textures: &[],
        },
        PassDef {
            label: "up",
            wgsl_source: include_str!("bokeh_shapes_up.wgsl"),
            inputs: &[PassInput::Scratch("blurred_down")],
            output: PassOutput::Scratch("blurred"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "blend",
            wgsl_source: include_str!("bokeh_shapes_blend.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("blurred")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self {
            radius: values[0],
            sides: values[1],
            strength: values[2],
            _padding: 0.0,
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    BokehShapesParams,
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
    fn test_bokeh_shapes_registry_entry_exists() {
        assert!(registry_by_id("bokeh_shapes").is_some());
    }

    #[test]
    fn test_bokeh_shapes_registry_display_name() {
        let reg = registry_by_id("bokeh_shapes").unwrap();
        assert_eq!(reg.meta.display_name, "Bokeh Shapes");
    }

    #[test]
    fn test_bokeh_shapes_registry_pass_count() {
        let reg = registry_by_id("bokeh_shapes").unwrap();
        assert_eq!(
            reg.meta.passes.len(),
            4,
            "Bokeh Shapes must have exactly 4 passes"
        );
    }

    #[test]
    fn test_bokeh_shapes_registry_param_kind_is_sliders() {
        let reg = registry_by_id("bokeh_shapes").unwrap();
        assert!(
            matches!(reg.meta.param, ParamKind::Sliders(_)),
            "param kind must be Sliders"
        );
    }

    #[test]
    fn test_bokeh_shapes_registry_slider_count() {
        let reg = registry_by_id("bokeh_shapes").unwrap();
        if let ParamKind::Sliders(sliders) = reg.meta.param {
            assert_eq!(
                sliders.len(),
                3,
                "Bokeh Shapes must expose exactly 3 sliders"
            );
        }
    }

    #[test]
    fn test_bokeh_shapes_registry_slider_defaults() {
        let reg = registry_by_id("bokeh_shapes").unwrap();
        if let ParamKind::Sliders(sliders) = reg.meta.param {
            assert_eq!(sliders[0].default, 0.0, "radius default must be 0.0");
            assert_eq!(sliders[1].default, 6.0, "sides default must be 6.0");
            assert_eq!(sliders[2].default, 0.0, "strength default must be 0.0");
        }
    }

    #[test]
    fn test_bokeh_shapes_registry_metadata() {
        let reg = registry_by_id("bokeh_shapes").unwrap();
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Radius",
                    min: 0.0,
                    max: 50.0,
                    default: 0.0,
                    description: "Bokeh kernel radius in pixels. At 0 no blur is applied.",
                },
                SliderDef {
                    name: "Sides",
                    min: 0.0,
                    max: 12.0,
                    default: 6.0,
                    description: "Number of polygon sides for the aperture shape. \
                                 0 = circle, 3 = triangle, 6 = hexagon.",
                },
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between original (0) and blurred result (1). \
                                 At 0 the output is unchanged (identity).",
                },
            ])
        );
    }

    // -------------------------------------------------------------------------
    // Uniform serialisation
    // -------------------------------------------------------------------------

    #[test]
    fn test_bokeh_shapes_make_uniform_known_value() {
        let reg = registry_by_id("bokeh_shapes").unwrap();
        let bytes = (reg.make_uniform)(&[10.0, 6.0, 0.5]);
        let expected = bytemuck::bytes_of(&BokehShapesParams {
            radius: 10.0,
            sides: 6.0,
            strength: 0.5,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_bokeh_shapes_make_uniform_zero_values() {
        let reg = registry_by_id("bokeh_shapes").unwrap();
        let bytes = (reg.make_uniform)(&[0.0, 0.0, 0.0]);
        let expected = bytemuck::bytes_of(&BokehShapesParams {
            radius: 0.0,
            sides: 0.0,
            strength: 0.0,
            _padding: 0.0,
        });
        assert_eq!(bytes, expected);
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — identity conditions
    // -------------------------------------------------------------------------

    /// Default parameters (radius=0, strength=0) must produce no change.
    /// With radius=0 the blur pass copies the centre pixel unchanged, and
    /// strength=0 means the blend pass outputs 100% of the original source.
    #[test]
    fn test_bokeh_shapes_default_params_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "bokeh_shapes",
                values: vec![0.0, 6.0, 0.0],
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

    /// strength=0 with a non-zero radius is still identity because the blend pass
    /// ignores the blurred scratch at blend weight 0.
    #[test]
    fn test_bokeh_shapes_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 20000, 30000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "bokeh_shapes",
                values: vec![10.0, 6.0, 0.0],
            }],
        );
        for pixel in out.pixels() {
            assert!((pixel[0] as i32 - 20000).abs() <= 64, "R channel mismatch");
            assert!((pixel[1] as i32 - 30000).abs() <= 64, "G channel mismatch");
            assert!((pixel[2] as i32 - 40000).abs() <= 64, "B channel mismatch");
        }
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — blur effect
    // -------------------------------------------------------------------------

    /// A checkerboard pattern blurred with a large radius and full strength must
    /// converge toward mid-gray. Adjacent pixels that were originally opposite
    /// extremes (0 and 65535) should have a difference less than half of max after
    /// blurring.
    #[test]
    fn test_bokeh_shapes_blur_reduces_high_frequency_detail() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 32×32 checkerboard: alternating 0 and 65535 per pixel.
        let mut img = crate::Rgba16Image::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v: u16 = if (x + y) % 2 == 0 { 65535 } else { 0 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "bokeh_shapes",
                values: vec![8.0, 6.0, 1.0],
            }],
        );

        // Centre pixel: adjacent pixels originally 0 and 65535 should blur toward
        // mid-gray; their difference must be less than half of max.
        let p0 = out.get_pixel(15, 15)[0] as i32;
        let p1 = out.get_pixel(16, 15)[0] as i32;
        let diff = (p0 - p1).abs();
        assert!(
            diff < 32767,
            "blurred checkerboard must converge toward mid-gray: diff={diff}"
        );
    }

    /// A solid-color image blurred at any radius and strength should remain close
    /// to the original, because every sample in the kernel has the same value.
    #[test]
    fn test_bokeh_shapes_solid_image_unchanged_after_blur() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 50000, 20000, 10000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "bokeh_shapes",
                values: vec![10.0, 6.0, 1.0],
            }],
        );
        for pixel in out.pixels() {
            assert!((pixel[0] as i32 - 50000).abs() <= 64, "R channel mismatch");
            assert!((pixel[1] as i32 - 20000).abs() <= 64, "G channel mismatch");
            assert!((pixel[2] as i32 - 10000).abs() <= 64, "B channel mismatch");
        }
    }

    /// Increasing radius with constant strength=1 must increase the blur: a
    /// step-edge image should show a wider transition zone at radius=15 than at
    /// radius=5.
    #[test]
    fn test_bokeh_shapes_larger_radius_blurs_more() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // 64×32 step image: left half 0, right half 65535.
        let mut img = crate::Rgba16Image::new(64, 32);
        for y in 0..32u32 {
            for x in 0..64u32 {
                let v: u16 = if x < 32 { 0 } else { 65535 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_small = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "bokeh_shapes",
                values: vec![3.0, 6.0, 1.0],
            }],
        );
        let out_large = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "bokeh_shapes",
                values: vec![15.0, 6.0, 1.0],
            }],
        );

        // At the edge (x=32), the large-radius blur should mix more dark and bright
        // pixels, pulling the value toward mid-gray more than the small-radius blur.
        // Concretely: large-radius output at x=32 should be closer to 32767 than
        // small-radius output.
        let edge_small = out_small.get_pixel(32, 16)[0] as i32;
        let edge_large = out_large.get_pixel(32, 16)[0] as i32;
        let dist_small = (edge_small - 32767).abs();
        let dist_large = (edge_large - 32767).abs();
        assert!(
            dist_large <= dist_small,
            "larger radius must blur edge more: dist_large={dist_large}, dist_small={dist_small}"
        );
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — polygon shapes
    // -------------------------------------------------------------------------

    /// Circle mode (sides=0) and hexagon mode (sides=6) both blur a checkerboard
    /// toward mid-gray — verifying that both shape paths execute without error and
    /// produce a blurred result.
    #[test]
    fn test_bokeh_shapes_circle_mode_blurs() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v: u16 = if (x + y) % 2 == 0 { 65535 } else { 0 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "bokeh_shapes",
                values: vec![8.0, 0.0, 1.0],
            }],
        );

        let p0 = out.get_pixel(15, 15)[0] as i32;
        let p1 = out.get_pixel(16, 15)[0] as i32;
        let diff = (p0 - p1).abs();
        assert!(
            diff < 32767,
            "circle mode must blur checkerboard toward mid-gray: diff={diff}"
        );
    }

    #[test]
    fn test_bokeh_shapes_triangle_mode_blurs() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v: u16 = if (x + y) % 2 == 0 { 65535 } else { 0 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "bokeh_shapes",
                values: vec![8.0, 3.0, 1.0],
            }],
        );

        let p0 = out.get_pixel(15, 15)[0] as i32;
        let p1 = out.get_pixel(16, 15)[0] as i32;
        let diff = (p0 - p1).abs();
        assert!(
            diff < 32767,
            "triangle mode must blur checkerboard toward mid-gray: diff={diff}"
        );
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — alpha preservation
    // -------------------------------------------------------------------------

    /// The blend pass copies alpha from the source; neither pass must alter the
    /// alpha channel.
    #[test]
    fn test_bokeh_shapes_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "bokeh_shapes",
                values: vec![10.0, 6.0, 0.8],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — chaining
    // -------------------------------------------------------------------------

    /// Bokeh Shapes followed by Grayscale must complete without panic and produce
    /// a valid non-zero image, verifying that the 2-pass scratch texture chains
    /// correctly with subsequent transforms.
    #[test]
    fn test_bokeh_shapes_chains_with_grayscale() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "bokeh_shapes",
                    values: vec![5.0, 6.0, 0.5],
                },
                Transform {
                    shader_id: "grayscale",
                    values: vec![],
                },
            ],
        );
        assert!(
            out.pixels().any(|p| p[0] > 0),
            "chained output must be non-zero"
        );
    }

    // -------------------------------------------------------------------------
    // GPU roundtrip — determinism
    // -------------------------------------------------------------------------

    /// Running the shader twice with identical inputs must produce bit-identical output.
    #[test]
    fn test_bokeh_shapes_deterministic() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let transform = Transform {
            shader_id: "bokeh_shapes",
            values: vec![8.0, 6.0, 0.7],
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
