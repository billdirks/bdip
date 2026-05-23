use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

/// Parameters for the Emboss effect.
///
/// The two meaningful fields pack into 8 bytes; two padding floats bring the
/// struct to 16 bytes to satisfy WebGPU's uniform alignment requirement.
///
/// # Identity design
///
/// The spec requires that default parameter values produce a no-op transformation.
/// A pure emboss at any non-zero `strength` replaces image colour with a grayscale
/// relief map, so a literal identity is not achievable once the effect is engaged.
///
/// The design uses `strength` as both the convolution scale and the source/emboss
/// blend weight: `mix(src, emboss_gray, strength)`. At `strength = 0.0` the output
/// equals the source regardless of `direction`, satisfying the identity requirement
/// with a single slider. This pattern mirrors the Pencil Sketch and Stained Glass
/// shaders.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EmbossParams {
    /// Blend factor and convolution scale: 0.0 = source unchanged (identity),
    /// 1.0 = full emboss relief. Range [0.0, 1.0].
    pub strength: f32,
    /// Lighting direction in degrees. Controls which axis the relief ridge appears
    /// to be lit from. 0° = right, 90° = down, 180° = left, 270° = up.
    pub direction: f32,
    pub _padding: [f32; 2],
}

impl TransformShader for EmbossParams {
    const ID: &'static str = "emboss";
    const DISPLAY_NAME: &'static str = "Emboss";
    const DESCRIPTION: &'static str = "Creates a raised relief appearance by sampling luminance differences between \
         opposing neighbours and mapping them to light and shadow ridges on a mid-gray base.";
    const PARAM: ParamKind = ParamKind::Sliders(&[
        SliderDef {
            name: "Strength",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            description: "Blend between the original image (0.0) and the full emboss relief \
                          (1.0). The identity value is 0.0.",
        },
        SliderDef {
            name: "Direction",
            min: 0.0,
            max: 360.0,
            default: 45.0,
            description: "Lighting direction in degrees (0° = right, 90° = down, \
                          180° = left, 270° = up). Controls the apparent angle of the \
                          relief illumination.",
        },
    ]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "emboss",
        wgsl_source: include_str!("emboss.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self {
            strength: values[0],
            direction: values[1],
            _padding: [0.0; 2],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<EmbossParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::GpuEngine;
    use crate::gpu::image_pipeline::Renderer;
    use crate::gpu::shaders::{ParamKind, SliderDef, Transform, registry_by_id};
    use crate::gpu::test_util::{make_solid_image, roundtrip};

    // ── Registry tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_emboss_registry_entry_exists() {
        assert!(registry_by_id("emboss").is_some());
    }

    #[test]
    fn test_emboss_registry_metadata() {
        let reg = registry_by_id("emboss").unwrap();
        assert_eq!(reg.meta.display_name, "Emboss");
        assert_eq!(
            reg.meta.param,
            ParamKind::Sliders(&[
                SliderDef {
                    name: "Strength",
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                    description: "Blend between the original image (0.0) and the full emboss \
                                  relief (1.0). The identity value is 0.0.",
                },
                SliderDef {
                    name: "Direction",
                    min: 0.0,
                    max: 360.0,
                    default: 45.0,
                    description: "Lighting direction in degrees (0° = right, 90° = down, \
                                  180° = left, 270° = up). Controls the apparent angle of the \
                                  relief illumination.",
                },
            ])
        );
        assert_eq!(reg.meta.passes.len(), 1, "Emboss must have exactly 1 pass");
    }

    #[test]
    fn test_emboss_make_uniform_known_value() {
        let reg = registry_by_id("emboss").unwrap();
        let bytes = (reg.make_uniform)(&[0.8, 135.0]);
        let expected = bytemuck::bytes_of(&EmbossParams {
            strength: 0.8,
            direction: 135.0,
            _padding: [0.0; 2],
        });
        assert_eq!(bytes, expected);
    }

    // ── GPU roundtrip tests ─────────────────────────────────────────────────────

    /// At strength=0.0 the shader outputs mix(src, emboss, 0.0) = src, so every
    /// pixel must be unchanged regardless of the direction value.
    #[test]
    fn test_emboss_zero_strength_is_identity() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 20000, 50000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "emboss",
                values: vec![0.0, 45.0],
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

    /// A uniform (solid-colour) image has zero luminance difference between every
    /// pair of opposing neighbours. Adding 0.5 to the zero difference produces a
    /// mid-gray emboss value (0.5 linear → ~32767 u16). At full strength every
    /// output pixel must be near mid-gray.
    #[test]
    fn test_emboss_solid_image_produces_mid_gray() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        // Use a bright non-gray source so any colour bleed would be detectable.
        let img = make_solid_image(16, 16, 60000, 10000, 40000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "emboss",
                values: vec![1.0, 45.0],
            }],
        );
        // 0.5 linear → sRGB ≈ 0.735 → u16 ≈ 48059. Allow ±500 for f16 rounding.
        for pixel in out.pixels() {
            assert!(
                (pixel[0] as i32 - 48059).abs() <= 500,
                "R on solid image at full strength: expected ~48059 (mid-gray), got {}",
                pixel[0]
            );
            assert!(
                (pixel[1] as i32 - 48059).abs() <= 500,
                "G on solid image at full strength: expected ~48059 (mid-gray), got {}",
                pixel[1]
            );
            assert!(
                (pixel[2] as i32 - 48059).abs() <= 500,
                "B on solid image at full strength: expected ~48059 (mid-gray), got {}",
                pixel[2]
            );
        }
    }

    /// The alpha channel must pass through unchanged at any parameter combination.
    #[test]
    fn test_emboss_alpha_preserved() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 32767, 32767, 32767);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "emboss",
                values: vec![1.0, 45.0],
            }],
        );
        for pixel in out.pixels() {
            assert_eq!(pixel[3], 65535, "alpha must be preserved");
        }
    }

    /// On a step image with a sharp horizontal edge, pixels at the edge must
    /// differ measurably from mid-gray (0.5 relief) at full strength. The ridge
    /// appears lighter on the bright-to-dark side and darker on the dark-to-bright
    /// side when lit from a given direction.
    #[test]
    fn test_emboss_edge_pixels_differ_from_mid_gray() {
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
                shader_id: "emboss",
                values: vec![1.0, 0.0], // 0° direction = lighting from the right
            }],
        );

        // Pixel at x=15 (left side of edge) has a bright neighbour to the right
        // and a dark neighbour to the left → positive height difference → brighter
        // than mid-gray.
        let edge_pixel = out.get_pixel(15, 8)[0] as i32;
        // Mid-gray in u16 is ~48059; the edge should deviate by at least 2000.
        assert!(
            (edge_pixel - 48059).abs() > 2000,
            "edge pixel must differ from mid-gray: got {edge_pixel}, mid-gray ~48059"
        );
    }

    /// Changing the direction parameter must produce a different emboss pattern on
    /// an image that has edges not aligned with a single axis. A diagonal direction
    /// will sample different offsets than a horizontal direction.
    #[test]
    fn test_emboss_direction_changes_output() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        // Step image with both horizontal and vertical edges.
        let mut img = crate::Rgba16Image::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 && y < 16 { 5000 } else { 60000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_0 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "emboss",
                values: vec![1.0, 0.0],
            }],
        );
        let out_90 = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "emboss",
                values: vec![1.0, 90.0],
            }],
        );

        // At least one pixel must differ between the two direction outputs.
        let any_different = out_0
            .pixels()
            .zip(out_90.pixels())
            .any(|(a, b)| (a[0] as i32 - b[0] as i32).abs() > 64);
        assert!(
            any_different,
            "direction=0° and direction=90° must produce different outputs on a two-axis step image"
        );
    }

    /// Higher strength must produce a larger deviation from mid-gray specifically at
    /// the edge pixels where the relief signal is non-zero. At the step boundary
    /// (x=15, lit from the right at 0°), the forward neighbour is bright and the
    /// backward neighbour is dark; the height difference is large. Doubling strength
    /// from 0.5 to 1.0 doubles the convolution scale before the +0.5 offset, so the
    /// edge pixel should deviate further from mid-gray at full strength.
    #[test]
    fn test_emboss_higher_strength_increases_edge_relief() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);

        let mut img = crate::Rgba16Image::new(32, 16);
        for y in 0..16u32 {
            for x in 0..32u32 {
                let v: u16 = if x < 16 { 5000 } else { 60000 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
            }
        }

        let out_half = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "emboss",
                values: vec![0.5, 0.0],
            }],
        );
        let out_full = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[Transform {
                shader_id: "emboss",
                values: vec![1.0, 0.0],
            }],
        );

        // At the edge pixel (x=15, y=8), lit from the right (0°):
        //   fwd = pixel at x=16 (bright, ~60000) → luma high
        //   bwd = pixel at x=14 (dark,  ~ 5000) → luma low
        //   height_diff > 0 → emboss_luma > 0.5 → brighter than mid-gray.
        // At strength=1.0 the mix weight and convolution scale are both higher,
        // so the deviation from mid-gray must exceed that at strength=0.5.
        let mid_gray: i32 = 48059;
        let edge_half = out_half.get_pixel(15, 8)[0] as i32;
        let edge_full = out_full.get_pixel(15, 8)[0] as i32;

        let dev_half = (edge_half - mid_gray).unsigned_abs();
        let dev_full = (edge_full - mid_gray).unsigned_abs();

        assert!(
            dev_full > dev_half,
            "edge pixel at x=15: strength=1.0 must deviate further from mid-gray than \
             strength=0.5: dev_half={dev_half}, dev_full={dev_full} \
             (half={edge_half}, full={edge_full}, mid_gray={mid_gray})"
        );
    }

    /// Chaining Emboss after Brightness must not panic, must return correct dimensions,
    /// and must preserve the alpha channel.
    #[test]
    fn test_emboss_chains_with_brightness() {
        let engine = GpuEngine::new().unwrap();
        let mut renderer = Renderer::new(&engine);
        let img = make_solid_image(16, 16, 30000, 30000, 30000);
        let out = roundtrip(
            &mut renderer,
            &engine,
            &img,
            &[
                Transform {
                    shader_id: "brightness",
                    values: vec![0.1],
                },
                Transform {
                    shader_id: "emboss",
                    values: vec![0.7, 45.0],
                },
            ],
        );
        assert_eq!(out.width(), 16);
        assert_eq!(out.height(), 16);
        for pixel in out.pixels() {
            assert_eq!(
                pixel[3], 65535,
                "alpha must be preserved through Brightness→Emboss"
            );
        }
    }
}
