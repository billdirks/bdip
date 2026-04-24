use crate::gpu::engine::GpuEngine;
use crate::gpu::image_pipeline::Renderer;
use crate::gpu::shaders::Transform;
use crate::gpu::test_util::{make_solid_image, roundtrip};

/// Stacking Brightness(+0.2) → Clarity(+0.5) must not cancel the brightness lift
/// that Brightness provides. Positive Clarity redistributes local contrast (boosts
/// bright edges, darkens dark edges) but must not cause a significant global mean
/// reduction relative to Brightness alone.
///
/// On interior pixels (uniform regions far from any edge) Clarity's hp=0, so the
/// combined output is approximately the same as Brightness alone. Over the whole
/// image the redistribution is roughly mean-neutral; we allow ≤ 32 u16 per-pixel
/// drop to confirm Clarity is not catastrophically suppressing Brightness.
#[test]
fn test_brightness_then_clarity() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Step image: left half dark, right half mid-gray. Gives Clarity genuine edges
    // to act on while keeping both sides well below the sRGB ceiling.
    let mut img = crate::Rgba16Image::new(32, 16);
    for y in 0..16u32 {
        for x in 0..32u32 {
            let v: u16 = if x < 16 { 8000 } else { 32767 };
            img.put_pixel(x, y, image::Rgba([v, v, v, 65535]));
        }
    }

    let out_brightness = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "brightness",
            values: vec![0.2],
        }],
    );
    let out_combined = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[
            Transform {
                shader_id: "brightness",
                values: vec![0.2],
            },
            Transform {
                shader_id: "clarity",
                values: vec![0.5],
            },
        ],
    );

    let num_pixels = (32u32 * 16u32) as i64;
    let sum_brightness: i64 = out_brightness.pixels().map(|p| p[0] as i64).sum();
    let sum_combined: i64 = out_combined.pixels().map(|p| p[0] as i64).sum();

    // Allow up to 32 u16 per-pixel drop — large enough to absorb Clarity's
    // contrast redistribution and f16 rounding, small enough to catch a real
    // regression where Clarity inverts or suppresses Brightness.
    assert!(
        sum_combined + 32 * num_pixels >= sum_brightness,
        "Brightness→Clarity must not significantly reduce mean brightness: \
         brightness_sum={sum_brightness}, combined_sum={sum_combined}"
    );
}

/// Composing Clarity(+0.5) then Vignette must not panic, must return an image of
/// the correct dimensions, and must preserve alpha on every pixel. This validates
/// that a multi-pass shader (Clarity) hands its output texture to a single-pass
/// shader (Vignette) correctly through the shared scratch pool.
#[test]
fn test_clarity_then_vignette() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(16, 16, 32767, 32767, 32767);

    // Vignette default params: radius=0.8, softness=0.5.
    let out = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[
            Transform {
                shader_id: "clarity",
                values: vec![0.5],
            },
            Transform {
                shader_id: "vignette",
                values: vec![0.8, 0.5],
            },
        ],
    );

    assert_eq!(out.width(), 16, "output width must match input");
    assert_eq!(out.height(), 16, "output height must match input");
    for pixel in out.pixels() {
        assert_eq!(
            pixel[3], 65535,
            "alpha must be preserved through Clarity→Vignette"
        );
    }
}

/// Composing Cartoon (posterization) then Saturation must not restore the colors
/// that Cartoon quantized away. Saturation at +1.0 scales the chroma of each pixel
/// but cannot introduce new posterization levels; the unique-color count after the
/// stack must remain within ±5% of the Cartoon-alone count.
#[test]
fn test_cartoon_then_saturation() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // Grayscale gradient (R = G = B) ensures saturation is exactly identity:
    // for any pixel where R = G = B, luma = R and the saturation formula gives
    // R_out = luma + (R − luma) × scale = R. This isolates the posterization
    // effect and prevents Saturation(1.0) from clipping upper quantized levels
    // to 1.0 (which would spuriously merge them and inflate the ratio).
    let mut img = crate::Rgba16Image::new(32, 32);
    let total = 32u32 * 32u32;
    for (i, pixel) in img.pixels_mut().enumerate() {
        let v = (1000 + (i as u32 * 63535 / total)) as u16;
        *pixel = image::Rgba([v, v, v, 65535]);
    }

    // strength=1 to engage posterization fully; edge_darkness=0 to isolate color
    // quantization without edge darkening complicating the unique-color count.
    let cartoon_params = vec![1.0f32, 4.0, 0.15, 0.10, 0.0];

    let out_cartoon = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[Transform {
            shader_id: "cartoon",
            values: cartoon_params.clone(),
        }],
    );
    let out_cartoon_then_sat = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[
            Transform {
                shader_id: "cartoon",
                values: cartoon_params,
            },
            Transform {
                shader_id: "saturation",
                values: vec![1.0],
            },
        ],
    );

    // Count distinct pixel values (R channel) in each output.
    let unique_cartoon: std::collections::HashSet<u16> =
        out_cartoon.pixels().map(|p| p[0]).collect();
    let unique_combined: std::collections::HashSet<u16> =
        out_cartoon_then_sat.pixels().map(|p| p[0]).collect();

    // Saturation must not restore quantized colors — the count stays within ±5%
    // of the Cartoon-alone count.
    let n_cartoon = unique_cartoon.len() as f64;
    let n_combined = unique_combined.len() as f64;
    let ratio = (n_combined - n_cartoon).abs() / n_cartoon.max(1.0);
    assert!(
        ratio <= 0.05,
        "Cartoon→Saturation must not restore posterized colors: \
         cartoon_unique={}, combined_unique={}, ratio={:.3}",
        unique_cartoon.len(),
        unique_combined.len(),
        ratio
    );
}

#[test]
fn test_brightness_saturation_commutativity() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(2, 2, 32767, 16384, 8192);

    // brightness (uniform additive offset) and saturation (linear scaling around luminance)
    // commute exactly when Rec.709 coefficients sum to 1.0 — which they do
    // (0.2126 + 0.7152 + 0.0722 = 1.0). Both orderings must produce algebraically
    // identical results.
    let bright_then_sat = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[
            Transform {
                shader_id: "brightness",
                values: vec![0.3],
            },
            Transform {
                shader_id: "saturation",
                values: vec![-0.5],
            },
        ],
    );
    let sat_then_bright = roundtrip(
        &mut renderer,
        &engine,
        &img,
        &[
            Transform {
                shader_id: "saturation",
                values: vec![-0.5],
            },
            Transform {
                shader_id: "brightness",
                values: vec![0.3],
            },
        ],
    );

    for y in 0..2u32 {
        for x in 0..2u32 {
            let a = bright_then_sat.get_pixel(x, y);
            let b = sat_then_bright.get_pixel(x, y);
            assert!(
                (a[0] as i32 - b[0] as i32).abs() <= 64,
                "R at ({x},{y}): order A={}, order B={}",
                a[0],
                b[0]
            );
            assert!(
                (a[1] as i32 - b[1] as i32).abs() <= 64,
                "G at ({x},{y}): order A={}, order B={}",
                a[1],
                b[1]
            );
            assert!(
                (a[2] as i32 - b[2] as i32).abs() <= 64,
                "B at ({x},{y}): order A={}, order B={}",
                a[2],
                b[2]
            );
        }
    }
}
