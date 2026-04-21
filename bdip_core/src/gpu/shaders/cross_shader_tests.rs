use crate::gpu::engine::GpuEngine;
use crate::gpu::image_pipeline::Renderer;
use crate::gpu::shaders::Transform;
use crate::gpu::test_util::{make_solid_image, roundtrip};

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
