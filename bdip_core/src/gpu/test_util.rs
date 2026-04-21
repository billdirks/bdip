use crate::gpu::engine::GpuEngine;
use crate::gpu::image_pipeline::Renderer;
use crate::gpu::shaders::Transform;
use crate::gpu::texture::{download_presentation_buffer, upload_texture};

pub fn make_solid_image(w: u32, h: u32, r: u16, g: u16, b: u16) -> crate::Rgba16Image {
    let mut img = crate::Rgba16Image::new(w, h);
    for pixel in img.pixels_mut() {
        *pixel = image::Rgba([r, g, b, 65535]);
    }
    img
}

pub fn roundtrip(
    renderer: &mut Renderer,
    engine: &GpuEngine,
    img: &crate::Rgba16Image,
    transforms: &[Transform],
) -> crate::Rgba16Image {
    let (w, h) = (img.width(), img.height());
    let upload = upload_texture(&engine.device, &engine.queue, img);
    let mut current = renderer.ingest(engine, &upload);
    for t in transforms {
        current = renderer.apply(engine, &current, t);
    }
    let buf = renderer.present(engine, &current);
    download_presentation_buffer(&engine.device, &engine.queue, &buf, w, h).unwrap()
}
