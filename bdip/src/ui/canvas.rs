use bdip_core::gpu::engine::GpuEngine;
use bdip_core::gpu::texture::download_presentation_buffer;
use iced::widget::{container, image, text};
use iced::{ContentFit, Element, Length};

use super::app::BdipApp;
use super::message::Message;

/// Converts the output of `Renderer::present` to an iced image handle. Called
/// on every render (load, commit, preview, undo/redo).
pub fn presentation_to_handle(
    engine: &GpuEngine,
    buf: &bdip_core::wgpu::Buffer,
    width: u32,
    height: u32,
) -> Option<image::Handle> {
    let img16 =
        download_presentation_buffer(&engine.device, &engine.queue, buf, width, height).ok()?;
    let img8 = bdip_core::image::DynamicImage::ImageRgba16(img16).into_rgba8();
    let (w, h) = img8.dimensions();
    let pixels = img8.into_raw();
    Some(image::Handle::from_rgba(w, h, pixels))
}

pub fn view(app: &BdipApp) -> Element<'_, Message> {
    if let Some(err) = &app.error_message {
        return container(text(err.as_str()))
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into();
    }

    if let Some(handle) = &app.image_handle {
        container(
            image(handle.clone())
                .content_fit(ContentFit::Contain)
                .width(Length::Fill)
                .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    } else {
        container(text("No image loaded — click Load Image to begin."))
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into()
    }
}
