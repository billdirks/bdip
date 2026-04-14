use bdip_core::gpu::engine::GpuEngine;
use bdip_core::gpu::pipeline::Renderer;
use iced::widget::{button, column, container, image, row, text};
use iced::{ContentFit, Element, Length};

use super::app::BdipApp;
use super::message::Message;
use super::style;

/// Converts the output of `Renderer::present` to an iced image handle. Called
/// on every render (load, commit, preview, undo/redo).
pub fn presentation_to_handle(
    renderer: &mut Renderer,
    engine: &GpuEngine,
    buf: &bdip_core::wgpu::Buffer,
    width: u32,
    height: u32,
) -> Option<image::Handle> {
    let img16 = renderer.download(engine, buf, width, height).ok()?;
    let img8 = bdip_core::image::DynamicImage::ImageRgba16(img16).into_rgba8();
    let (w, h) = img8.dimensions();
    let pixels = img8.into_raw();
    Some(image::Handle::from_rgba(w, h, pixels))
}

pub fn view(app: &BdipApp) -> Element<'_, Message> {
    let canvas_content: Element<'_, Message> = if let Some(handle) = &app.image_handle {
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
    };

    if let Some(err) = &app.error_message {
        // Show a dismissible error banner above the canvas content so the image
        // (or placeholder) remains visible while the error is displayed.
        let banner = container(
            row![
                text(err.as_str()).width(Length::Fill),
                button("Dismiss").on_press(Message::DismissError),
            ]
            .spacing(8)
            .padding(8),
        )
        .style(style::error_banner)
        .width(Length::Fill);

        column![banner, canvas_content]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    } else {
        canvas_content
    }
}
