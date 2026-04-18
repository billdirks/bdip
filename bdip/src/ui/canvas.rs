use bdip_core::gpu::engine::GpuEngine;
use bdip_core::gpu::pipeline::Renderer;
use iced::widget::{button, column, container, image, row, text};
use iced::{ContentFit, Element, Length};

use super::app::BdipApp;
use super::message::Message;
use super::style;

/// Converts the output of `Renderer::present` to an iced image handle. Called
/// on every render (load, commit, preview, undo/redo).
///
/// Uses `download_slice` to borrow pixel data directly from `Renderer`'s
/// internal `pixel_vec`, avoiding a 192 MB `Vec<u16>` allocation on every
/// frame. The 16-bit values are shifted to 8-bit inline using the correct
/// formula (`p * 255 / 65535`), which matches the behaviour of
/// `image::DynamicImage::into_rgba8()`.
///
/// Note: the resulting `Vec<u8>` (~96 MB for a 24 MP image) is still
/// allocated each frame because `image::Handle::from_rgba` requires
/// ownership of the buffer. If this allocation shows up in profiles,
/// investigate whether iced exposes a zero-copy path or whether the
/// allocator free-list makes it negligible in practice.
pub fn presentation_to_handle(
    renderer: &mut Renderer,
    engine: &GpuEngine,
    buf: &bdip_core::wgpu::Buffer,
    width: u32,
    height: u32,
) -> Option<image::Handle> {
    let pixels_16 = renderer.download_slice(engine, buf, width, height).ok()?;

    // Convert 16-bit to 8-bit using the same formula the `image` crate applies
    // in `into_rgba8()` — correct round-trip division rather than a truncating
    // `>> 8` shift, avoiding off-by-one surprises at boundary values.
    let mut u8_pixels = Vec::with_capacity(pixels_16.len());
    u8_pixels.extend(pixels_16.iter().map(|&p| (p as u32 * 255 / 65535) as u8));

    Some(image::Handle::from_rgba(width, height, u8_pixels))
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
        container(text("Load an image to begin."))
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
