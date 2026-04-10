use anyhow::Context;
use bdip_core::gpu::engine::GpuEngine;
use bdip_core::gpu::pipeline::Renderer;
use bdip_core::gpu::texture::{download_texture, upload_texture};
use iced::widget::{container, image};
use iced::{Element, Length, Task};
use std::path::PathBuf;
use std::time::Instant;

pub fn run(input: Option<PathBuf>) -> anyhow::Result<()> {
    iced::application(
        move || SpikeApp::new(input.clone()),
        SpikeApp::update,
        SpikeApp::view,
    )
    .title("bdip - Spike Prototype")
    .run()
    .context("Failed to run iced application")?;

    Ok(())
}

struct SpikeApp {
    image_handle: Option<image::Handle>,
}

#[derive(Debug, Clone)]
enum Message {}

impl SpikeApp {
    fn new(input_path: Option<PathBuf>) -> (Self, Task<Message>) {
        let mut app = SpikeApp { image_handle: None };

        let loaded_image = if let Some(path) = input_path {
            bdip_core::io::load_image(&path).expect("Failed to load image via UI spike")
        } else {
            // Generate an 800x600 test image (gradient pattern)
            bdip_core::Rgba16Image::from_fn(800, 600, |x, y| {
                let r = ((x % 255) as u16) * 257;
                let g = ((y % 255) as u16) * 257;
                bdip_core::image::Rgba([r, g, 32768, 65535])
            })
        };

        // Initialize GPU
        let engine = GpuEngine::new().expect("Failed to init GPU engine");
        let renderer = Renderer::new(&engine);

        let current_texture = upload_texture(&engine.device, &engine.queue, &loaded_image);
        println!("Applying hardcoded brightness: 0.5");
        let processed_texture = renderer.apply_brightness(&engine, &current_texture, 0.5);

        let (width, height) = loaded_image.dimensions();

        println!("Starting CPU bridge readback...");
        let start = Instant::now();
        let final_image = download_texture(
            &engine.device,
            &engine.queue,
            &processed_texture,
            width,
            height,
        )
        .expect("Failed to download texture");
        let elapsed = start.elapsed();
        println!(
            "CPU Bridge readback took: {:.2} ms",
            elapsed.as_secs_f64() * 1000.0
        );

        let final_image_8bit =
            bdip_core::image::DynamicImage::ImageRgba16(final_image).into_rgba8();

        let (out_width, out_height) = final_image_8bit.dimensions();
        let pixels = final_image_8bit.into_raw();

        app.image_handle = Some(image::Handle::from_rgba(out_width, out_height, pixels));
        (app, Task::none())
    }

    fn update(&mut self, _message: Message) -> Task<Message> {
        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        if let Some(handle) = &self.image_handle {
            container(
                image(handle.clone())
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into()
        } else {
            container(iced::widget::text("Loading...")).into()
        }
    }
}
