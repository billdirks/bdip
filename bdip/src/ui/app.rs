use bdip_core::HistoryManager;
use bdip_core::gpu::engine::GpuEngine;
use bdip_core::gpu::pipeline::Renderer;
use bdip_core::gpu::texture::upload_texture;
use iced::widget::{column, container, row};
use iced::{Element, Length, Subscription, Task};
use std::path::PathBuf;

use super::canvas;
use super::canvas::presentation_to_handle;
use super::menu_bar;
use super::message::{Message, TransformOption};
use super::sidebar;

pub struct BdipApp {
    // Image state
    pub base_image: Option<bdip_core::Rgba16Image>,
    pub image_handle: Option<iced::widget::image::Handle>,

    // GPU state
    pub engine: Option<GpuEngine>,
    pub renderer: Option<Renderer>,
    pub cached_base_texture: Option<bdip_core::wgpu::Texture>,

    // Transform state
    pub history: HistoryManager,
    pub selected_transform: TransformOption,
    // Used in PR 2 (live preview).
    #[allow(dead_code)]
    pub preview_value: f32,
    #[allow(dead_code)]
    pub is_previewing: bool,

    // UI state
    pub error_message: Option<String>,
    pub is_loading: bool,
    // Used in PR 4 (save flow).
    #[allow(dead_code)]
    pub is_saving: bool,
}

impl BdipApp {
    pub fn new(input_path: Option<PathBuf>) -> (Self, Task<Message>) {
        let (engine, renderer, error_message) = match GpuEngine::new() {
            Ok(e) => {
                let r = Renderer::new(&e);
                (Some(e), Some(r), None)
            }
            Err(e) => (None, None, Some(format!("GPU init failed: {e}"))),
        };

        let has_input = input_path.is_some();
        let task = match input_path {
            Some(path) => load_image_task(path),
            None => Task::none(),
        };

        (
            BdipApp {
                base_image: None,
                image_handle: None,
                engine,
                renderer,
                cached_base_texture: None,
                history: HistoryManager::new(),
                selected_transform: TransformOption::Brightness,
                preview_value: 0.0,
                is_previewing: false,
                error_message,
                is_loading: has_input,
                is_saving: false,
            },
            task,
        )
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::LoadImagePressed => {
                if self.is_loading {
                    return Task::none();
                }
                self.is_loading = true;
                Task::perform(
                    async {
                        let handle = rfd::AsyncFileDialog::new()
                            .add_filter("Images", &["png", "jpg", "jpeg", "gif", "tif", "tiff"])
                            .pick_file()
                            .await;
                        match handle {
                            Some(h) => {
                                let path = h.path().to_path_buf();
                                let img =
                                    bdip_core::io::load_image(&path).map_err(|e| e.to_string())?;
                                Ok((path, img))
                            }
                            None => Err("cancelled".to_string()),
                        }
                    },
                    |result: Result<(PathBuf, bdip_core::Rgba16Image), String>| match result {
                        Ok(pair) => Message::ImageLoaded(Ok(pair)),
                        Err(e) if e == "cancelled" => Message::Noop,
                        Err(e) => Message::ImageLoaded(Err(e)),
                    },
                )
            }

            Message::ImageLoaded(Ok((_, img))) => {
                let (w, h) = img.dimensions();
                if let (Some(engine), Some(renderer)) = (&self.engine, &mut self.renderer) {
                    let uploaded = upload_texture(&engine.device, &engine.queue, &img);
                    let linear = renderer.ingest(engine, &uploaded);
                    let buf = renderer.present(engine, &linear);
                    self.image_handle = presentation_to_handle(engine, &buf, w, h);
                    self.cached_base_texture = Some(linear);
                }
                self.base_image = Some(img);
                self.history.clear();
                self.is_loading = false;
                self.error_message = None;
                Task::none()
            }

            Message::ImageLoaded(Err(e)) => {
                self.error_message = Some(e);
                self.is_loading = false;
                Task::none()
            }

            Message::SaveImagePressed => {
                // Implemented in PR 4.
                Task::none()
            }

            Message::ImageSaved(_) => Task::none(),

            Message::TransformSelected(opt) => {
                self.selected_transform = opt;
                Task::none()
            }

            Message::SliderChanged(_) | Message::SliderReleased | Message::ApplyParameterless => {
                // Implemented in PR 2.
                Task::none()
            }

            Message::Undo | Message::Redo => {
                // Implemented in PR 3.
                Task::none()
            }

            Message::DismissError => {
                self.error_message = None;
                Task::none()
            }

            Message::Noop => Task::none(),
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        let menu = menu_bar::view(self);
        let sidebar = sidebar::view(self);
        let canvas = canvas::view(self);

        let content = row![
            container(sidebar).width(Length::Fixed(250.0)),
            container(canvas).width(Length::Fill).height(Length::Fill),
        ];

        column![menu, content]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    pub fn subscription(&self) -> Subscription<Message> {
        Subscription::none()
    }
}

fn load_image_task(path: PathBuf) -> Task<Message> {
    Task::perform(
        async move {
            let img = bdip_core::io::load_image(&path).map_err(|e| e.to_string())?;
            Ok((path, img))
        },
        Message::ImageLoaded,
    )
}
