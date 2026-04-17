use bdip_core::HistoryManager;
use bdip_core::gpu::engine::GpuEngine;
use bdip_core::gpu::pipeline::Renderer;
use bdip_core::gpu::shaders::{
    ParamKind, ShaderOption, Transform, registry_by_id, sorted_registrations,
};
use bdip_core::gpu::texture::upload_texture;
use iced::widget::{column, container, row};
use iced::{Element, Length, Subscription, Task};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::canvas;
use super::canvas::presentation_to_handle;
use super::menu_bar;
use super::message::Message;
use super::scheduler::{CompleteResult, RenderRequest, RenderScheduler, ScheduleResult};
use super::sidebar;

// ---------------------------------------------------------------------------
// GPU state
// ---------------------------------------------------------------------------

/// Owns all GPU resources shared between the UI thread and background render
/// tasks. Wrapped in `Arc<Mutex<>>` so that `Task::perform` closures can hold
/// a clone of the `Arc` and lock it on the background executor while the UI
/// thread is free to process other messages.
struct GpuState {
    engine: GpuEngine,
    renderer: Renderer,
    cached_base_texture: Option<bdip_core::wgpu::Texture>,
}

pub struct BdipApp {
    // Image state
    pub base_image: Option<bdip_core::Rgba16Image>,
    pub image_handle: Option<iced::widget::image::Handle>,

    // GPU state — shared with background render tasks via Arc<Mutex>.
    gpu: Option<Arc<Mutex<GpuState>>>,

    // Scheduling state — at most one GPU task is in-flight at a time.
    scheduler: RenderScheduler,

    // Transform state
    pub history: HistoryManager,
    pub selected_transform: ShaderOption,
    /// Current slider display value. During a drag this is the live position;
    /// otherwise it equals the last committed value for the selected transform
    /// (derived from the trailing run in history).
    pub preview_value: f32,
    pub is_previewing: bool,

    // UI state
    pub error_message: Option<String>,
    pub is_loading: bool,
    pub is_saving: bool,
}

impl BdipApp {
    pub fn new(input_path: Option<PathBuf>) -> (Self, Task<Message>) {
        let (gpu, error_message) = match GpuEngine::new() {
            Ok(engine) => {
                let renderer = Renderer::new(&engine);
                (
                    Some(Arc::new(Mutex::new(GpuState {
                        engine,
                        renderer,
                        cached_base_texture: None,
                    }))),
                    None,
                )
            }
            Err(e) => (None, Some(format!("GPU init failed: {e}"))),
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
                gpu,
                scheduler: RenderScheduler::new(),
                history: HistoryManager::new(),
                selected_transform: sorted_registrations()
                    .first()
                    .map(|r| ShaderOption {
                        id: r.meta.id,
                        display_name: r.meta.display_name,
                    })
                    .expect("shader registry must not be empty"),
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
                // Upload and ingest synchronously: these two operations are fast (they
                // submit GPU commands without blocking on completion) and must finish
                // before any render task can proceed, since they populate
                // `cached_base_texture`. The lock is brief.
                if let Some(gpu_arc) = &self.gpu {
                    let mut gpu = gpu_arc.lock().unwrap();
                    let uploaded = upload_texture(&gpu.engine.device, &gpu.engine.queue, &img);
                    let linear = gpu.renderer.ingest(&gpu.engine, &uploaded);
                    gpu.cached_base_texture = Some(linear);
                }
                self.base_image = Some(img);
                self.history.clear();
                self.preview_value = 0.0;
                self.is_previewing = false;
                self.is_loading = false;
                self.error_message = None;
                // Dispatch the initial preview render asynchronously.
                let render_list = build_render_list(&self.history, None);
                self.spawn_render(RenderRequest::Preview {
                    render_list,
                    width: w,
                    height: h,
                })
            }

            Message::ImageLoaded(Err(e)) => {
                self.error_message = Some(e);
                self.is_loading = false;
                Task::none()
            }

            Message::SaveImagePressed => {
                if self.is_saving || !self.has_base_texture() {
                    return Task::none();
                }
                let Some(base_image) = &self.base_image else {
                    return Task::none();
                };
                let (w, h) = base_image.dimensions();
                let render_list = build_render_list(&self.history, None);
                self.is_saving = true;
                self.spawn_render(RenderRequest::Save {
                    render_list,
                    width: w,
                    height: h,
                })
            }

            Message::ImageSaved(result) => {
                self.is_saving = false;
                match result {
                    Ok(_) => {}                      // File saved — no banner needed.
                    Err(e) if e == "cancelled" => {} // User cancelled — not an error.
                    Err(e) => self.error_message = Some(e),
                }
                Task::none()
            }

            Message::TransformSelected(opt) => {
                self.preview_value = self.active_transform_value(&opt);
                self.selected_transform = opt;
                self.is_previewing = false;
                Task::none()
            }

            Message::SliderChanged(val) => {
                self.preview_value = val;
                self.is_previewing = true;
                if !self.has_base_texture() {
                    return Task::none();
                }
                let Some(base_image) = &self.base_image else {
                    return Task::none();
                };
                let (w, h) = base_image.dimensions();
                let preview = Transform {
                    shader_id: self.selected_transform.id,
                    value: val,
                };
                let render_list = build_render_list(&self.history, Some(&preview));
                self.spawn_render(RenderRequest::Preview {
                    render_list,
                    width: w,
                    height: h,
                })
            }

            Message::SliderReleased => {
                if !self.is_previewing {
                    return Task::none();
                }
                let t = Transform {
                    shader_id: self.selected_transform.id,
                    value: self.preview_value,
                };
                self.history.apply(t);
                self.is_previewing = false;
                // preview_value stays at its current position — the slider does not reset.
                if !self.has_base_texture() {
                    return Task::none();
                }
                let Some(base_image) = &self.base_image else {
                    return Task::none();
                };
                let (w, h) = base_image.dimensions();
                let render_list = build_render_list(&self.history, None);
                self.spawn_render(RenderRequest::Preview {
                    render_list,
                    width: w,
                    height: h,
                })
            }

            Message::ToggleParameterless => {
                if !self.has_base_texture() {
                    return Task::none();
                }
                let is_active = self.is_transform_active(&self.selected_transform);
                if is_active {
                    self.history.undo();
                } else {
                    let t = Transform {
                        shader_id: self.selected_transform.id,
                        value: 0.0,
                    };
                    self.history.apply(t);
                }
                let Some(base_image) = &self.base_image else {
                    return Task::none();
                };
                let (w, h) = base_image.dimensions();
                let render_list = build_render_list(&self.history, None);
                self.spawn_render(RenderRequest::Preview {
                    render_list,
                    width: w,
                    height: h,
                })
            }

            Message::Undo => {
                if self.history.undo().is_some() && self.has_base_texture() {
                    self.preview_value = self.active_transform_value(&self.selected_transform);
                    let Some(base_image) = &self.base_image else {
                        return Task::none();
                    };
                    let (w, h) = base_image.dimensions();
                    let render_list = build_render_list(&self.history, None);
                    return self.spawn_render(RenderRequest::Preview {
                        render_list,
                        width: w,
                        height: h,
                    });
                }
                Task::none()
            }

            Message::Redo => {
                if self.history.redo().is_some() && self.has_base_texture() {
                    self.preview_value = self.active_transform_value(&self.selected_transform);
                    let Some(base_image) = &self.base_image else {
                        return Task::none();
                    };
                    let (w, h) = base_image.dimensions();
                    let render_list = build_render_list(&self.history, None);
                    return self.spawn_render(RenderRequest::Preview {
                        render_list,
                        width: w,
                        height: h,
                    });
                }
                Task::none()
            }

            Message::PreviewReady(generation, handle) => {
                match self.scheduler.complete(generation) {
                    CompleteResult::Stale => Task::none(),
                    CompleteResult::Accept(pending) => {
                        self.image_handle = handle;
                        pending
                            .map(|req| self.spawn_render(req))
                            .unwrap_or(Task::none())
                    }
                }
            }

            Message::SaveRenderReady(generation, img) => {
                match self.scheduler.complete(generation) {
                    CompleteResult::Stale => Task::none(),
                    CompleteResult::Accept(pending) => {
                        let pending_task = pending
                            .map(|req| self.spawn_render(req))
                            .unwrap_or(Task::none());

                        let save_task = if let Some(img) = img {
                            Task::perform(
                                async move {
                                    let handle = rfd::AsyncFileDialog::new()
                                        .add_filter(
                                            "Images",
                                            &["png", "jpg", "jpeg", "tif", "tiff"],
                                        )
                                        .save_file()
                                        .await;
                                    match handle {
                                        Some(h) => {
                                            let path = h.path().to_path_buf();
                                            bdip_core::io::save_image(&img, &path)
                                                .map_err(|e| e.to_string())?;
                                            Ok(path)
                                        }
                                        None => Err("cancelled".to_string()),
                                    }
                                },
                                |result: Result<PathBuf, String>| match result {
                                    Ok(path) => Message::ImageSaved(Ok(path)),
                                    Err(e) if e == "cancelled" => {
                                        Message::ImageSaved(Err("cancelled".to_string()))
                                    }
                                    Err(e) => Message::ImageSaved(Err(e)),
                                },
                            )
                        } else {
                            // GPU render failed — reset so the user can retry.
                            self.is_saving = false;
                            Task::none()
                        };

                        Task::batch([pending_task, save_task])
                    }
                }
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
        use iced::keyboard::Event;
        use iced::keyboard::Key;

        iced::keyboard::listen().map(|event| match event {
            Event::KeyPressed { key, modifiers, .. } => {
                // ⌘Z → Undo,  ⌘⇧Z → Redo
                if modifiers.command() && key.as_ref() == Key::Character("z") {
                    if modifiers.shift() {
                        return Message::Redo;
                    } else {
                        return Message::Undo;
                    }
                }
                Message::Noop
            }
            _ => Message::Noop,
        })
    }

    /// Submits a render request to the scheduler and, if no task is already
    /// in-flight, immediately dispatches a `Task::perform` that runs the GPU
    /// pipeline on a background executor. If a task IS already in-flight, the
    /// request is queued for dispatch when the current task completes.
    ///
    /// At most one GPU task is in-flight at a time. Rapid renders (e.g. slider
    /// drags) coalesce: only the most recent queued request is retained.
    fn spawn_render(&mut self, request: RenderRequest) -> Task<Message> {
        // Clone the request so the scheduler can take ownership of the queued
        // copy while we pass the original into the async block if dispatching.
        let request_for_task = request.clone();
        match self.scheduler.request(request) {
            ScheduleResult::Queued => Task::none(),
            ScheduleResult::Dispatch(generation) => {
                let Some(gpu_arc) = self.gpu.clone() else {
                    return Task::none();
                };
                Task::perform(
                    async move {
                        // Lock the GPU state for the duration of this render.
                        // There are no `.await` points inside the lock, so
                        // MutexGuard<GpuState> is never held across a
                        // suspension point — making this future Send.
                        let mut gpu = gpu_arc.lock().unwrap();
                        match request_for_task {
                            RenderRequest::Preview {
                                render_list,
                                width,
                                height,
                            } => {
                                let Some(buf) = execute_render_pipeline(&mut gpu, &render_list)
                                else {
                                    return Message::PreviewReady(generation, None);
                                };
                                // Reborrow as &mut GpuState so the borrow checker
                                // can split the field borrows (engine vs renderer).
                                let gpu_state = &mut *gpu;
                                let engine = &gpu_state.engine;
                                let renderer = &mut gpu_state.renderer;
                                let handle =
                                    presentation_to_handle(renderer, engine, &buf, width, height);
                                Message::PreviewReady(generation, handle)
                            }
                            RenderRequest::Save {
                                render_list,
                                width,
                                height,
                            } => {
                                let Some(buf) = execute_render_pipeline(&mut gpu, &render_list)
                                else {
                                    return Message::SaveRenderReady(generation, None);
                                };
                                let gpu_state = &mut *gpu;
                                let img = gpu_state
                                    .renderer
                                    .download(&gpu_state.engine, &buf, width, height)
                                    .ok();
                                Message::SaveRenderReady(generation, img)
                            }
                        }
                    },
                    |m| m,
                )
            }
        }
    }

    /// Checks if the provided `ShaderOption` represents the most recently applied transformation.
    pub fn is_transform_active(&self, opt: &ShaderOption) -> bool {
        self.history
            .applied_transforms()
            .last()
            .is_some_and(|t| t.shader_id == opt.id)
    }

    /// Returns true if the GPU is initialized and a base image is currently
    /// resident in GPU memory as a texture.
    fn has_base_texture(&self) -> bool {
        self.gpu
            .as_ref()
            .is_some_and(|g| g.lock().unwrap().cached_base_texture.is_some())
    }

    /// Returns the slider value for `opt` by examining the trailing run of the
    /// history. If the last entry in `history` is of type `opt`, returns that
    /// value. Otherwise returns the shader's declared default (0.0 for all
    /// current shaders).
    pub fn active_transform_value(&self, opt: &ShaderOption) -> f32 {
        self.history
            .applied_transforms()
            .last()
            .filter(|t| t.shader_id == opt.id)
            .map(|t| t.value)
            .unwrap_or_else(|| {
                registry_by_id(opt.id)
                    .and_then(|r| match &r.meta.param {
                        ParamKind::Slider { default, .. } => Some(*default),
                        ParamKind::Toggle => None,
                    })
                    .unwrap_or(0.0)
            })
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

/// Collapses adjacent runs of the same transform type, keeping only the last
/// entry in each run.
///
/// Example: `[B(0.3), B(0.7), S(0.5), S(0.3), B(0.1)]`
///       -> `[B(0.7), S(0.3), B(0.1)]`
fn collapse_adjacent(transforms: &[Transform]) -> Vec<Transform> {
    let mut result: Vec<Transform> = Vec::new();
    for t in transforms {
        if let Some(last) = result.last()
            && last.shader_id == t.shader_id
        {
            // Same shader as previous — replace it.
            *result.last_mut().unwrap() = t.clone();
            continue;
        }
        result.push(t.clone());
    }
    result
}

/// Builds the final render list from the committed history and an optional
/// preview transform.
///
/// Collapses adjacent runs of the same type (keeping only the last in each
/// run). If `preview` is `Some`, replaces any trailing entry of the same type
/// with the preview value, or appends the preview if no such trailing entry
/// exists. This mirrors the live-preview semantics of slider drags: the
/// in-progress value overlays the last committed value for the same transform
/// rather than stacking on top of it.
fn build_render_list(history: &HistoryManager, preview: Option<&Transform>) -> Vec<Transform> {
    let committed: Vec<Transform> = history.applied_transforms().to_vec();
    let collapsed = collapse_adjacent(&committed);
    match preview {
        Some(p) => {
            let mut list = collapsed;
            if let Some(last) = list.last()
                && last.shader_id == p.shader_id
            {
                list.pop();
            }
            list.push(p.clone());
            list
        }
        None => collapsed,
    }
}

/// Executes the GPU render pipeline: applies each transform in order starting
/// from `gpu.cached_base_texture`, then calls `Renderer::present` to produce
/// an RGBA16 presentation buffer.
///
/// Returns `None` if `cached_base_texture` is not set (no image has been
/// loaded yet).
fn execute_render_pipeline(
    gpu: &mut GpuState,
    render_list: &[Transform],
) -> Option<bdip_core::wgpu::Buffer> {
    let base = gpu.cached_base_texture.as_ref()?;
    let mut current: Option<bdip_core::wgpu::Texture> = None;
    for t in render_list {
        let new_tex = {
            let src = current.as_ref().unwrap_or(base);
            gpu.renderer.apply(&gpu.engine, src, t)
        };
        current = Some(new_tex);
    }
    let final_tex = current.as_ref().unwrap_or(base);
    Some(gpu.renderer.present(&gpu.engine, final_tex))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapse_adjacent_empty() {
        assert!(collapse_adjacent(&[]).is_empty());
    }

    #[test]
    fn test_collapse_adjacent_no_duplicates() {
        let input = vec![
            Transform {
                shader_id: "brightness",
                value: 0.3,
            },
            Transform {
                shader_id: "saturation",
                value: 0.5,
            },
        ];
        let result = collapse_adjacent(&input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_collapse_adjacent_consecutive_same_type() {
        let input = vec![
            Transform {
                shader_id: "brightness",
                value: 0.3,
            },
            Transform {
                shader_id: "brightness",
                value: 0.7,
            },
            Transform {
                shader_id: "saturation",
                value: 0.5,
            },
            Transform {
                shader_id: "saturation",
                value: 0.3,
            },
            Transform {
                shader_id: "brightness",
                value: 0.1,
            },
        ];
        let result = collapse_adjacent(&input);
        assert_eq!(
            result,
            vec![
                Transform {
                    shader_id: "brightness",
                    value: 0.7
                },
                Transform {
                    shader_id: "saturation",
                    value: 0.3
                },
                Transform {
                    shader_id: "brightness",
                    value: 0.1
                },
            ]
        );
    }

    #[test]
    fn test_collapse_adjacent_single_entry() {
        let input = vec![Transform {
            shader_id: "brightness",
            value: 0.5,
        }];
        assert_eq!(collapse_adjacent(&input), input);
    }

    #[test]
    fn test_collapse_adjacent_all_same_type() {
        let input = vec![
            Transform {
                shader_id: "brightness",
                value: 0.1,
            },
            Transform {
                shader_id: "brightness",
                value: 0.5,
            },
            Transform {
                shader_id: "brightness",
                value: 0.9,
            },
        ];
        assert_eq!(
            collapse_adjacent(&input),
            vec![Transform {
                shader_id: "brightness",
                value: 0.9
            }]
        );
    }

    #[test]
    fn test_slider_value_trailing_run_matches() {
        let (mut app, _) = BdipApp::new(None);
        app.history.apply(Transform {
            shader_id: "brightness",
            value: 0.3,
        });
        app.history.apply(Transform {
            shader_id: "saturation",
            value: 0.5,
        });
        assert_eq!(
            app.active_transform_value(&ShaderOption {
                id: "saturation",
                display_name: "Saturation"
            }),
            0.5
        );
    }

    #[test]
    fn test_slider_value_trailing_run_interrupted() {
        let (mut app, _) = BdipApp::new(None);
        app.history.apply(Transform {
            shader_id: "brightness",
            value: 0.3,
        });
        app.history.apply(Transform {
            shader_id: "saturation",
            value: 0.5,
        });
        assert_eq!(
            app.active_transform_value(&ShaderOption {
                id: "brightness",
                display_name: "Brightness"
            }),
            0.0
        );
    }

    #[test]
    fn test_slider_value_empty_history() {
        let (app, _) = BdipApp::new(None);
        assert_eq!(
            app.active_transform_value(&ShaderOption {
                id: "brightness",
                display_name: "Brightness"
            }),
            0.0
        );
    }

    #[test]
    fn test_slider_value_multiple_trailing_same_type() {
        let (mut app, _) = BdipApp::new(None);
        app.history.apply(Transform {
            shader_id: "brightness",
            value: 0.3,
        });
        app.history.apply(Transform {
            shader_id: "saturation",
            value: 0.2,
        });
        app.history.apply(Transform {
            shader_id: "saturation",
            value: 0.5,
        });
        assert_eq!(
            app.active_transform_value(&ShaderOption {
                id: "saturation",
                display_name: "Saturation"
            }),
            0.5
        );
    }
}
