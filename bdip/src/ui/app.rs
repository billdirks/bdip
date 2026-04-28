use bdip_core::HistoryManager;
use bdip_core::gpu::engine::GpuEngine;
use bdip_core::gpu::image_pipeline::Renderer;
use bdip_core::gpu::shaders::{
    ParamKind, ShaderOption, Transform, registry_by_id, sorted_registrations,
};
use bdip_core::gpu::texture::upload_texture;
use iced::widget::{Space, column, container, mouse_area, row, rule, stack};
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

/// Tracks an in-progress slider drag: which slider (by index within the shader's
/// `SliderDef` list) and its current live value.
#[derive(Debug, Clone)]
pub struct PreviewSlider {
    pub param_index: usize,
    pub value: f32,
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
    /// Live slider drag state. `Some` while a slider is being dragged;
    /// `None` at rest. Sidebar derives display values from `current_values_for`
    /// when this is `None`.
    pub preview_slider: Option<PreviewSlider>,

    // UI state
    pub error_message: Option<String>,
    pub is_loading: bool,
    pub is_saving: bool,
    pub menu_open: bool,
    pub loaded_path: Option<PathBuf>,
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
                preview_slider: None,
                error_message,
                is_loading: has_input,
                is_saving: false,
                menu_open: false,
                loaded_path: None,
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
                self.menu_open = false;
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
                    Message::ImageLoaded,
                )
            }

            Message::ImageLoaded(Ok((path, img))) => {
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
                self.loaded_path = Some(path);
                self.history.clear();
                self.preview_slider = None;
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
                self.is_loading = false;
                if e != "cancelled" {
                    self.error_message = Some(e);
                }
                Task::none()
            }

            Message::SaveImagePressed => {
                self.menu_open = false;
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
                    Ok(path) => self.loaded_path = Some(path),
                    Err(e) if e == "cancelled" => {} // User cancelled — not an error.
                    Err(e) => self.error_message = Some(e),
                }
                Task::none()
            }

            Message::TransformSelected(opt) => {
                self.selected_transform = opt;
                self.preview_slider = None;
                Task::none()
            }

            Message::SliderChanged { param_index, value } => {
                self.preview_slider = Some(PreviewSlider { param_index, value });
                if !self.has_base_texture() {
                    return Task::none();
                }
                let Some(base_image) = &self.base_image else {
                    return Task::none();
                };
                let (w, h) = base_image.dimensions();
                let preview = self.build_slider_transform(param_index, value);
                let render_list = build_render_list(&self.history, Some(&preview));
                self.spawn_render(RenderRequest::Preview {
                    render_list,
                    width: w,
                    height: h,
                })
            }

            Message::SliderReleased { param_index, value } => {
                if self.preview_slider.is_none() {
                    return Task::none();
                }
                let t = self.build_slider_transform(param_index, value);
                self.history.apply(t);
                self.preview_slider = None;
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
                        values: vec![],
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
                    self.preview_slider = None;
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
                    self.preview_slider = None;
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

            Message::ToggleFileMenu => {
                self.menu_open = !self.menu_open;
                Task::none()
            }

            Message::CloseFileMenu => {
                self.menu_open = false;
                Task::none()
            }

            Message::ExportPipelinePressed => {
                self.menu_open = false;
                let render_list = build_render_list(&self.history, None);
                if render_list.is_empty() {
                    return Task::none();
                }
                let content = render_list
                    .iter()
                    .map(serialize_transform)
                    .collect::<Vec<_>>()
                    .join("\n");
                Task::perform(
                    async move {
                        let handle = rfd::AsyncFileDialog::new()
                            .add_filter("Text", &["txt"])
                            .save_file()
                            .await;
                        match handle {
                            Some(h) => std::fs::write(h.path(), content).map_err(|e| e.to_string()),
                            None => Err("cancelled".to_string()),
                        }
                    },
                    Message::PipelineExported,
                )
            }

            Message::PipelineExported(result) => {
                match result {
                    Ok(()) => {}
                    Err(e) if e == "cancelled" => {}
                    Err(e) => self.error_message = Some(e),
                }
                Task::none()
            }

            Message::Noop => Task::none(),
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        let menu = menu_bar::view(self);
        let sidebar = sidebar::view(self);
        let canvas = canvas::view(self);

        let content_row = row![
            container(sidebar).width(Length::Fixed(250.0)),
            rule::vertical(1),
            container(canvas).width(Length::Fill).height(Length::Fill),
        ]
        .width(Length::Fill)
        .height(Length::Fill);

        let full_app = column![menu, rule::horizontal(1), content_row]
            .width(Length::Fill)
            .height(Length::Fill);

        // When the pulldown is open, layer a full-window dismiss backdrop and the
        // floating pulldown over the entire app. The backdrop covers the menu bar too,
        // so clicking anywhere outside the pulldown (including empty menu bar space)
        // closes it. The pulldown sits on top of the backdrop so its buttons fire first.
        if self.menu_open {
            let backdrop = mouse_area(
                container(Space::new())
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .on_press(Message::CloseFileMenu);
            // Offset the pulldown below the menu bar (height ~27px) + separator (1px).
            let floating_pulldown = container(menu_bar::pulldown(self))
                .width(Length::Fill)
                .height(Length::Fill)
                .align_x(iced::alignment::Horizontal::Left)
                .align_y(iced::alignment::Vertical::Top)
                .padding(iced::Padding::default().top(28));
            stack![full_app, backdrop, floating_pulldown]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            full_app.into()
        }
    }

    pub fn subscription(&self) -> Subscription<Message> {
        use iced::keyboard::Event;
        use iced::keyboard::Key;
        use iced::keyboard::key;

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
                if modifiers.command() && key.as_ref() == Key::Character("o") {
                    return Message::LoadImagePressed;
                }
                if modifiers.command() && key.as_ref() == Key::Character("s") {
                    return Message::SaveImagePressed;
                }
                if modifiers.command() && key.as_ref() == Key::Character("e") {
                    return Message::ExportPipelinePressed;
                }
                if key.as_ref() == Key::Named(key::Named::Escape) {
                    return Message::CloseFileMenu;
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

    /// Builds a `Transform` for the currently selected shader by reading base
    /// values from history and overriding the parameter at `param_index` with
    /// `value`.
    fn build_slider_transform(&self, param_index: usize, value: f32) -> Transform {
        let mut values = registry_by_id(self.selected_transform.id)
            .map(|r| current_values_for(self.selected_transform.id, &self.history, &r.meta.param))
            .unwrap_or_default();
        if param_index < values.len() {
            values[param_index] = value;
        }
        Transform {
            shader_id: self.selected_transform.id,
            values,
        }
    }

    /// Returns true if the GPU is initialized and a base image is currently
    /// resident in GPU memory as a texture.
    fn has_base_texture(&self) -> bool {
        self.gpu
            .as_ref()
            .is_some_and(|g| g.lock().unwrap().cached_base_texture.is_some())
    }
}

/// Returns the current parameter values for a shader by checking the last history entry.
/// If the last entry matches `shader_id`, returns its values; otherwise returns the
/// `SliderDef` defaults from `meta`.
pub(crate) fn current_values_for(
    shader_id: &'static str,
    history: &HistoryManager,
    param: &ParamKind,
) -> Vec<f32> {
    let ParamKind::Sliders(defs) = param else {
        return vec![];
    };
    let defaults = || defs.iter().map(|d| d.default).collect::<Vec<_>>();
    match history.applied_transforms().last() {
        Some(t) if t.shader_id == shader_id => t.values.clone(),
        _ => defaults(),
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

/// Formats an f32 slider value for pipeline export.
///
/// Sliders snap to multiples of `SLIDER_STEP`, so formatting at
/// `SLIDER_DECIMAL_PLACES` produces clean output like `0.43`. However, the
/// stored f32 (e.g. `0.42999997735...`) may not be the exact nearest f32 to
/// that decimal (e.g. `0.43000000745...`) because `SLIDER_STEP` itself isn't
/// representable exactly in binary. When those two f32s differ, we fall back
/// to `to_string()`, which prints the shortest decimal that parses back to
/// the exact original f32 bits (may be `0.42999998`). This guarantees
/// bit-exact round-trip through `parse_transform` at the cost of some ugly
/// values.
fn format_value(v: f32) -> String {
    let clean = {
        let s = format!("{:.*}", super::sidebar::SLIDER_DECIMAL_PLACES as usize, v);
        let s = s.trim_end_matches('0').trim_end_matches('.');
        s.to_string()
    };
    if clean.parse::<f32>() == Ok(v) {
        clean
    } else {
        v.to_string()
    }
}

/// Serializes a `Transform` to the pipeline file line format understood by
/// `bdip --headless --pipeline`.
///
/// Parameterless shaders: `shader_id`
/// Slider shaders: `shader_id:val1:val2:...`
fn serialize_transform(t: &Transform) -> String {
    if t.values.is_empty() {
        t.shader_id.to_string()
    } else {
        let mut s = t.shader_id.to_string();
        for v in &t.values {
            s.push(':');
            s.push_str(&format_value(*v));
        }
        s
    }
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
            match gpu.renderer.apply(&gpu.engine, src, t) {
                Ok(tex) => tex,
                Err(e) => {
                    eprintln!("Failed to apply transform '{}': {e}", t.shader_id);
                    return None;
                }
            }
        };
        current = Some(new_tex);
    }
    let final_tex = current.as_ref().unwrap_or(base);
    Some(gpu.renderer.present(&gpu.engine, final_tex))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bdip_core::gpu::shaders::{ParamKind, RuntimeShaderMeta, SliderDef};

    #[test]
    fn test_serialize_transform_parameterless() {
        let t = Transform {
            shader_id: "grayscale",
            values: vec![],
        };
        assert_eq!(serialize_transform(&t), "grayscale");
    }

    #[test]
    fn test_serialize_transform_single_value() {
        let t = Transform {
            shader_id: "brightness",
            values: vec![0.5],
        };
        assert_eq!(serialize_transform(&t), "brightness:0.5");
    }

    #[test]
    fn test_serialize_transform_multiple_values() {
        // 0.3f32 and -0.7f32 round-trip exactly at SLIDER_DECIMAL_PLACES.
        let t = Transform {
            shader_id: "multi",
            values: vec![0.3f32, -0.7f32],
        };
        assert_eq!(serialize_transform(&t), "multi:0.3:-0.7");
    }

    #[test]
    fn test_format_value_round_trips_to_original_f32() {
        // Every exported value, clean or ugly, must parse back to the exact
        // f32 bits it was serialized from.
        for v in [0.43f32, -0.7, 1.49, 0.0, 43.0 * 0.01, 0.32999998] {
            assert_eq!(
                format_value(v).parse::<f32>().ok(),
                Some(v),
                "round-trip failed for {}",
                v
            );
        }
    }

    #[test]
    fn test_collapse_adjacent_empty() {
        assert!(collapse_adjacent(&[]).is_empty());
    }

    #[test]
    fn test_collapse_adjacent_no_duplicates() {
        let input = vec![
            Transform {
                shader_id: "brightness",
                values: vec![0.3],
            },
            Transform {
                shader_id: "saturation",
                values: vec![0.5],
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
                values: vec![0.3],
            },
            Transform {
                shader_id: "brightness",
                values: vec![0.7],
            },
            Transform {
                shader_id: "saturation",
                values: vec![0.5],
            },
            Transform {
                shader_id: "saturation",
                values: vec![0.3],
            },
            Transform {
                shader_id: "brightness",
                values: vec![0.1],
            },
        ];
        let result = collapse_adjacent(&input);
        assert_eq!(
            result,
            vec![
                Transform {
                    shader_id: "brightness",
                    values: vec![0.7]
                },
                Transform {
                    shader_id: "saturation",
                    values: vec![0.3]
                },
                Transform {
                    shader_id: "brightness",
                    values: vec![0.1]
                },
            ]
        );
    }

    #[test]
    fn test_collapse_adjacent_single_entry() {
        let input = vec![Transform {
            shader_id: "brightness",
            values: vec![0.5],
        }];
        assert_eq!(collapse_adjacent(&input), input);
    }

    #[test]
    fn test_collapse_adjacent_all_same_type() {
        let input = vec![
            Transform {
                shader_id: "brightness",
                values: vec![0.1],
            },
            Transform {
                shader_id: "brightness",
                values: vec![0.5],
            },
            Transform {
                shader_id: "brightness",
                values: vec![0.9],
            },
        ];
        assert_eq!(
            collapse_adjacent(&input),
            vec![Transform {
                shader_id: "brightness",
                values: vec![0.9]
            }]
        );
    }

    fn brightness_meta() -> RuntimeShaderMeta {
        RuntimeShaderMeta {
            id: "brightness",
            display_name: "Brightness",
            description: "",
            passes: &[],
            param: ParamKind::Sliders(&[SliderDef {
                name: "Amount",
                min: -1.0,
                max: 1.0,
                default: 0.0,
                description: "",
            }]),
        }
    }

    fn two_param_meta() -> RuntimeShaderMeta {
        RuntimeShaderMeta {
            id: "test_two",
            display_name: "Test Two",
            description: "",
            passes: &[],
            param: ParamKind::Sliders(&[
                SliderDef {
                    name: "A",
                    min: 0.0,
                    max: 1.0,
                    default: 0.1,
                    description: "",
                },
                SliderDef {
                    name: "B",
                    min: 0.0,
                    max: 1.0,
                    default: 0.2,
                    description: "",
                },
            ]),
        }
    }

    #[test]
    fn test_current_values_for_empty_history_returns_defaults() {
        let history = HistoryManager::new();
        let meta = brightness_meta();
        let vals = current_values_for("brightness", &history, &meta.param);
        assert_eq!(vals, vec![0.0]);
    }

    #[test]
    fn test_current_values_for_last_entry_same_shader_returns_its_values() {
        let mut history = HistoryManager::new();
        history.apply(Transform {
            shader_id: "brightness",
            values: vec![0.4],
        });
        let meta = brightness_meta();
        let vals = current_values_for("brightness", &history, &meta.param);
        assert_eq!(vals, vec![0.4]);
    }

    #[test]
    fn test_current_values_for_last_entry_different_shader_returns_defaults() {
        let mut history = HistoryManager::new();
        history.apply(Transform {
            shader_id: "brightness",
            values: vec![0.4],
        });
        history.apply(Transform {
            shader_id: "saturation",
            values: vec![0.9],
        });
        let meta = brightness_meta();
        let vals = current_values_for("brightness", &history, &meta.param);
        assert_eq!(vals, vec![0.0]);
    }

    #[test]
    fn test_current_values_for_multi_param_empty_history_returns_all_defaults() {
        let history = HistoryManager::new();
        let meta = two_param_meta();
        let vals = current_values_for("test_two", &history, &meta.param);
        assert_eq!(vals, vec![0.1, 0.2]);
    }

    #[test]
    fn test_current_values_for_multi_param_last_entry_same_shader_returns_all_values() {
        let mut history = HistoryManager::new();
        history.apply(Transform {
            shader_id: "test_two",
            values: vec![0.5, 0.7],
        });
        let meta = two_param_meta();
        let vals = current_values_for("test_two", &history, &meta.param);
        assert_eq!(vals, vec![0.5, 0.7]);
    }
}
