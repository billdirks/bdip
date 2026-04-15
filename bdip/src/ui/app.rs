use bdip_core::gpu::engine::GpuEngine;
use bdip_core::gpu::pipeline::Renderer;
use bdip_core::gpu::texture::upload_texture;
use bdip_core::{HistoryManager, Transformation};
use iced::widget::{column, container, row};
use iced::{Element, Length, Subscription, Task};
use std::path::PathBuf;

use super::canvas;
use super::canvas::presentation_to_handle;
use super::menu_bar;
use super::message::{Message, TransformOption};
use super::sidebar;

// ---------------------------------------------------------------------------
// Scheduling types
// ---------------------------------------------------------------------------

/// Captures everything the background worker needs to replay a render.
#[derive(Debug, Clone)]
enum RenderRequest {
    /// Standard preview or commit render — produces an iced image handle.
    Preview {
        render_list: Vec<Transformation>,
        width: u32,
        height: u32,
    },
    /// Full-resolution 16-bit render for saving.
    Save {
        render_list: Vec<Transformation>,
        width: u32,
        height: u32,
    },
}

/// Returned by [`RenderScheduler::request`].
#[derive(Debug, PartialEq)]
enum ScheduleResult {
    /// No task is in-flight. The caller should dispatch this request and
    /// embed the provided generation counter in the task.
    Dispatch(u64),
    /// A task is already in-flight. The request has been queued for dispatch
    /// once the current task completes.
    Queued,
}

/// Returned by [`RenderScheduler::complete`].
#[derive(Debug)]
enum CompleteResult {
    /// The generation matches the latest dispatch. Contains the pending
    /// request (if any) that should be dispatched next.
    Accept(Option<RenderRequest>),
    /// The generation is stale — a newer task was dispatched after this one.
    /// The caller should discard the result.
    Stale,
}

/// Two-state state machine (idle / in-flight) with a one-slot queue for the
/// most recent pending render request.
///
/// At most one GPU task is in-flight at a time. If a new request arrives while
/// one is in-flight, it replaces the previous pending slot so that only the
/// latest request is executed once the current task completes.
struct RenderScheduler {
    is_rendering: bool,
    pending: Option<RenderRequest>,
    generation: u64,
}

impl RenderScheduler {
    fn new() -> Self {
        RenderScheduler {
            is_rendering: false,
            pending: None,
            generation: 0,
        }
    }

    /// Submit a render request. If idle, transitions to in-flight and returns
    /// `Dispatch` with the new generation. If already in-flight, replaces any
    /// existing pending slot and returns `Queued`.
    fn request(&mut self, req: RenderRequest) -> ScheduleResult {
        if self.is_rendering {
            // Only the latest request is retained; earlier pending values are
            // discarded because they would produce stale intermediate results.
            self.pending = Some(req);
            ScheduleResult::Queued
        } else {
            self.is_rendering = true;
            self.generation += 1;
            ScheduleResult::Dispatch(self.generation)
        }
    }

    /// Signal that a background task with the given generation has completed.
    ///
    /// If the generation matches the current generation, transitions to idle
    /// and returns `Accept` with any pending request. If the generation is
    /// older (stale), returns `Stale` without changing state, because the
    /// in-flight task may still complete in the future.
    fn complete(&mut self, generation_id: u64) -> CompleteResult {
        if generation_id != self.generation {
            return CompleteResult::Stale;
        }
        self.is_rendering = false;
        CompleteResult::Accept(self.pending.take())
    }
}

// ---------------------------------------------------------------------------
// GPU state
// ---------------------------------------------------------------------------

/// Owns all GPU resources shared across render operations. Grouped into a
/// single struct so it can be extracted from `BdipApp` as a unit — a
/// prerequisite for wrapping in `Arc<Mutex<>>` in a later PR to move renders
/// off the UI thread.
struct GpuState {
    engine: GpuEngine,
    renderer: Renderer,
    cached_base_texture: Option<bdip_core::wgpu::Texture>,
}

pub struct BdipApp {
    // Image state
    pub base_image: Option<bdip_core::Rgba16Image>,
    pub image_handle: Option<iced::widget::image::Handle>,

    // GPU state — all three GPU-owned resources live here so they can be
    // moved as a unit in a later refactor (Arc<Mutex<GpuState>>).
    gpu: Option<GpuState>,

    // Transform state
    pub history: HistoryManager,
    pub selected_transform: TransformOption,
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
                    Some(GpuState {
                        engine,
                        renderer,
                        cached_base_texture: None,
                    }),
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
                if let Some(gpu) = &mut self.gpu {
                    let uploaded = upload_texture(&gpu.engine.device, &gpu.engine.queue, &img);
                    let linear = gpu.renderer.ingest(&gpu.engine, &uploaded);
                    let buf = gpu.renderer.present(&gpu.engine, &linear);
                    self.image_handle =
                        presentation_to_handle(&mut gpu.renderer, &gpu.engine, &buf, w, h);
                    gpu.cached_base_texture = Some(linear);
                }
                self.base_image = Some(img);
                self.history.clear();
                self.preview_value = 0.0;
                self.is_previewing = false;
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
                if self.is_saving || self.base_image.is_none() {
                    return Task::none();
                }
                let Some(img) = self.render_to_rgba16() else {
                    return Task::none();
                };
                self.is_saving = true;
                Task::perform(
                    async move {
                        let handle = rfd::AsyncFileDialog::new()
                            .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff"])
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
                if self.has_base_texture() {
                    let preview = make_transform(&self.selected_transform, val);
                    self.image_handle = self.render_to_handle(Some(&preview));
                }
                Task::none()
            }

            Message::SliderReleased => {
                if !self.is_previewing {
                    return Task::none();
                }
                let t = make_transform(&self.selected_transform, self.preview_value);
                self.history.apply(t);
                self.is_previewing = false;
                // preview_value stays at its current position — the slider
                // does not reset.
                if self.has_base_texture() {
                    self.image_handle = self.render_to_handle(None);
                }
                Task::none()
            }

            Message::ToggleParameterless => {
                if !self.has_base_texture() {
                    return Task::none();
                }
                let is_active = self.is_transform_active(&self.selected_transform);
                if is_active {
                    self.history.undo();
                } else {
                    let t = make_transform(&self.selected_transform, 0.0);
                    self.history.apply(t);
                }
                self.image_handle = self.render_to_handle(None);
                Task::none()
            }

            Message::Undo => {
                if self.history.undo().is_some() && self.has_base_texture() {
                    self.preview_value = self.active_transform_value(&self.selected_transform);
                    self.image_handle = self.render_to_handle(None);
                }
                Task::none()
            }

            Message::Redo => {
                if self.history.redo().is_some() && self.has_base_texture() {
                    self.preview_value = self.active_transform_value(&self.selected_transform);
                    self.image_handle = self.render_to_handle(None);
                }
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

    /// Executes the transformation pipeline on the GPU and returns the final presentation buffer
    /// along with the image dimensions.
    fn render_pipeline(
        &mut self,
        preview: Option<&Transformation>,
    ) -> Option<(bdip_core::wgpu::Buffer, u32, u32)> {
        let committed: Vec<Transformation> = self.history.applied_transforms().to_vec();
        let collapsed = collapse_adjacent(&committed);

        // Build the final render list: collapse the committed history, then
        // either replace the trailing entry of the same type as the preview
        // or append the preview.
        let render_list = match preview {
            Some(p) => {
                let preview_kind = TransformOption::from_transformation(p);
                let mut list = collapsed;
                if let Some(last) = list.last()
                    && TransformOption::from_transformation(last) == preview_kind
                {
                    // Replace the trailing entry of the same type.
                    list.pop();
                }
                list.push(p.clone());
                list
            }
            None => collapsed,
        };

        let (w, h) = self.base_image.as_ref()?.dimensions();
        let gpu = self.gpu.as_mut()?;
        let base = gpu.cached_base_texture.as_ref()?;

        let mut current: Option<bdip_core::wgpu::Texture> = None;
        for t in &render_list {
            let new_tex = {
                let src = current.as_ref().unwrap_or(base);
                gpu.renderer.apply(&gpu.engine, src, t)
            };
            current = Some(new_tex);
        }

        let final_tex = current.as_ref().unwrap_or(base);
        let buf = gpu.renderer.present(&gpu.engine, final_tex);
        Some((buf, w, h))
    }

    /// Replays the collapsed transform stack from `cached_base_texture` and
    /// returns an updated iced image handle. If `preview` is `Some`, it is
    /// treated as a tentative value for the currently selected transform type.
    fn render_to_handle(
        &mut self,
        preview: Option<&Transformation>,
    ) -> Option<iced::widget::image::Handle> {
        let (buf, w, h) = self.render_pipeline(preview)?;
        let gpu = self.gpu.as_mut()?;
        canvas::presentation_to_handle(&mut gpu.renderer, &gpu.engine, &buf, w, h)
    }

    /// Renders the committed transform stack from `cached_base_texture` and
    /// returns the result as a 16-bit RGBA image suitable for saving. Uses the
    /// same GPU pipeline as `render_to_handle` but skips the 8-bit conversion.
    fn render_to_rgba16(&mut self) -> Option<bdip_core::Rgba16Image> {
        let (buf, w, h) = self.render_pipeline(None)?;
        let gpu = self.gpu.as_mut()?;
        gpu.renderer.download(&gpu.engine, &buf, w, h).ok()
    }

    /// Checks if the provided `TransformOption` represents the most recently applied transformation.
    pub fn is_transform_active(&self, opt: &TransformOption) -> bool {
        self.history
            .applied_transforms()
            .last()
            .map(|t| TransformOption::from_transformation(t) == *opt)
            .unwrap_or(false)
    }

    /// Returns true if the GPU is initialized and a base image is currently
    /// resident in GPU memory as a texture.
    fn has_base_texture(&self) -> bool {
        self.gpu
            .as_ref()
            .is_some_and(|g| g.cached_base_texture.is_some())
    }

    /// Returns the slider value for `opt` by examining the trailing run of the
    /// history. If the last entry in `history` is of type `opt`, returns that
    /// value. Otherwise returns 0.0 (the type was interrupted by a different
    /// transform, or history is empty).
    pub fn active_transform_value(&self, opt: &TransformOption) -> f32 {
        let Some(last) = self.history.applied_transforms().last() else {
            return 0.0;
        };
        if TransformOption::from_transformation(last) != *opt {
            return 0.0;
        }
        match last {
            Transformation::Brightness(v)
            | Transformation::Saturation(v)
            | Transformation::Contrast(v) => *v,
            Transformation::Grayscale | Transformation::Invert => 0.0,
        }
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

/// Constructs a `Transformation` from a `TransformOption` and a parameter value.
/// For parameterless variants (Grayscale, Invert), the `val` argument is ignored.
fn make_transform(opt: &TransformOption, val: f32) -> Transformation {
    match opt {
        TransformOption::Brightness => Transformation::Brightness(val),
        TransformOption::Saturation => Transformation::Saturation(val),
        TransformOption::Contrast => Transformation::Contrast(val),
        TransformOption::Grayscale => Transformation::Grayscale,
        TransformOption::Invert => Transformation::Invert,
    }
}

/// Collapses adjacent runs of the same transform type, keeping only the last
/// entry in each run.
///
/// Example: `[B(0.3), B(0.7), S(0.5), S(0.3), B(0.1)]`
///       -> `[B(0.7), S(0.3), B(0.1)]`
fn collapse_adjacent(transforms: &[Transformation]) -> Vec<Transformation> {
    let mut result: Vec<Transformation> = Vec::new();
    for t in transforms {
        if let Some(last) = result.last()
            && TransformOption::from_transformation(last) == TransformOption::from_transformation(t)
        {
            // Same type as previous — replace it.
            *result.last_mut().unwrap() = t.clone();
            continue;
        }
        result.push(t.clone());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use bdip_core::Transformation;

    // -----------------------------------------------------------------------
    // RenderScheduler tests
    // -----------------------------------------------------------------------

    fn make_preview_request() -> RenderRequest {
        RenderRequest::Preview {
            render_list: vec![],
            width: 100,
            height: 100,
        }
    }

    #[test]
    fn test_request_when_idle_returns_dispatch() {
        let mut sched = RenderScheduler::new();
        let result = sched.request(make_preview_request());
        assert_eq!(result, ScheduleResult::Dispatch(1));
    }

    #[test]
    fn test_request_when_in_flight_returns_queued() {
        let mut sched = RenderScheduler::new();
        sched.request(make_preview_request());
        let result = sched.request(make_preview_request());
        assert_eq!(result, ScheduleResult::Queued);
    }

    #[test]
    fn test_pending_replaces_previous_pending() {
        let mut sched = RenderScheduler::new();
        sched.request(make_preview_request());
        sched.request(RenderRequest::Preview {
            render_list: vec![Transformation::Brightness(0.3)],
            width: 100,
            height: 100,
        });
        // Third request should overwrite the second.
        sched.request(RenderRequest::Preview {
            render_list: vec![Transformation::Brightness(0.9)],
            width: 100,
            height: 100,
        });
        // Complete generation 1 and check pending carries the third request.
        let CompleteResult::Accept(Some(pending)) = sched.complete(1) else {
            panic!("expected Accept(Some(…))");
        };
        let RenderRequest::Preview { render_list, .. } = pending else {
            panic!("expected Preview variant");
        };
        assert_eq!(render_list, vec![Transformation::Brightness(0.9)]);
    }

    #[test]
    fn test_complete_matching_generation_returns_accept() {
        let mut sched = RenderScheduler::new();
        let ScheduleResult::Dispatch(g) = sched.request(make_preview_request()) else {
            panic!("expected Dispatch");
        };
        let result = sched.complete(g);
        assert!(matches!(result, CompleteResult::Accept(None)));
    }

    #[test]
    fn test_complete_returns_pending_request() {
        let mut sched = RenderScheduler::new();
        let ScheduleResult::Dispatch(g) = sched.request(make_preview_request()) else {
            panic!("expected Dispatch");
        };
        sched.request(make_preview_request());
        let result = sched.complete(g);
        assert!(matches!(result, CompleteResult::Accept(Some(_))));
    }

    #[test]
    fn test_complete_stale_generation_returns_stale() {
        let mut sched = RenderScheduler::new();
        sched.request(make_preview_request()); // gen 1 in-flight
        // Simulate completing an older, never-dispatched generation (0).
        let result = sched.complete(0);
        assert!(matches!(result, CompleteResult::Stale));
        // Scheduler must still consider itself in-flight.
        let result2 = sched.request(make_preview_request());
        assert_eq!(result2, ScheduleResult::Queued);
    }

    #[test]
    fn test_generation_increments_per_dispatch() {
        let mut sched = RenderScheduler::new();
        let ScheduleResult::Dispatch(g1) = sched.request(make_preview_request()) else {
            panic!("expected Dispatch");
        };
        sched.complete(g1);

        let ScheduleResult::Dispatch(g2) = sched.request(make_preview_request()) else {
            panic!("expected Dispatch");
        };
        sched.complete(g2);

        let ScheduleResult::Dispatch(g3) = sched.request(make_preview_request()) else {
            panic!("expected Dispatch");
        };

        assert!(g1 < g2);
        assert!(g2 < g3);
    }

    #[test]
    fn test_collapse_adjacent_empty() {
        assert!(collapse_adjacent(&[]).is_empty());
    }

    #[test]
    fn test_collapse_adjacent_no_duplicates() {
        let input = vec![
            Transformation::Brightness(0.3),
            Transformation::Saturation(0.5),
        ];
        let result = collapse_adjacent(&input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_collapse_adjacent_consecutive_same_type() {
        let input = vec![
            Transformation::Brightness(0.3),
            Transformation::Brightness(0.7),
            Transformation::Saturation(0.5),
            Transformation::Saturation(0.3),
            Transformation::Brightness(0.1),
        ];
        let result = collapse_adjacent(&input);
        assert_eq!(
            result,
            vec![
                Transformation::Brightness(0.7),
                Transformation::Saturation(0.3),
                Transformation::Brightness(0.1),
            ]
        );
    }

    #[test]
    fn test_collapse_adjacent_single_entry() {
        let input = vec![Transformation::Brightness(0.5)];
        assert_eq!(collapse_adjacent(&input), input);
    }

    #[test]
    fn test_collapse_adjacent_all_same_type() {
        let input = vec![
            Transformation::Brightness(0.1),
            Transformation::Brightness(0.5),
            Transformation::Brightness(0.9),
        ];
        assert_eq!(
            collapse_adjacent(&input),
            vec![Transformation::Brightness(0.9)]
        );
    }

    #[test]
    fn test_slider_value_trailing_run_matches() {
        let (mut app, _) = BdipApp::new(None);
        app.history.apply(Transformation::Brightness(0.3));
        app.history.apply(Transformation::Saturation(0.5));
        assert_eq!(
            app.active_transform_value(&TransformOption::Saturation),
            0.5
        );
    }

    #[test]
    fn test_slider_value_trailing_run_interrupted() {
        let (mut app, _) = BdipApp::new(None);
        app.history.apply(Transformation::Brightness(0.3));
        app.history.apply(Transformation::Saturation(0.5));
        assert_eq!(
            app.active_transform_value(&TransformOption::Brightness),
            0.0
        );
    }

    #[test]
    fn test_slider_value_empty_history() {
        let (app, _) = BdipApp::new(None);
        assert_eq!(
            app.active_transform_value(&TransformOption::Brightness),
            0.0
        );
    }

    #[test]
    fn test_slider_value_multiple_trailing_same_type() {
        let (mut app, _) = BdipApp::new(None);
        app.history.apply(Transformation::Brightness(0.3));
        app.history.apply(Transformation::Saturation(0.2));
        app.history.apply(Transformation::Saturation(0.5));
        assert_eq!(
            app.active_transform_value(&TransformOption::Saturation),
            0.5
        );
    }
}
