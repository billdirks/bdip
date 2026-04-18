use std::path::PathBuf;

pub use bdip_core::gpu::shaders::ShaderOption;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Message {
    // I/O
    LoadImagePressed,
    ImageLoaded(Result<(PathBuf, bdip_core::Rgba16Image), String>),
    SaveImagePressed,
    ImageSaved(Result<PathBuf, String>),

    // Transform controls
    TransformSelected(ShaderOption),
    SliderChanged {
        param_index: usize,
        value: f32,
    },
    SliderReleased {
        param_index: usize,
        value: f32,
    },
    ToggleParameterless,

    // History
    Undo,
    Redo,

    // Async render completions
    /// Background preview render completed. The generation counter is used to
    /// discard results from superseded tasks.
    PreviewReady(u64, Option<iced::widget::image::Handle>),
    /// Background 16-bit render completed (for saving). The generation counter
    /// is used to discard results from superseded tasks.
    SaveRenderReady(u64, Option<bdip_core::Rgba16Image>),

    // Error handling
    DismissError,

    // File menu
    ToggleFileMenu,
    CloseFileMenu,
    ExportPipelinePressed,
    PipelineExported(Result<(), String>),

    // Misc
    Noop,
}
