use std::path::PathBuf;

/// UI-only enum used for the transform `pick_list`. Maps to
/// `bdip_core::Transformation` variants but carries no parameter values.
/// Contrast/Grayscale/Invert variants are wired to the pick_list in PRs 6–8.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TransformOption {
    Brightness,
    Saturation,
    Contrast,
    Grayscale,
    Invert,
}

impl std::fmt::Display for TransformOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformOption::Brightness => write!(f, "Brightness"),
            TransformOption::Saturation => write!(f, "Saturation"),
            TransformOption::Contrast => write!(f, "Contrast"),
            TransformOption::Grayscale => write!(f, "Grayscale"),
            TransformOption::Invert => write!(f, "Invert"),
        }
    }
}

/// Variants beyond LoadImagePressed/ImageLoaded/TransformSelected/Noop are
/// wired in PRs 2–4.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Message {
    // I/O
    LoadImagePressed,
    ImageLoaded(Result<(PathBuf, bdip_core::Rgba16Image), String>),
    SaveImagePressed,
    ImageSaved(Result<PathBuf, String>),

    // Transform controls
    TransformSelected(TransformOption),
    SliderChanged(f32),
    SliderReleased,
    ApplyParameterless,

    // History
    Undo,
    Redo,

    // Error handling
    DismissError,

    // Misc
    Noop,
}
