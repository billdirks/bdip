use std::path::PathBuf;

/// UI-only enum used for the transform `pick_list`. Maps to
/// `bdip_core::Transformation` variants but carries no parameter values.
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

impl TransformOption {
    /// Returns the `TransformOption` that corresponds to a `Transformation`.
    pub fn from_transformation(t: &bdip_core::Transformation) -> Self {
        match t {
            bdip_core::Transformation::Brightness(_) => Self::Brightness,
            bdip_core::Transformation::Saturation(_) => Self::Saturation,
            bdip_core::Transformation::Contrast(_) => Self::Contrast,
            bdip_core::Transformation::Grayscale => Self::Grayscale,
            bdip_core::Transformation::Invert => Self::Invert,
        }
    }
}

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
    ToggleParameterless,

    // History
    Undo,
    Redo,

    // Error handling
    DismissError,

    // Misc
    Noop,
}
