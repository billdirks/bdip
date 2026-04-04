use thiserror::Error;

#[derive(Debug, Error)]
pub enum BdipError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image decoding/encoding error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Invalid transformation parameter: {0}")]
    InvalidParameter(String),
}
