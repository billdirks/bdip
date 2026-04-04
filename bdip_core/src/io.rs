use crate::BdipError;
use image::{ImageFormat, RgbaImage};
use std::path::Path;

pub fn load_image<P: AsRef<Path>>(path: P) -> Result<RgbaImage, BdipError> {
    let img = image::open(path).map_err(|e| match e {
        image::ImageError::IoError(io_err) => BdipError::Io(io_err),
        other => BdipError::Image(other),
    })?;
    Ok(img.to_rgba8()) // Convert to Rgba8
}

pub fn save_image<P: AsRef<Path>>(image: &RgbaImage, path: P) -> Result<(), BdipError> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
        .ok_or_else(|| BdipError::UnsupportedFormat("Missing or invalid formatting extension".to_string()))?;

    let format = ImageFormat::from_extension(&extension).ok_or_else(|| {
        BdipError::UnsupportedFormat(format!("Unsupported format for extension: {}", extension))
    })?;

    if format == ImageFormat::Jpeg {
        let rgb_img = image::DynamicImage::ImageRgba8(image.clone()).into_rgb8();
        rgb_img.save_with_format(path, format).map_err(BdipError::Image)
    } else {
        image.save_with_format(path, format).map_err(BdipError::Image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, ImageBuffer};
    use std::fs;

    fn create_test_image() -> RgbaImage {
        ImageBuffer::from_pixel(64, 64, Rgba([255, 0, 0, 255]))
    }

    fn setup_test_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("bdip_test");
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_save_and_load_png() {
        let img = create_test_image();
        let path = setup_test_dir().join("test_save.png");
        
        save_image(&img, &path).unwrap();
        let loaded = load_image(&path).unwrap();
        
        assert_eq!(loaded.dimensions(), (64, 64));
        assert_eq!(loaded.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
    }

    #[test]
    fn test_save_and_load_jpg() {
        let img = create_test_image();
        let path = setup_test_dir().join("test_save.jpg");
        
        save_image(&img, &path).unwrap();
        let loaded = load_image(&path).unwrap();
        
        assert_eq!(loaded.dimensions(), (64, 64));
    }

    #[test]
    fn test_save_unsupported_extension() {
        let img = create_test_image();
        let path = setup_test_dir().join("test_invalid.fake");
        
        let err = save_image(&img, &path).unwrap_err();
        assert!(matches!(err, BdipError::UnsupportedFormat(_)));
    }

    #[test]
    fn test_load_nonexistent_file() {
        let path = setup_test_dir().join("nonexistent_test.png");
        
        let err = load_image(&path).unwrap_err();
        assert!(matches!(err, BdipError::Io(_)));
    }
}
