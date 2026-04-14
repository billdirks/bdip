#[derive(Debug, Clone, PartialEq)]
pub enum Transformation {
    Brightness(f32), // -1.0 (full dark) to 1.0 (full bright)
    Contrast(f32),   // -1.0 (flat gray) to 1.0 (max contrast)
    Saturation(f32), // -1.0 (grayscale) to 1.0 (max saturation)
    Grayscale,       // No parameters — converts to luminance
    Invert,          // No parameters — inverts all channels
}

impl std::fmt::Display for Transformation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Transformation::Brightness(v) => write!(f, "Brightness: {v:.2}"),
            Transformation::Contrast(v) => write!(f, "Contrast: {v:.2}"),
            Transformation::Saturation(v) => write!(f, "Saturation: {v:.2}"),
            Transformation::Grayscale => write!(f, "Grayscale"),
            Transformation::Invert => write!(f, "Invert"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equality_matching_variants() {
        let t1 = Transformation::Brightness(0.5);
        let t2 = Transformation::Brightness(0.5);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_inequality_differing_parameters() {
        let t1 = Transformation::Contrast(0.5);
        let t2 = Transformation::Contrast(0.8);
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_inequality_differing_variants() {
        let t1 = Transformation::Brightness(0.5);
        let t2 = Transformation::Grayscale;
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_display_parameterized_brightness() {
        assert_eq!(
            Transformation::Brightness(0.35).to_string(),
            "Brightness: 0.35"
        );
    }

    #[test]
    fn test_display_parameterized_saturation() {
        assert_eq!(
            Transformation::Saturation(-0.50).to_string(),
            "Saturation: -0.50"
        );
    }

    #[test]
    fn test_display_parameterized_contrast() {
        assert_eq!(Transformation::Contrast(1.0).to_string(), "Contrast: 1.00");
    }

    #[test]
    fn test_display_parameterless_grayscale() {
        assert_eq!(Transformation::Grayscale.to_string(), "Grayscale");
    }

    #[test]
    fn test_display_parameterless_invert() {
        assert_eq!(Transformation::Invert.to_string(), "Invert");
    }
}
