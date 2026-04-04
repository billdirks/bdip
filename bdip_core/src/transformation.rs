#[derive(Debug, Clone, PartialEq)]
pub enum Transformation {
    Brightness(f32),    // -1.0 (full dark) to 1.0 (full bright)
    Contrast(f32),      // -1.0 (flat gray) to 1.0 (max contrast)
    Saturation(f32),    // -1.0 (grayscale) to 1.0 (max saturation)
    Grayscale,          // No parameters — converts to luminance
    Invert,             // No parameters — inverts all channels
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
}
