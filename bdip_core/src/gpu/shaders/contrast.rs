use crate::gpu::shaders::{ParamKind, ShaderMeta, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ContrastParams {
    pub value: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for ContrastParams {
    const META: ShaderMeta = ShaderMeta {
        id: "contrast",
        display_name: "Contrast",
        wgsl_source: include_str!("../contrast.wgsl"),
        param: ParamKind::Slider {
            min: -1.0,
            max: 1.0,
            default: 0.0,
        },
    };

    fn from_value(value: f32) -> Self {
        Self {
            value,
            _padding: [0.0; 3],
        }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ContrastParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shaders::registry_by_id;

    #[test]
    fn test_contrast_registry_entry_exists() {
        assert!(registry_by_id("contrast").is_some());
    }

    #[test]
    fn test_contrast_registry_metadata() {
        let reg = registry_by_id("contrast").unwrap();
        assert_eq!(reg.meta.display_name, "Contrast");
        assert_eq!(
            reg.meta.param,
            ParamKind::Slider {
                min: -1.0,
                max: 1.0,
                default: 0.0,
            }
        );
    }

    #[test]
    fn test_contrast_make_uniform_known_value() {
        let reg = registry_by_id("contrast").unwrap();
        let bytes = (reg.make_uniform)(0.5);
        let expected = bytemuck::bytes_of(&ContrastParams {
            value: 0.5,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }
}
