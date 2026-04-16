use crate::gpu::shaders::{ParamKind, ShaderMeta, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrightnessParams {
    pub value: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for BrightnessParams {
    const META: ShaderMeta = ShaderMeta {
        id: "brightness",
        display_name: "Brightness",
        wgsl_source: include_str!("../brightness.wgsl"),
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

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    BrightnessParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shaders::{ParamKind, registry_by_id};

    #[test]
    fn test_brightness_registry_entry_exists() {
        assert!(registry_by_id("brightness").is_some());
    }

    #[test]
    fn test_brightness_registry_metadata() {
        let reg = registry_by_id("brightness").unwrap();
        assert_eq!(reg.meta.display_name, "Brightness");
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
    fn test_brightness_make_uniform_known_value() {
        let reg = registry_by_id("brightness").unwrap();
        let bytes = (reg.make_uniform)(0.5);
        let expected = bytemuck::bytes_of(&BrightnessParams {
            value: 0.5,
            _padding: [0.0; 3],
        });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_transform_display_slider() {
        use crate::gpu::shaders::Transform;
        let t = Transform {
            shader_id: "brightness",
            value: 0.35,
        };
        assert_eq!(t.to_string(), "Brightness: 0.35");
    }
}
