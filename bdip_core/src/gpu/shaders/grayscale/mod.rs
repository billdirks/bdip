use crate::gpu::shaders::{ParamKind, ShaderMeta, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrayscaleParams {
    pub _unused: [f32; 4],
}

impl TransformShader for GrayscaleParams {
    const META: ShaderMeta = ShaderMeta {
        id: "grayscale",
        display_name: "Grayscale",
        wgsl_source: include_str!("grayscale.wgsl"),
        param: ParamKind::Toggle,
    };

    fn from_value(_: f32) -> Self {
        Self { _unused: [0.0; 4] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<
    GrayscaleParams,
>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shaders::{Transform, registry_by_id};

    #[test]
    fn test_grayscale_registry_entry_exists() {
        assert!(registry_by_id("grayscale").is_some());
    }

    #[test]
    fn test_grayscale_registry_metadata() {
        let reg = registry_by_id("grayscale").unwrap();
        assert_eq!(reg.meta.display_name, "Grayscale");
        assert_eq!(reg.meta.param, ParamKind::Toggle);
    }

    #[test]
    fn test_grayscale_make_uniform_known_value() {
        let reg = registry_by_id("grayscale").unwrap();
        let bytes = (reg.make_uniform)(0.0);
        let expected = bytemuck::bytes_of(&GrayscaleParams { _unused: [0.0; 4] });
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_transform_display_toggle() {
        let t = Transform {
            shader_id: "grayscale",
            value: 0.0,
        };
        assert_eq!(t.to_string(), "Grayscale");
    }
}
