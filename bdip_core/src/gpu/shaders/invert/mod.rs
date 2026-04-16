use crate::gpu::shaders::{ParamKind, ShaderMeta, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InvertParams {
    pub _unused: [f32; 4],
}

impl TransformShader for InvertParams {
    const META: ShaderMeta = ShaderMeta {
        id: "invert",
        display_name: "Invert",
        wgsl_source: include_str!("invert.wgsl"),
        param: ParamKind::Toggle,
    };

    fn from_value(_: f32) -> Self {
        Self { _unused: [0.0; 4] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<InvertParams>());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shaders::registry_by_id;

    #[test]
    fn test_invert_registry_entry_exists() {
        assert!(registry_by_id("invert").is_some());
    }

    #[test]
    fn test_invert_registry_metadata() {
        let reg = registry_by_id("invert").unwrap();
        assert_eq!(reg.meta.display_name, "Invert");
        assert_eq!(reg.meta.param, ParamKind::Toggle);
    }

    #[test]
    fn test_invert_make_uniform_known_value() {
        let reg = registry_by_id("invert").unwrap();
        let bytes = (reg.make_uniform)(0.0);
        let expected = bytemuck::bytes_of(&InvertParams { _unused: [0.0; 4] });
        assert_eq!(bytes, expected);
    }
}
