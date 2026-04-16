/// Metadata that the UI, CLI, and pipeline all read from a registered shader.
#[derive(Debug, Clone)]
pub struct ShaderMeta {
    pub id: &'static str,
    pub display_name: &'static str,
    pub wgsl_source: &'static str,
    pub param: ParamKind,
}

/// Describes what kind of parameter a shader accepts.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamKind {
    Slider { min: f32, max: f32, default: f32 },
    Toggle,
}

/// Type-erased registration entry collected by `inventory` at link time.
pub struct ShaderRegistration {
    pub meta: ShaderMeta,
    /// Creates the uniform byte buffer from a parameter value.
    pub make_uniform: fn(f32) -> Vec<u8>,
}

inventory::collect!(ShaderRegistration);

impl ShaderRegistration {
    pub fn new<T: TransformShader>() -> Self {
        Self {
            meta: T::META,
            make_uniform: |val| {
                let params = T::from_value(val);
                bytemuck::bytes_of(&params).to_vec()
            },
        }
    }
}

/// Implemented by each shader's params struct. The `inventory::submit!` call in
/// each shader module registers it without editing any central list.
pub trait TransformShader: bytemuck::Pod {
    const META: ShaderMeta;
    fn from_value(value: f32) -> Self;
    fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// A transform instance: which shader + what parameter value.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    pub shader_id: &'static str,
    pub value: f32,
}

impl std::fmt::Display for Transform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match registry_by_id(self.shader_id) {
            Some(reg) => match reg.meta.param {
                ParamKind::Slider { .. } => {
                    write!(f, "{}: {:.2}", reg.meta.display_name, self.value)
                }
                ParamKind::Toggle => write!(f, "{}", reg.meta.display_name),
            },
            None => write!(f, "{}: {:.2}", self.shader_id, self.value),
        }
    }
}

/// Returns the registration for `id`, or `None` if no shader has that ID.
pub fn registry_by_id(id: &str) -> Option<&'static ShaderRegistration> {
    inventory::iter::<ShaderRegistration>
        .into_iter()
        .find(|r| r.meta.id == id)
}

/// Returns an iterator over all registered shaders (in linker order).
pub fn all_registrations() -> impl Iterator<Item = &'static ShaderRegistration> {
    inventory::iter::<ShaderRegistration>.into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_display_unknown_id_fallback() {
        let t = Transform {
            shader_id: "nonexistent_shader",
            value: 1.23,
        };
        assert_eq!(t.to_string(), "nonexistent_shader: 1.23");
    }

    #[test]
    fn test_registry_by_id_unknown_returns_none() {
        assert!(registry_by_id("nonexistent").is_none());
    }
}
