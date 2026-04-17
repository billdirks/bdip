pub mod brightness;
pub mod contrast;
pub mod grayscale;
pub mod invert;
pub mod saturation;

#[cfg(test)]
mod cross_shader_tests;

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

fn make_uniform_for<T: TransformShader>(val: f32) -> Vec<u8> {
    bytemuck::bytes_of(&T::from_value(val)).to_vec()
}

impl ShaderRegistration {
    pub const fn new<T: TransformShader>() -> Self {
        Self {
            meta: T::META,
            make_uniform: make_uniform_for::<T>,
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

/// Pick-list item for the sidebar transform selector. Built from the shader
/// registry; one per registered shader.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderOption {
    pub id: &'static str,
    pub display_name: &'static str,
}

impl std::fmt::Display for ShaderOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name)
    }
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

/// Returns all registered shaders sorted alphabetically by display name.
pub fn sorted_registrations() -> Vec<&'static ShaderRegistration> {
    let mut regs: Vec<&'static ShaderRegistration> = all_registrations().collect();
    regs.sort_by_key(|r| r.meta.display_name);
    regs
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

    #[test]
    fn test_shader_registry_no_duplicate_ids() {
        let mut ids = std::collections::HashSet::new();
        for reg in inventory::iter::<ShaderRegistration> {
            assert!(
                ids.insert(reg.meta.id),
                "Duplicate shader ID: '{}'",
                reg.meta.id
            );
        }
    }

    #[test]
    fn test_shader_registry_no_duplicate_display_names() {
        let mut names = std::collections::HashSet::new();
        for reg in inventory::iter::<ShaderRegistration> {
            assert!(
                names.insert(reg.meta.display_name),
                "Duplicate display name: '{}'",
                reg.meta.display_name
            );
        }
    }
}
