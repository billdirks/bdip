pub mod brightness;
pub mod contrast;
pub mod exposure;
pub mod grayscale;
pub mod highlights;
pub mod invert;
pub mod saturation;
pub mod shadows;
pub mod temperature;
pub mod tint;
pub mod vignette;

#[cfg(test)]
mod cross_shader_tests;

/// A named slider definition — one entry per adjustable parameter on a shader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliderDef {
    pub name: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

/// Describes what kind of parameter a shader accepts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamKind {
    Sliders(&'static [SliderDef]),
    Toggle,
}

/// Which resource a pass reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassInput {
    Source,
    Scratch(&'static str),
}

/// Where a pass writes its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassOutput {
    Scratch(&'static str),
    Final,
}

/// Declarative description of one compute pass.
#[derive(Debug, Clone, Copy)]
pub struct PassDef {
    pub label: &'static str,
    pub wgsl_source: &'static str,
    pub inputs: &'static [PassInput],
    pub output: PassOutput,
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeShaderMeta {
    pub id: &'static str,
    pub display_name: &'static str,
    pub passes: &'static [PassDef],
    pub param: ParamKind,
}

/// Type-erased registration entry collected by `inventory` at link time.
pub struct ShaderRegistration {
    pub meta: RuntimeShaderMeta,
    /// Creates the uniform byte buffer from parameter values.
    pub make_uniform: fn(&[f32]) -> Vec<u8>,
}

pub const fn validate_pass_list(_passes: &[PassDef]) {
    // Implemented in PR 1
}

inventory::collect!(ShaderRegistration);

fn make_uniform_for<T: TransformShader>(values: &[f32]) -> Vec<u8> {
    bytemuck::bytes_of(&T::from_values(values)).to_vec()
}

impl ShaderRegistration {
    pub const fn new<T: TransformShader>() -> Self {
        validate_pass_list(T::PASSES);
        Self {
            meta: RuntimeShaderMeta {
                id: T::ID,
                display_name: T::DISPLAY_NAME,
                passes: T::PASSES,
                param: T::PARAM,
            },
            make_uniform: make_uniform_for::<T>,
        }
    }
}

/// Implemented by each shader's params struct. The `inventory::submit!` call in
/// each shader module registers it without editing any central list.
///
/// Implemented by each shader's params struct. Single-pass shaders provide a
/// one-element `PASSES` slice with `inputs: &[PassInput::Source]` and
/// `output: PassOutput::Final`. Multi-pass shaders provide the full ordered pass list.
pub trait TransformShader: bytemuck::Pod {
    const ID: &'static str;
    const DISPLAY_NAME: &'static str;
    const PARAM: ParamKind;
    const PASSES: &'static [PassDef];

    fn from_values(values: &[f32]) -> Self;
    fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// A transform instance: which shader + what parameter values.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    pub shader_id: &'static str,
    pub values: Vec<f32>,
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
        let format_values = |values: &[f32]| -> String {
            values
                .iter()
                .map(|v| format!("{v:.2}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        match registry_by_id(self.shader_id) {
            Some(reg) => match reg.meta.param {
                ParamKind::Sliders(_) => {
                    write!(
                        f,
                        "{}: {}",
                        reg.meta.display_name,
                        format_values(&self.values)
                    )
                }
                ParamKind::Toggle => write!(f, "{}", reg.meta.display_name),
            },
            None => write!(f, "{}: {}", self.shader_id, format_values(&self.values)),
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
            values: vec![1.23],
        };
        assert_eq!(t.to_string(), "nonexistent_shader: 1.23");
    }

    #[test]
    fn test_transform_display_multi_value_format() {
        let t = Transform {
            shader_id: "nonexistent_shader",
            values: vec![0.50, 0.40],
        };
        assert_eq!(t.to_string(), "nonexistent_shader: 0.50, 0.40");
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
