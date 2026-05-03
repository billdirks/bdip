pub mod abstract_geometry;
pub mod antique_gold;
pub mod blueprint;
pub mod bokeh_shapes;
pub mod brightness;
pub mod candy_color;
pub mod cartoon;
pub mod chalkboard;
pub mod charcoal_sketch;
pub mod clarity;
pub mod coffee_stained;
pub mod color_lut;
pub mod comic_book;
pub mod console_16bit;
pub mod contrast;
pub mod cross_process;
pub mod cyanotype;
pub mod cyberpunk;
pub mod daguerreotype;
pub mod double_exposure;
pub mod duo_tone;
pub mod emboss;
pub mod exposure;
pub mod fade_1970s;
pub mod film_grain_blue;
pub mod fisheye;
pub mod frost_ice;
pub mod glitch_art;
pub mod golden_hour;
pub mod gouache;
pub mod graffiti;
pub mod grayscale;
pub mod halftone_dots;
pub mod high_key;
pub mod highlights;
pub mod infrared;
pub mod instamatic;
pub mod invert;
pub mod kaleidoscope;
pub mod kodachrome;
pub mod light_leak;
pub mod line_art;
pub mod lomo;
pub mod low_key;
pub mod magnifying_glass;
pub mod mirror_reflection;
pub mod monochrome_green;
pub mod moody_blue;
pub mod mosaic;
pub mod old_map;
pub mod parchment;
pub mod pastel_dreams;
pub mod pastel_punch;
pub mod pencil_sketch;
pub mod pixel_art_8bit;
pub mod pixelate;
pub mod pointillism;
pub mod polaroid;
pub mod polygon;
pub mod pop_art;
pub mod rainbow_flare;
pub mod retro_game_boy;
pub mod retro_newspaper;
pub mod ripple;
pub mod saturation;
pub mod selective_color;
pub mod sepia;
pub mod shadows;
pub mod silhouette;
pub mod sliced_image;
pub mod sparkle;
pub mod stained_glass;
pub mod swirl;
pub mod teal_and_orange;
pub mod technicolor;
pub mod temperature;
pub mod thermal;
pub mod tilt_shift;
pub mod tint;
pub mod tintype;
pub mod tiny_planet;
pub mod underwater;
pub mod vignette;
pub mod vortex;
pub mod watercolor_edge;
pub mod x_ray;

#[cfg(test)]
mod cross_shader_tests;

/// A named slider definition — one entry per adjustable parameter on a shader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliderDef {
    pub name: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub description: &'static str,
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

/// Output texture resolution relative to the source image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassScale {
    /// Same dimensions as the source image.
    Full,
    /// Integer downscale: output is `(source_width / N, source_height / N)`.
    Down(u32),
}

/// Describes an auxiliary texture a pass needs bound at render time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuxTextureDef {
    pub name: &'static str,
    pub dimension: AuxTextureDimension,
    pub filter: AuxSamplerFilter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxTextureDimension {
    D2,
    D3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuxSamplerFilter {
    Linear,
    Nearest,
}

/// Declarative description of one compute pass.
#[derive(Debug, Clone, Copy)]
pub struct PassDef {
    pub label: &'static str,
    pub wgsl_source: &'static str,
    pub inputs: &'static [PassInput],
    pub output: PassOutput,
    pub output_scale: PassScale,
    pub aux_textures: &'static [AuxTextureDef],
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeShaderMeta {
    pub id: &'static str,
    pub display_name: &'static str,
    pub description: &'static str,
    pub passes: &'static [PassDef],
    pub param: ParamKind,
}

/// Type-erased registration entry collected by `inventory` at link time.
pub struct ShaderRegistration {
    pub meta: RuntimeShaderMeta,
    /// Creates the uniform byte buffer from parameter values.
    pub make_uniform: fn(&[f32]) -> Vec<u8>,
}

/// Byte-level equality for `&'static str` values, usable in const context.
const fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Validates the structural integrity of a pass list.
///
/// Invoked by `ShaderRegistration::new::<T>()` in a const context, so violations
/// are reported as build errors. The same function is also called at test time by
/// `test_all_registered_pass_lists_validate` as a tier-2 safety net.
///
/// Rules enforced:
/// 1. `PassOutput::Final` must appear exactly once, on the last pass.
/// 2. Every `PassInput::Scratch(s)` must resolve to a prior `PassOutput::Scratch(s)`.
/// 3. No two passes may declare the same `PassOutput::Scratch(name)`.
pub const fn validate_pass_list(passes: &[PassDef]) {
    let n = passes.len();
    if n == 0 {
        return;
    }

    let mut i = 0;
    while i < n {
        // Final must not appear before the last pass.
        if let PassOutput::Final = passes[i].output
            && i != n - 1
        {
            panic!("validate_pass_list: PassOutput::Final must only appear on the last pass");
        }

        // No two passes may write to the same scratch name.
        if let PassOutput::Scratch(out_name) = passes[i].output {
            let mut j = 0;
            while j < i {
                if let PassOutput::Scratch(prev_name) = passes[j].output
                    && bytes_eq(out_name.as_bytes(), prev_name.as_bytes())
                {
                    panic!(
                        "validate_pass_list: duplicate PassOutput::Scratch name — each scratch name may only be written by one pass"
                    );
                }
                j += 1;
            }
        }

        // Every scratch input must reference a scratch name written by an earlier pass.
        let mut k = 0;
        while k < passes[i].inputs.len() {
            if let PassInput::Scratch(req) = passes[i].inputs[k] {
                let mut found = false;
                let mut j = 0;
                while j < i {
                    if let PassOutput::Scratch(name) = passes[j].output
                        && bytes_eq(req.as_bytes(), name.as_bytes())
                    {
                        found = true;
                        break;
                    }
                    j += 1;
                }
                if !found {
                    panic!(
                        "validate_pass_list: PassInput::Scratch references a name not written by any earlier pass"
                    );
                }
            }
            k += 1;
        }

        i += 1;
    }

    // The last pass must output Final.
    if !matches!(passes[n - 1].output, PassOutput::Final) {
        panic!("validate_pass_list: the last pass must have PassOutput::Final");
    }

    // No auxiliary texture name may collide with a scratch output name.
    let mut ai = 0;
    while ai < n {
        let mut ak = 0;
        while ak < passes[ai].aux_textures.len() {
            let aux_name = passes[ai].aux_textures[ak].name;
            let mut sj = 0;
            while sj < n {
                if let PassOutput::Scratch(scratch_name) = passes[sj].output
                    && bytes_eq(aux_name.as_bytes(), scratch_name.as_bytes())
                {
                    panic!(
                        "validate_pass_list: auxiliary texture name collides with a scratch output name"
                    );
                }
                sj += 1;
            }
            ak += 1;
        }
        ai += 1;
    }
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
                description: T::DESCRIPTION,
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
    const DESCRIPTION: &'static str;
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

    /// Tier-2 safety net: validates every registered shader's pass list at test time.
    ///
    /// Tier 1 (`validate_pass_list` called inside `ShaderRegistration::new::<T>()`) runs
    /// at compile time and covers all shaders registered via the `new` constructor. Tier 2
    /// catches cases tier 1 cannot reach:
    ///
    /// - A `ShaderRegistration` constructed directly (bypassing `new::<T>()`) by writing
    ///   to the public `meta` and `make_uniform` fields. Currently no shader does this, but
    ///   the struct fields are `pub` so it is possible.
    /// - Any future registration path that does not call `validate_pass_list` at compile
    ///   time (e.g., a dynamic registration mechanism added later).
    ///
    /// In the current codebase all shaders go through `new::<T>()`, so tier 1 already
    /// catches every violation. Tier 2 is retained as a low-cost guard against future
    /// registration patterns that might bypass the compile-time check.
    #[test]
    fn test_all_registered_pass_lists_validate() {
        for reg in inventory::iter::<ShaderRegistration> {
            let result = std::panic::catch_unwind(|| validate_pass_list(reg.meta.passes));
            assert!(
                result.is_ok(),
                "validate_pass_list failed for shader '{}': {:?}",
                reg.meta.id,
                result.err()
            );
        }
    }

    /// A pass list where `Final` appears before the last position must be rejected.
    #[test]
    fn test_validate_pass_list_rejects_final_in_middle() {
        const PASSES: &[PassDef] = &[
            PassDef {
                label: "a",
                wgsl_source: "",
                inputs: &[PassInput::Source],
                output: PassOutput::Final, // Final not at last position
                output_scale: PassScale::Full,
                aux_textures: &[],
            },
            PassDef {
                label: "b",
                wgsl_source: "",
                inputs: &[PassInput::Source],
                output: PassOutput::Final,
                output_scale: PassScale::Full,
                aux_textures: &[],
            },
        ];
        let result = std::panic::catch_unwind(|| validate_pass_list(PASSES));
        assert!(
            result.is_err(),
            "expected panic: Final must be the last pass"
        );
    }

    /// A pass list whose first (and only) pass reads a scratch name that was never
    /// written must be rejected.
    #[test]
    fn test_validate_pass_list_rejects_missing_scratch_write() {
        const PASSES: &[PassDef] = &[PassDef {
            label: "a",
            wgsl_source: "",
            inputs: &[PassInput::Scratch("h")], // "h" never written
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
        }];
        let result = std::panic::catch_unwind(|| validate_pass_list(PASSES));
        assert!(result.is_err(), "expected panic: unresolved scratch input");
    }

    /// A pass list where two passes both declare `PassOutput::Scratch` with the same
    /// name must be rejected.
    #[test]
    fn test_validate_pass_list_rejects_duplicate_scratch_output() {
        const PASSES: &[PassDef] = &[
            PassDef {
                label: "a",
                wgsl_source: "",
                inputs: &[PassInput::Source],
                output: PassOutput::Scratch("h"),
                output_scale: PassScale::Full,
                aux_textures: &[],
            },
            PassDef {
                label: "b",
                wgsl_source: "",
                inputs: &[PassInput::Source],
                output: PassOutput::Scratch("h"), // duplicate write
                output_scale: PassScale::Full,
                aux_textures: &[],
            },
            PassDef {
                label: "c",
                wgsl_source: "",
                inputs: &[PassInput::Scratch("h")],
                output: PassOutput::Final,
                output_scale: PassScale::Full,
                aux_textures: &[],
            },
        ];
        let result = std::panic::catch_unwind(|| validate_pass_list(PASSES));
        assert!(
            result.is_err(),
            "expected panic: duplicate scratch output name"
        );
    }

    #[test]
    fn test_validate_pass_list_aux_name_collides_with_scratch() {
        const PASSES: &[PassDef] = &[
            PassDef {
                label: "a",
                wgsl_source: "",
                inputs: &[PassInput::Source],
                output: PassOutput::Scratch("shared"),
                output_scale: PassScale::Full,
                aux_textures: &[],
            },
            PassDef {
                label: "b",
                wgsl_source: "",
                inputs: &[PassInput::Scratch("shared")],
                output: PassOutput::Final,
                output_scale: PassScale::Full,
                aux_textures: &[AuxTextureDef {
                    name: "shared",
                    dimension: AuxTextureDimension::D2,
                    filter: AuxSamplerFilter::Linear,
                }],
            },
        ];
        let result = std::panic::catch_unwind(|| validate_pass_list(PASSES));
        assert!(
            result.is_err(),
            "expected panic: aux name collides with scratch name"
        );
    }

    #[test]
    fn test_all_aux_textures_have_registered_assets() {
        for reg in inventory::iter::<ShaderRegistration> {
            for pass in reg.meta.passes {
                for aux in pass.aux_textures {
                    assert!(
                        crate::gpu::assets::find_asset_by_name(aux.name).is_some(),
                        "Shader '{}', pass '{}': auxiliary texture '{}' \
                         has no registered AuxAssetRegistration",
                        reg.meta.id,
                        pass.label,
                        aux.name,
                    );
                }
            }
        }
    }
}
