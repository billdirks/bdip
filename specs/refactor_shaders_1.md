# Shader Isolation Refactor: Design Options

## Goal

Enable contributors to add a new GPU shader by writing only a `.wgsl` file and a single
Rust struct with metadata. No editing of pipeline machinery, UI routing, CLI parsing, or
central enums. This is critical for open-source contribution at scale -- when shaders
become the primary extensibility point, contributors should be able to add one without
understanding the rendering pipeline internals.

**Baseline requirement:** `Transformation`, `TransformKind`, and all central dispatch
machinery must be untouched when adding a shader. A shader added without updating a central
dispatch table must produce a compile error or a clearly absent feature -- never a silent
incorrect result or a runtime panic.

---

## Current Pain Points

Adding a shader today requires editing **7 files** with **~12 insertion points** total:

| File | Touch points |
|------|-------------|
| `bdip_core/src/gpu/<name>.wgsl` | New file |
| `bdip_core/src/gpu/pipeline.rs` | Params struct, `TransformKind` variant, `From` impl, `compile()` arm, `apply()` arm |
| `bdip_core/src/transformation.rs` | `Transformation` variant, `Display` arm |
| `bdip/src/main.rs` | `parse_transform` arm |
| `bdip/src/ui/message.rs` | `TransformOption` variant, `Display` arm, `from_transformation` arm |
| `bdip/src/ui/sidebar.rs` | `TRANSFORM_OPTIONS` slice, `transform_view` match arms |
| `bdip/src/ui/app.rs` | `make_transform`, `active_transform_value` arms |

The central problem: `Transformation` is a closed enum that carries both identity and
parameter values. Every new shader forces a variant onto this enum, which cascades through
the entire codebase. At 100+ shaders this is unmaintainable and hostile to contributors.

## Design Principles (shared by all options)

1. **Kill the closed enum.** `Transformation` must become open/extensible or be replaced
   entirely.
2. **Shader = one module.** A contributor writes a `.wgsl` file and a Rust companion, and
   nothing else.
3. **UI is derived, not hand-coded.** Whether a shader gets a slider vs. a toggle, its name
   in the picker, its CLI parse string -- all of this comes from metadata the shader declares,
   not from match arms in UI code.
4. **Performance is non-negotiable.** The pipeline cache, bind group contract, and dispatch
   mechanics stay the same. The refactor only changes how shaders *register* -- not how they
   *run*.

---

## Option A: Inventory-Based Auto-Registration (Recommended)

### Concept

Each shader is a struct that implements a `TransformShader` trait. The
[`inventory`](https://docs.rs/inventory) crate collects all implementations at link time --
no central list to maintain. The `Transformation` enum is replaced by a trait object or a
`TransformId` + params blob.

### Shader Author Experience

A contributor creates one file: `bdip_core/src/gpu/shaders/brightness.rs`

```rust
use crate::gpu::shaders::{TransformShader, ShaderMeta, ParamKind};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrightnessParams {
    pub value: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for BrightnessParams {
    const META: ShaderMeta = ShaderMeta {
        id: "brightness",                     // unique string key
        display_name: "Brightness",
        wgsl_source: include_str!("../brightness.wgsl"),
        param: ParamKind::Slider { min: -1.0, max: 1.0, default: 0.0 },
    };

    fn from_value(value: f32) -> Self {
        Self { value, _padding: [0.0; 3] }
    }

    fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

// This macro call is the *only* registration step.
inventory::submit!(ShaderRegistration::new::<BrightnessParams>());
```

For parameterless shaders:

```rust
impl TransformShader for GrayscaleParams {
    const META: ShaderMeta = ShaderMeta {
        id: "grayscale",
        display_name: "Grayscale",
        wgsl_source: include_str!("../grayscale.wgsl"),
        param: ParamKind::Toggle,
    };

    fn from_value(_: f32) -> Self {
        Self { _unused: [0.0; 4] }
    }
    // ...
}
```

That's it. No other files to edit.

### Core Architecture

```rust
/// Metadata that the UI, CLI, and pipeline all read.
pub struct ShaderMeta {
    pub id: &'static str,          // "brightness", "hsl_hue", etc.
    pub display_name: &'static str,
    pub wgsl_source: &'static str,
    pub param: ParamKind,
}

pub enum ParamKind {
    Slider { min: f32, max: f32, default: f32 },
    Toggle,
    // Future: MultiSlider, ColorPicker, CurvesEditor, etc.
}

/// Type-erased registration entry collected by `inventory`.
pub struct ShaderRegistration {
    pub meta: ShaderMeta,
    /// Creates the uniform bytes from a parameter value.
    pub make_uniform: fn(f32) -> Vec<u8>,
}
```

**`Transformation` replacement:**

```rust
/// A transform instance: which shader + what parameter value.
/// This replaces the old `Transformation` enum entirely.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    pub shader_id: &'static str,  // key into the registry
    pub value: f32,               // 0.0 for parameterless
}
```

**Pipeline dispatch:** `PipelineCache` is keyed by `&'static str` (the shader ID) instead
of `TransformKind`. On first use for a given ID, it looks up the `ShaderRegistration`,
compiles the WGSL source with the standard bind group contract, and caches the result.
`Renderer::apply` looks up the registration, calls `make_uniform(value)` to produce the
params buffer, and dispatches -- all generic, no match arms.

**UI:** The sidebar iterates `inventory::iter::<ShaderRegistration>()` to populate the pick
list. `ParamKind::Slider` renders a slider; `ParamKind::Toggle` renders a toggler. No match
arms in `sidebar.rs`, `app.rs`, or `message.rs`. The `TransformOption` enum is eliminated.

**CLI:** `parse_transform` iterates the registry to find a matching `id` string. No match
arms.

### Performance Impact

Zero. The `inventory` collection happens once at program startup (it's essentially a static
slice built by the linker). Pipeline compilation, bind group creation, and dispatch are
identical to today. The `make_uniform` function pointer adds one indirect call per dispatch
-- negligible compared to GPU work. The perf test passes unchanged.

### Pros
- **True zero-touch registration.** No central list, no file to edit beyond the shader
  module.
- **Scales to 100+ shaders** with no growth in any file except `shaders/mod.rs` (which just
  has `mod` declarations, or can use a glob-import macro).
- **UI, CLI, and pipeline all derive from the same metadata** -- impossible to forget a
  match arm.
- **Compile-time safety**: if you forget `inventory::submit!`, the shader simply doesn't
  appear in the registry. No panic, no runtime crash -- it's just absent.

### Cons
- **New dependency** (`inventory`). It's a widely-used, zero-overhead crate (~100 lines of
  linker section manipulation). But it is a dependency.
- **Type erasure** loses some compile-time guarantees -- e.g., you can't exhaustively match
  on all shader types. In practice this doesn't matter because the system is open by design.
- **`Transform` uses a string ID** instead of a type-safe enum variant. Typos in shader IDs
  are caught at registration time (duplicate check) but not at construction time unless you
  use associated constants.

---

## Option B: Declarative Macro with Central Registry List

### Concept

Similar trait + metadata, but instead of `inventory`, a declarative macro in `shaders/mod.rs`
lists all shader types. The macro expands into the dispatch function, the pick list, and the
CLI parser. Contributors add their shader module and one line to the macro invocation.

### Shader Author Experience

Same as Option A for the per-shader file. Then add one line to `shaders/mod.rs`:

```rust
register_shaders! {
    brightness::BrightnessParams,
    saturation::SaturationParams,
    contrast::ContrastParams,
    grayscale::GrayscaleParams,
    invert::InvertParams,
    // Contributor adds one line here:
    hsl_hue::HslHueParams,
}
```

The macro generates:
- A `dispatch` function that tries each type's `from_transformation` in order
- A `fn all_shaders() -> &[ShaderMeta]` for the UI/CLI
- A `fn make_uniform(id, value) -> Vec<u8>` for the pipeline

### Pros
- **No external dependency.** Pure Rust, no linker tricks.
- **Explicit registration list** -- easy to see all shaders at a glance, easy to reorder
  for UI presentation.
- **Familiar pattern** -- many Rust projects use this approach (e.g., `serde`'s format
  registry).

### Cons
- **Still requires editing one central file** (`shaders/mod.rs`). For open-source
  contribution this means every shader PR touches the same file, causing merge conflicts
  when multiple contributors work in parallel.
- **Macro complexity.** The macro body is non-trivial to write and debug. Contributors who
  want to understand the system must read macro expansion.
- **Slightly worse contributor experience** than Option A -- "add your file AND add one line
  to the registry" vs. "add your file."

---

## Option C: Build-Script Discovery (`.wgsl`-First)

### Concept

A `build.rs` script scans `bdip_core/src/gpu/shaders/` for `.wgsl` files, reads metadata
from a structured comment header in each `.wgsl` file, and generates the Rust params struct,
trait impl, and registration code automatically. Contributors write *only* the WGSL file.

### Shader Author Experience

Create `bdip_core/src/gpu/shaders/brightness.wgsl`:

```wgsl
// @meta id: brightness
// @meta name: Brightness
// @meta param: slider(-1.0, 1.0, 0.0)
// @meta uniform: value f32

struct BrightnessParams {
    value: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}
@group(1) @binding(0) var<uniform> params: BrightnessParams;
// ... rest of shader
```

The build script parses the `@meta` comments and generates the Rust side.

### Pros
- **Ultimate contributor simplicity.** Write one file in one language. No Rust knowledge
  needed.
- **Attractive for shader artists** who know WGSL/GLSL but not Rust.

### Cons
- **Build script complexity.** Parsing WGSL comments, generating Rust code, handling edge
  cases (multi-param shaders, custom types) -- this is a significant maintenance burden.
- **Poor IDE support.** Generated code doesn't exist until `cargo build` runs. No
  autocomplete, no go-to-definition, no type checking on the generated structs until after
  build.
- **Fragile.** A typo in a `@meta` comment produces a build-script error that's harder to
  diagnose than a Rust compiler error.
- **Limited expressiveness.** Complex shaders that need custom param types, validation, or
  multi-buffer bindings are hard to express in comment metadata.
- **Performance risk.** Build scripts run on every `cargo build` invocation. Scanning files
  and generating code adds latency to the development cycle.

---

## Comparative Analysis

| Criterion | A: Inventory | B: Macro Registry | C: Build Script |
|-----------|:-----------:|:-----------------:|:---------------:|
| Files contributor edits | **1** | 2 | **1** |
| External dependencies | 1 (`inventory`) | 0 | 0 |
| Merge conflict risk | **None** | Moderate (central list) | **None** |
| IDE support | Full | Full | Poor |
| Compile-time safety | High | **Highest** (exhaustive) | Low |
| Contributor Rust knowledge needed | Some | Some | **None** |
| Maintenance burden | Low | Medium (macro) | **High** (build script) |
| Expressiveness for complex shaders | **High** | **High** | Low |
| Performance impact on pipeline | **Zero** | **Zero** | **Zero** |
| Performance impact on builds | None | None | Moderate |

## Recommendation

**Option A (Inventory)** is the best fit for an open-source project expecting 100+ shader
contributions. The single-file-per-shader property eliminates merge conflicts entirely --
the most common friction point for OSS contributors. The `inventory` crate is battle-tested,
zero-overhead, and widely used in the Rust ecosystem. The small loss of exhaustive matching
is the right trade for an intentionally open extension system.

Option B is a reasonable fallback if you want to avoid any external dependency. The merge
conflict cost is real but manageable with clear contribution guidelines.

Option C is appealing for non-Rust contributors but the maintenance cost and fragility
outweigh the benefit. You could add it *on top* of Option A later as a convenience layer
that generates the Rust module from WGSL metadata, without needing it as the primary path.

## Migration Path

The refactor can be done incrementally:

1. **Add the trait + registry infrastructure** (new files only, no existing code changes).
2. **Port one shader** (brightness) to the new system, keeping the old code path as
   fallback. Verify `test_perf_gpu_roundtrip_24mp` still passes.
3. **Port remaining 4 shaders.**
4. **Remove the old `Transformation` enum, `TransformKind`, and all the match arms.**
5. **Update `adding_a_shader.md`** to reflect the new single-file process.

Steps 1-3 can coexist with the old code, so there's no big-bang migration.

## Impact on `Transformation` and `TransformOption`

Both the `Transformation` enum and `TransformOption` enum are eliminated. They're replaced
by:

- `Transform { shader_id: &'static str, value: f32 }` -- carries identity + parameter
- `ShaderMeta` -- carries everything the UI and CLI need (name, param kind, etc.)

`HistoryManager`, `collapse_adjacent`, `build_render_list`, and `active_transform_value`
all operate on `Transform` instead of `Transformation`. Since `Transform` has a uniform
shape (id + value), these functions become simpler -- no match arms at all.

`Display` for history entries: `format!("{}: {:.2}", meta.display_name, self.value)` or
just `meta.display_name` for toggles. Looked up from the registry by `shader_id`.
