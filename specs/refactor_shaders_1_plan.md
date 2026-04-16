# Shader Isolation Refactor: Implementation Plan

## Goal

Refactor the shader registration system so that adding a new GPU shader requires
writing only a `.wgsl` file and a single Rust struct with metadata. No editing of
pipeline machinery, UI routing, CLI parsing, or central enums. This uses Option A
(Inventory-Based Auto-Registration) from `specs/refactor_shaders_1.md`.

After this refactor:
- The `Transformation` enum and `TransformOption` enum are eliminated.
- `TransformKind` and all `From<&Transformation>` dispatch are eliminated.
- All match arms in `pipeline.rs`, `app.rs`, `sidebar.rs`, `message.rs`, and
  `main.rs` that dispatch on shader identity are replaced by generic registry
  lookups.
- Adding a shader requires creating one Rust file and one `.wgsl` file. No other
  files are edited.

---

## Deliverables

1. **`TransformShader` trait and `ShaderRegistration` type** in `bdip_core`.
2. **`ShaderMeta` and `ParamKind`** metadata types that the UI, CLI, and pipeline
   all derive behavior from.
3. **`Transform` struct** replacing the `Transformation` enum (`shader_id` + `value`).
4. **Five shader modules** (`brightness`, `saturation`, `contrast`, `grayscale`,
   `invert`) ported to self-registering `TransformShader` impls.
5. **Generic `PipelineCache`** keyed by `&'static str` instead of `TransformKind`.
6. **Generic `Renderer::apply`** that looks up the registration and produces the
   uniform buffer without match arms.
7. **Registry-driven UI** — sidebar, pick list, and slider/toggle routing derived
   from `ShaderMeta` and `ParamKind`.
8. **Registry-driven CLI** — `parse_transform` iterates the registry.
9. **Updated `HistoryManager`**, `collapse_adjacent`, `build_render_list`, and
   `active_transform_value` operating on `Transform` instead of `Transformation`.
10. **Registry integrity test** validating uniqueness of shader IDs and display names.
11. **Updated `specs/adding_a_shader.md`** reflecting the new single-file process.

---

## Solution Architecture

### Registry Infrastructure

```
bdip_core/src/gpu/shaders/
    mod.rs              -- pub mod declarations + trait + registry types + Transform
    brightness.rs       -- BrightnessParams + TransformShader impl + inventory::submit!
    saturation.rs       -- SaturationParams + ...
    contrast.rs         -- ContrastParams + ...
    grayscale.rs        -- GrayscaleParams + ...
    invert.rs           -- InvertParams + ...
```

**Core types** (all in `bdip_core::gpu::shaders`):

```rust
#[derive(Debug, Clone)]
pub struct ShaderMeta {
    pub id: &'static str,
    pub display_name: &'static str,
    pub wgsl_source: &'static str,
    pub param: ParamKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParamKind {
    Slider { min: f32, max: f32, default: f32 },
    Toggle,
}

pub struct ShaderRegistration {
    pub meta: ShaderMeta,
    /// Creates the uniform byte buffer from a parameter value.
    pub make_uniform: fn(f32) -> Vec<u8>,
}

pub trait TransformShader: bytemuck::Pod {
    const META: ShaderMeta;
    fn from_value(value: f32) -> Self;
    fn to_bytes(&self) -> &[u8] { bytemuck::bytes_of(self) }
}
```

**`ShaderRegistration` constructor:**

```rust
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
```

**`inventory` setup** — required in `shaders/mod.rs`:

```rust
inventory::collect!(ShaderRegistration);
```

**Registry lookup helpers** — public functions in `shaders/mod.rs`:

```rust
/// Returns the registration for `id`, or `None` if no shader has that ID.
pub fn registry_by_id(id: &str) -> Option<&'static ShaderRegistration> {
    inventory::iter::<ShaderRegistration>.into_iter().find(|r| r.meta.id == id)
}

/// Returns an iterator over all registered shaders (in linker order).
/// `inventory::iter::<T>` is not a function call — it is a value of type
/// `inventory::iter<T>` that implements `IntoIterator`. The helper wraps
/// it so callers get a standard iterator.
pub fn all_registrations() -> impl Iterator<Item = &'static ShaderRegistration> {
    inventory::iter::<ShaderRegistration>.into_iter()
}
```

### `Transform` (replaces `Transformation`)

Defined in `bdip_core::gpu::shaders`:

```rust
/// A transform instance: which shader + what parameter value.
/// Replaces the old `Transformation` enum.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    pub shader_id: &'static str,
    pub value: f32,               // 0.0 for parameterless shaders
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
            // Fallback for unknown IDs (should not happen in practice;
            // caught by registry integrity tests).
            None => write!(f, "{}: {:.2}", self.shader_id, self.value),
        }
    }
}
```

### Pipeline Dispatch

`PipelineCache` is keyed by `&'static str` (shader ID). On cache miss, it looks up
the `ShaderRegistration` via `registry_by_id`, compiles `meta.wgsl_source` with the
standard bind group contract, and caches the result. `Renderer::apply` takes
`&Transform`, looks up the registration, calls `make_uniform(transform.value)`, and
dispatches. No match arms.

### UI

The sidebar iterates `all_registrations()` to build the pick list.
`ParamKind::Slider` renders a slider with the declared `min`/`max`/`default`.
`ParamKind::Toggle` renders a toggler. No `TransformOption` enum.

**Pick list item type:** iced's `pick_list` widget requires items that implement
`Display + Clone + PartialEq + Eq + Hash`. A lightweight `ShaderOption` type wraps
the static metadata:

```rust
/// Pick-list item for the sidebar transform selector. Built from the
/// shader registry; one per registered shader.
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
```

`ShaderOption` is defined in `bdip_core::gpu::shaders` (in `shaders/mod.rs`,
added in PR 5) so it is available to both the UI and CLI crates. It is
constructed from `ShaderRegistration::meta` fields during sidebar rendering.
`BdipApp::selected_transform` changes from `TransformOption` to `ShaderOption`.
`Message::TransformSelected` carries a `ShaderOption`.

### CLI

`parse_transform` iterates the registry to find a matching `id` string and
constructs a `Transform`. No match arms.

---

## Uniqueness Validation Strategy

The `inventory` crate collects registrations at link time into a flat list. The
compiler cannot enforce that `ShaderMeta::id` or `ShaderMeta::display_name` are
unique across separately compiled modules. Two shaders claiming the same ID would
cause silent incorrect behavior (one pipeline cached for both, wrong uniforms
dispatched).

**Chosen approach: mandatory unit test.** A test in `bdip_core` scans the full
registry and asserts that all `id` and `display_name` values are unique:

```rust
#[test]
fn test_shader_registry_no_duplicate_ids() {
    let mut ids = std::collections::HashSet::new();
    for reg in inventory::iter::<ShaderRegistration> {
        assert!(
            ids.insert(reg.meta.id),
            "Duplicate shader ID: '{}'", reg.meta.id
        );
    }
}

#[test]
fn test_shader_registry_no_duplicate_display_names() {
    let mut names = std::collections::HashSet::new();
    for reg in inventory::iter::<ShaderRegistration> {
        assert!(
            names.insert(reg.meta.display_name),
            "Duplicate display name: '{}'", reg.meta.display_name
        );
    }
}
```

This is the correct strategy for several reasons:

1. **Link-time collections cannot be validated at compile time.** There is no
   `const`-evaluable way to iterate `inventory` items, so compile-time uniqueness
   checks are not possible.
2. **Tests run in CI before merge.** A contributor who introduces a duplicate ID
   gets a clear, actionable test failure — not a runtime panic in production.
3. **Granular tests per AGENTS.md standards.** ID uniqueness and display name
   uniqueness are tested separately, so a failure message immediately identifies
   which constraint was violated.
4. **No runtime cost.** Unlike a `debug_assert!` at startup, the validation runs
   only during testing — zero overhead in release builds.

An alternative considered was a startup-time assertion (`debug_assert!` in
`Renderer::new`). This was rejected because it only fires in debug builds, produces
a panic rather than a test failure, and adds unnecessary startup work. The test
approach is strictly better for an open-source project where CI gates all merges.

---

## Startup Performance Assessment

### How `inventory` Works

The `inventory` crate uses platform-specific linker sections (`.init_array` on
Linux, `__DATA,__mod_init_func` on macOS) to register static constructor functions
that append each `ShaderRegistration` to a global `Vec`. These constructors run
before `main()`. On first iteration, `inventory::iter::<ShaderRegistration>()` reads
this pre-built `Vec` — it is a pointer dereference plus a slice iteration with no
allocation, no locking, and no syscalls.

### Cost Per Registration

Each `inventory::submit!` call adds one entry to a linker section. At program
startup, this translates to one function call per registration that pushes a pointer
onto a `Vec`. The `ShaderRegistration` struct itself is small (two pointers for
`ShaderMeta` fields, one function pointer for `make_uniform` — roughly 80 bytes on
64-bit).

### Projected Startup Impact

| Shader count | Registration cost | Memory overhead | Relative to app startup |
|-------------|------------------|-----------------|------------------------|
| 5 (current) | < 1 us | ~400 bytes | Unmeasurable |
| 100 | ~5-10 us | ~8 KB | Unmeasurable |
| 1,000 | ~50-100 us | ~80 KB | Negligible |

For context, `GpuEngine::new()` (wgpu adapter/device initialization) takes 10-50 ms
on Apple Silicon. The `inventory` registration cost at 1,000 shaders is roughly
1/500th of GPU initialization time.

### Pipeline Compilation (the Actual Startup Concern)

`inventory` registration is not the bottleneck — pipeline compilation is. Each
shader's WGSL source must be compiled into a Metal/Vulkan pipeline on first use.
With the current lazy `PipelineCache`, this cost is amortized: only shaders that the
user actually applies are compiled. At 1,000 registered shaders, if a user applies 5
different shaders in a session, only 5 pipelines are compiled.

If eager compilation were ever needed (e.g., to avoid first-use latency spikes), the
cost would be roughly 1-5 ms per shader (Metal pipeline compilation), or 1-5 seconds
for 1,000 shaders. This is not a concern for the current lazy approach but is noted
for future reference.

### Conclusion

`inventory` registration adds negligible startup overhead at any realistic shader
count. The lazy pipeline cache ensures that the number of registered shaders has no
impact on startup time beyond the microsecond-scale registration cost. No mitigation
is needed.

---

## PR Breakdown

> **Before starting any PR below:** read `specs/refactor_shaders_1.md` (the design
> doc this plan implements) per the AGENTS.md `specs/*goal*` rule. It contains
> the Goal section, the rejected design alternatives, and the constraints
> (zero-touch registration, bind group contract preservation) that this plan
> must satisfy.

### PR 1: Trait and Registry Infrastructure (No Behavioral Changes)

**Goal:** Add the `TransformShader` trait, `ShaderMeta`, `ParamKind`,
`ShaderRegistration`, `Transform`, and the `inventory` dependency. No existing code
is modified — this is purely additive.

**Files created:**
- `bdip_core/src/gpu/shaders/mod.rs` — all the types listed in the Solution
  Architecture section: `ShaderMeta`, `ParamKind`, `ShaderRegistration` (with
  `new::<T>()`), `TransformShader` trait, `Transform` (with `Display` impl),
  `registry_by_id()`, `all_registrations()`, and the `inventory::collect!`
  call. Also contains `#[cfg(test)] mod tests` with the tests listed below.

**Files modified:**
- `bdip_core/Cargo.toml` — add `inventory = "0.3"` to `[dependencies]`. Pin to
  the `0.3` major version so a future `0.4` release cannot silently change the
  `submit!` / `iter` semantics this plan relies on.
- `bdip_core/src/gpu/mod.rs` — add `pub mod shaders;`

**Tests** (in `bdip_core/src/gpu/shaders/mod.rs`):
- `test_transform_equality_same` — two `Transform` values with same `shader_id`
  and `value` are equal.
- `test_transform_inequality_different_id` — two `Transform` values with different
  `shader_id` are not equal.
- `test_transform_inequality_different_value` — same `shader_id`, different `value`.
- `test_transform_display_unknown_id_fallback` — `Transform` with an unregistered
  `shader_id` falls back to `"{shader_id}: {value:.2}"` format (verifies the
  `Display` impl does not panic on unknown IDs).
- `test_registry_by_id_unknown_returns_none` — `registry_by_id("nonexistent")`
  returns `None`.

**Dependencies:** None. This PR can land independently.

---

### PR 2: Port Brightness Shader to New System (Dual-Path Coexistence)

**Goal:** Implement `TransformShader` for brightness and register it via
`inventory::submit!`. The old code path in `pipeline.rs` remains fully functional
and unchanged — this PR only adds new files alongside it to prove the registry works.

**Important coexistence note:** The `BrightnessParams` struct in `pipeline.rs`
(line 14) stays untouched. The new `shaders/brightness.rs` module defines its own
params struct (which can also be named `BrightnessParams` — it lives in a different
module so there is no name collision). The old struct is only removed in PR 4 when
the old dispatch path is deleted.

**Files created:**
- `bdip_core/src/gpu/shaders/brightness.rs` — contains:
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
          id: "brightness",
          display_name: "Brightness",
          // Path is relative to this file's location (shaders/brightness.rs).
          // The .wgsl files live one directory up in gpu/.
          wgsl_source: include_str!("../brightness.wgsl"),
          param: ParamKind::Slider { min: -1.0, max: 1.0, default: 0.0 },
      };

      fn from_value(value: f32) -> Self {
          Self { value, _padding: [0.0; 3] }
      }
  }

  inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<BrightnessParams>());
  ```

**Files modified:**
- `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod brightness;`

**Tests** (in `bdip_core/src/gpu/shaders/mod.rs`):
- `test_brightness_registry_entry_exists` — `registry_by_id("brightness")` returns
  `Some`.
- `test_brightness_registry_metadata` — verify `display_name` is `"Brightness"` and
  `param` is `ParamKind::Slider { min: -1.0, max: 1.0, default: 0.0 }`.
- `test_brightness_make_uniform_known_value` — call `make_uniform(0.5)` and verify
  the returned bytes match `bytemuck::bytes_of(&BrightnessParams { value: 0.5,
  _padding: [0.0; 3] })`.
- `test_transform_display_slider` — `Transform { shader_id: "brightness",
  value: 0.35 }.to_string()` equals `"Brightness: 0.35"`.
- `test_shader_registry_no_duplicate_ids` — (now exercises one real entry).
- `test_shader_registry_no_duplicate_display_names` — (now exercises one real entry).

**Dependencies:** PR 1.

---

### PR 3: Port Remaining Four Shaders

**Goal:** Implement `TransformShader` for saturation, contrast, grayscale, and
invert. All five shaders are now registered in the inventory. The old code path
in `pipeline.rs` still works unchanged.

**Coexistence note:** Same as PR 2 — the old params structs in `pipeline.rs` remain.
Each new module defines its own params struct. They are removed in PR 4.

**Files created** (each follows the same pattern as `brightness.rs` in PR 2):
- `bdip_core/src/gpu/shaders/saturation.rs`
  - `id: "saturation"`, `display_name: "Saturation"`
  - `param: ParamKind::Slider { min: -1.0, max: 1.0, default: 0.0 }`
  - `wgsl_source: include_str!("../saturation.wgsl")`
  - `from_value(value)` → `SaturationParams { value, _padding: [0.0; 3] }`
- `bdip_core/src/gpu/shaders/contrast.rs`
  - `id: "contrast"`, `display_name: "Contrast"`
  - `param: ParamKind::Slider { min: -1.0, max: 1.0, default: 0.0 }`
  - `wgsl_source: include_str!("../contrast.wgsl")`
  - `from_value(value)` → `ContrastParams { value, _padding: [0.0; 3] }`
- `bdip_core/src/gpu/shaders/grayscale.rs`
  - `id: "grayscale"`, `display_name: "Grayscale"`
  - `param: ParamKind::Toggle`
  - `wgsl_source: include_str!("../grayscale.wgsl")`
  - `from_value(_)` → `GrayscaleParams { _unused: [0.0; 4] }`
- `bdip_core/src/gpu/shaders/invert.rs`
  - `id: "invert"`, `display_name: "Invert"`
  - `param: ParamKind::Toggle`
  - `wgsl_source: include_str!("../invert.wgsl")`
  - `from_value(_)` → `InvertParams { _unused: [0.0; 4] }`

**Note on `include_str!` paths:** The `.wgsl` files live in `bdip_core/src/gpu/`
while the shader modules live in `bdip_core/src/gpu/shaders/`. All `include_str!`
paths use `"../<name>.wgsl"` to reach one directory up.

**Files modified:**
- `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod saturation;`,
  `pub mod contrast;`, `pub mod grayscale;`, `pub mod invert;`

**Tests** (in `bdip_core/src/gpu/shaders/mod.rs`):
- For each shader: `test_<name>_registry_entry_exists`,
  `test_<name>_registry_metadata`, `test_<name>_make_uniform_known_value`
- `test_transform_display_toggle` — `Transform { shader_id: "grayscale",
  value: 0.0 }.to_string()` equals `"Grayscale"` (no value shown for toggles).
- `test_shader_registry_no_duplicate_ids` — now validates all five entries.
- `test_shader_registry_no_duplicate_display_names` — now validates all five.

**Dependencies:** PR 2. (Could also depend only on PR 1, but reviewing the full set
is easier after the pattern is established by PR 2.)

---

### PR 4: Generic Pipeline Dispatch

**Goal:** Replace `TransformKind`, the `From<&Transformation>` impl, the
`PipelineCache::compile()` match arms, and the `Renderer::apply()` match arms with
generic registry-driven dispatch. `PipelineCache` is keyed by `&'static str`.

After this PR, the per-shader params structs in `pipeline.rs` (`BrightnessParams`,
`SaturationParams`, `ContrastParams`, `GrayscaleParams`, `InvertParams`) are
deleted — they now live in the shader modules. `TransformKind` and its `From` impl
are deleted.

**Transition strategy for `Renderer::apply` signature:** The public signature of
`Renderer::apply` changes from `&Transformation` to `&Transform`. This breaks
callers outside `pipeline.rs` (`execute_render_pipeline` in `app.rs` and the
headless CLI loop in `main.rs`). To keep the workspace compiling between PR 4 and
PR 5, a temporary bridge conversion is added:

```rust
// In bdip_core/src/gpu/shaders/mod.rs (or transformation.rs):
impl Transform {
    /// Temporary bridge: converts from the legacy `Transformation` enum.
    /// Removed in PR 5 when `Transformation` is deleted.
    pub fn from_legacy(t: &crate::Transformation) -> Self {
        match t {
            crate::Transformation::Brightness(v) => Transform { shader_id: "brightness", value: *v },
            crate::Transformation::Saturation(v) => Transform { shader_id: "saturation", value: *v },
            crate::Transformation::Contrast(v)   => Transform { shader_id: "contrast", value: *v },
            crate::Transformation::Grayscale      => Transform { shader_id: "grayscale", value: 0.0 },
            crate::Transformation::Invert          => Transform { shader_id: "invert", value: 0.0 },
        }
    }
}
```

Callers in `app.rs` and `main.rs` are updated minimally:
- Add `use bdip_core::gpu::shaders::Transform;` to each file's imports.
- Wrap each `renderer.apply(engine, texture, &transformation)` call with
  `renderer.apply(engine, texture, &Transform::from_legacy(&transformation))`.

This keeps the workspace compiling. The `Transformation` enum and `from_legacy`
are both deleted in PR 5.

**Specific changes to `pipeline.rs`:**

1. **Delete** the five per-shader params structs (`BrightnessParams` through
   `InvertParams`). `PresentParams` stays (it is not a transform shader).

2. **Delete** `TransformKind` enum and its `From<&Transformation>` impl.

3. **Rewrite `PipelineCache`:**
   - `cache: HashMap<TransformKind, CachedPipeline>` →
     `cache: HashMap<&'static str, CachedPipeline>`
   - `get_or_create(&mut self, device, kind: TransformKind)` →
     `get_or_create(&mut self, device, shader_id: &'static str)`
   - `compile(device, kind: TransformKind)` →
     `compile(device, shader_id: &'static str)`. The full new body (replaces the
     existing one verbatim — only the label/source derivation changes, the bind
     group layout, pipeline layout, and pipeline creation are byte-identical to
     the current implementation in `pipeline.rs:110-207`):
     ```rust
     fn compile(device: &wgpu::Device, shader_id: &'static str) -> CachedPipeline {
         let reg = registry_by_id(shader_id)
             .unwrap_or_else(|| panic!("Unknown shader ID: '{shader_id}'"));
         let meta = &reg.meta;

         // Labels are `format!`-generated from `meta.display_name` instead of
         // selected by a match arm. These are GPU debug labels; the allocation
         // happens once per shader on first compile, so cost is negligible.
         let shader_label = format!("{} Shader", meta.display_name);
         let pipeline_label = format!("{} Pipeline", meta.display_name);
         let texture_bgl_label = format!("{} Texture BGL", meta.display_name);
         let params_bgl_label = format!("{} Params BGL", meta.display_name);
         let pl_label = format!("{} Pipeline Layout", meta.display_name);

         let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
             label: Some(&shader_label),
             source: wgpu::ShaderSource::Wgsl(meta.wgsl_source.into()),
         });

         let texture_bind_group_layout =
             make_texture_only_bind_group_layout(device, &texture_bgl_label);

         let params_bind_group_layout =
             device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                 label: Some(&params_bgl_label),
                 entries: &[BindGroupLayoutEntry {
                     binding: 0,
                     visibility: ShaderStages::COMPUTE,
                     ty: BindingType::Buffer {
                         ty: wgpu::BufferBindingType::Uniform,
                         has_dynamic_offset: false,
                         min_binding_size: None,
                     },
                     count: None,
                 }],
             });

         let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
             label: Some(&pl_label),
             bind_group_layouts: &[
                 Some(&texture_bind_group_layout),
                 Some(&params_bind_group_layout),
             ],
             immediate_size: 0,
         });

         let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
             label: Some(&pipeline_label),
             layout: Some(&pipeline_layout),
             module: &shader,
             entry_point: Some("main"),
             compilation_options: Default::default(),
             cache: None,
         });

         CachedPipeline {
             pipeline,
             texture_bind_group_layout,
             params_bind_group_layout,
         }
     }
     ```

4. **Rewrite `Renderer::apply`:**
   ```rust
   pub fn apply(
       &mut self,
       engine: &GpuEngine,
       src_texture: &wgpu::Texture,
       transform: &Transform,
   ) -> wgpu::Texture {
       let reg = registry_by_id(transform.shader_id)
           .unwrap_or_else(|| panic!("Unknown shader ID: '{}'", transform.shader_id));
       let cached = self.pipeline_cache.get_or_create(
           &engine.device, transform.shader_id
       );

       // ... dst_texture creation (unchanged) ...

       let uniform_bytes = (reg.make_uniform)(transform.value);
       let params_buffer = engine.device.create_buffer_init(
           &wgpu::util::BufferInitDescriptor {
               label: Some("Apply Params Buffer"),
               contents: &uniform_bytes,
               usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
           }
       );

       // ... bind group creation and dispatch (unchanged) ...
   }
   ```

**Files modified:**
- `bdip_core/src/gpu/pipeline.rs` — all changes described above
- `bdip_core/src/gpu/shaders/mod.rs` — add `Transform::from_legacy` bridge method
- `bdip/src/ui/app.rs` — `execute_render_pipeline` wraps each `renderer.apply` call
  with `Transform::from_legacy`
- `bdip/src/main.rs` — headless CLI loop wraps each `renderer.apply` call with
  `Transform::from_legacy`

**Test-file scope for this PR:** the only test file that needs editing is
`bdip_core/src/gpu/pipeline.rs` (its `#[cfg(test)] mod tests`). The bridge
`Transform::from_legacy` keeps `Transformation` alive, so:
- `bdip_core/src/transformation.rs` tests — **unchanged** (deleted in PR 5).
- `bdip_core/src/history.rs` tests — **unchanged** (deleted/migrated in PR 5).
- `bdip/src/ui/scheduler.rs` tests — **unchanged** (migrated in PR 5).
- `bdip/src/ui/app.rs` tests — **unchanged** (migrated in PR 5).

**Tests** (in `bdip_core/src/gpu/pipeline.rs` only):
- All ~30 existing GPU roundtrip tests updated to use `Transform` values instead
  of `Transformation` variants. Example: `Transformation::Brightness(0.5)`
  becomes `Transform { shader_id: "brightness", value: 0.5 }`. Numeric
  assertions are unchanged — this is a pure refactor with no behavioral change.
  Affected tests include all `test_brightness_*`, `test_saturation_*`,
  `test_contrast_*`, `test_grayscale_*`, `test_invert_*`, chaining tests
  (`test_chained_*`, `test_multiple_same_transform`), and headroom tests.
- Pipeline cache tests updated: `TransformKind::Brightness` →`"brightness"`,
  `TransformKind::Saturation` → `"saturation"` in
  `test_pipeline_cache_returns_same_pipeline` and
  `test_pipeline_cache_different_kinds`.
- `test_perf_gpu_roundtrip_24mp` must still pass with the same thresholds.
- Tiling tests (`test_present_tiling_*`, `test_ingest_*`,
  `test_presentation_buffer_layout`) do not call `Renderer::apply` and need no
  changes.

**Note on `Transformation` re-export:** `Transformation` is re-exported at
`bdip_core::Transformation` (see `bdip_core/src/lib.rs`). The bridge method
references it as `crate::Transformation` from within `bdip_core`, and external
callers (`app.rs`, `main.rs`) keep their existing
`use bdip_core::Transformation;` imports through PR 4.

**Dependencies:** PR 3 (all shaders must be registered before the old dispatch can
be removed).

---

### PR 5: Replace `Transformation` Enum with `Transform`

**Goal:** Delete the `Transformation` enum, `TransformOption` enum, and the
`Transform::from_legacy` bridge. All consumers switch to `Transform` natively.
This is the cascading change across the codebase.

**Specific changes by file:**

1. **`bdip_core/src/transformation.rs`** — Delete the `Transformation` enum, its
   `Display` impl, and all tests. Replace the file contents with a re-export:
   ```rust
   pub use crate::gpu::shaders::Transform;
   ```

2. **`bdip_core/src/lib.rs`** — Currently has `pub use transformation::Transformation;`.
   Change to `pub use transformation::Transform;`. This keeps `bdip_core::Transform`
   accessible at the crate root. All `use bdip_core::Transformation` imports across
   the workspace become `use bdip_core::Transform` (or
   `use bdip_core::gpu::shaders::Transform` directly).

3. **`bdip_core/src/history.rs`** — `HistoryManager` changes
   `applied_transforms: Vec<Transformation>` → `Vec<Transform>` and
   `redo_stack: Vec<Transformation>` → `Vec<Transform>`. The `apply`, `undo`,
   `redo`, `applied_transforms`, and `redo_transforms` methods are unchanged in
   logic — only the type annotations change. The dedup check
   (`self.applied_transforms.last() == Some(&t)`) works because `Transform` derives
   `PartialEq`.

4. **`bdip_core/src/gpu/shaders/mod.rs`** — Delete `Transform::from_legacy`. Add
   the `ShaderOption` struct (see architecture section for full definition with
   derive traits and `Display` impl).

5. **`bdip/src/ui/message.rs`** — Delete `TransformOption` enum entirely (all
   variants, `Display` impl, `from_transformation` method). Change:
   - `TransformSelected(TransformOption)` → `TransformSelected(ShaderOption)`
   - Add `use bdip_core::gpu::shaders::ShaderOption;`
   - All other `Message` variants are unchanged.

6. **`bdip/src/ui/scheduler.rs`** — `RenderRequest::Preview` and
   `RenderRequest::Save` both have `render_list: Vec<Transformation>`. Change to
   `Vec<Transform>`. Update the import from `use bdip_core::Transformation;` to
   `use bdip_core::Transform;`. Update tests: `Transformation::Brightness(0.9)`
   becomes `Transform { shader_id: "brightness", value: 0.9 }`, etc.

7. **`bdip/src/ui/sidebar.rs`** — Delete `TRANSFORM_OPTIONS` constant. Rewrite
   `transform_view`:
   ```rust
   fn transform_view(app: &BdipApp) -> Element<'_, Message> {
       // Build pick list items from registry.
       let options: Vec<ShaderOption> = all_registrations()
           .map(|reg| ShaderOption {
               id: reg.meta.id,
               display_name: reg.meta.display_name,
           })
           .collect();

       let transform_picker = pick_list(
           options,
           Some(app.selected_transform.clone()),
           Message::TransformSelected,
       );

       // Derive control type from the selected shader's ParamKind.
       let selected_reg = registry_by_id(app.selected_transform.id);
       let transform_control: Element<'_, Message> = match selected_reg
           .map(|r| &r.meta.param)
       {
           Some(ParamKind::Slider { min, max, .. }) => {
               let s = slider(*min..=*max, app.preview_value, Message::SliderChanged)
                   .step(0.01)
                   .on_release(Message::SliderReleased);
               let value_label = text(format!("{:.2}", app.preview_value));
               column![s, value_label].spacing(4).into()
           }
           Some(ParamKind::Toggle) | None => {
               let is_active = app.is_transform_active(&app.selected_transform);
               row![
                   text("Apply"),
                   toggler(is_active).on_toggle(|_| Message::ToggleParameterless),
               ]
               .spacing(8)
               .align_y(iced::Alignment::Center)
               .into()
           }
       };

       column![transform_picker, transform_control]
           .spacing(16)
           .padding(8)
           .width(Length::Fill)
           .into()
   }
   ```

8. **`bdip/src/ui/app.rs`** — Several changes:

   - `selected_transform: TransformOption` → `selected_transform: ShaderOption`.
     Initialize to `ShaderOption { id: "brightness", display_name: "Brightness" }`
     in `BdipApp::new`.

   - **Delete `make_transform`** helper. Callers construct `Transform` directly:
     `Transform { shader_id: app.selected_transform.id, value: val }`. For toggles:
     `Transform { shader_id: app.selected_transform.id, value: 0.0 }`.

   - **Rewrite `active_transform_value`:** With `Transform`, this simplifies to:
     ```rust
     pub fn active_transform_value(&self, opt: &ShaderOption) -> f32 {
         self.history.applied_transforms().last()
             .filter(|t| t.shader_id == opt.id)
             .map(|t| t.value)
             .unwrap_or_else(|| {
                 // Return the default from the shader's metadata.
                 registry_by_id(opt.id)
                     .and_then(|r| match &r.meta.param {
                         ParamKind::Slider { default, .. } => Some(*default),
                         ParamKind::Toggle => None,
                     })
                     .unwrap_or(0.0)
             })
     }
     ```

   - **Rewrite `is_transform_active`:**
     ```rust
     pub fn is_transform_active(&self, opt: &ShaderOption) -> bool {
         self.history.applied_transforms().last()
             .is_some_and(|t| t.shader_id == opt.id)
     }
     ```

   - **Rewrite `collapse_adjacent`:** Compare `shader_id` instead of
     `TransformOption::from_transformation`:
     ```rust
     fn collapse_adjacent(transforms: &[Transform]) -> Vec<Transform> {
         let mut result: Vec<Transform> = Vec::new();
         for t in transforms {
             if let Some(last) = result.last()
                 && last.shader_id == t.shader_id
             {
                 *result.last_mut().unwrap() = t.clone();
                 continue;
             }
             result.push(t.clone());
         }
         result
     }
     ```

   - **Rewrite `build_render_list`:** Same logic, compare `shader_id`:
     ```rust
     fn build_render_list(
         history: &HistoryManager,
         preview: Option<&Transform>,
     ) -> Vec<Transform> {
         let committed: Vec<Transform> = history.applied_transforms().to_vec();
         let collapsed = collapse_adjacent(&committed);
         match preview {
             Some(p) => {
                 let mut list = collapsed;
                 if let Some(last) = list.last()
                     && last.shader_id == p.shader_id
                 {
                     list.pop();
                 }
                 list.push(p.clone());
                 list
             }
             None => collapsed,
         }
     }
     ```

   - **Update `execute_render_pipeline`:** Remove `Transform::from_legacy` wrapping
     (the render list is now `Vec<Transform>` natively).

   - **Update all `Message` handlers** that construct transforms: `SliderChanged`,
     `SliderReleased`, `ToggleParameterless` now create `Transform { shader_id:
     self.selected_transform.id, value: ... }` directly.

9. **`bdip/src/main.rs`** — Rewrite `parse_transform`:
   ```rust
   fn parse_transform(s: &str) -> anyhow::Result<Transform> {
       let parts: Vec<&str> = s.split(':').collect();
       let name = parts[0].to_lowercase();

       // Look up the shader by ID in the registry.
       let reg = registry_by_id(&name)
           .ok_or_else(|| anyhow::anyhow!(
               "Unknown transformation: '{}'. Available: {}",
               name,
               all_registrations()
                   .map(|r| r.meta.id)
                   .collect::<Vec<_>>()
                   .join(", ")
           ))?;

       let value = match &reg.meta.param {
           ParamKind::Slider { .. } => {
               if parts.len() != 2 {
                   return Err(anyhow::anyhow!(
                       "{} requires a float value. E.g., {}:0.5",
                       reg.meta.display_name, reg.meta.id
                   ));
               }
               parts[1].parse::<f32>()?
           }
           ParamKind::Toggle => 0.0,
       };

       Ok(Transform { shader_id: reg.meta.id, value })
   }
   ```
   Also update the headless CLI loop to pass `&transform` directly to
   `renderer.apply` (remove `Transform::from_legacy` wrapping added in PR 4).

**Tests:**
- `bdip_core/src/history.rs` tests: update all `Transformation::Brightness(v)`
  to `Transform { shader_id: "brightness", value: v }`, etc.
- `bdip/src/ui/scheduler.rs` tests: update `Transformation::Brightness(0.3)`
  to `Transform { shader_id: "brightness", value: 0.3 }`, etc.
- `bdip/src/ui/app.rs` tests:
  - `collapse_adjacent` tests: same updates.
  - `active_transform_value` tests: `TransformOption::Brightness` →
    `ShaderOption { id: "brightness", display_name: "Brightness" }`, etc.
- Manual verification: load image, apply transforms via slider and toggle,
  undo/redo, save.

**Dependencies:** PR 4.

---

### PR 6: Documentation Update

**Goal:** Update `specs/adding_a_shader.md` to reflect the new single-file process.
Update `specs/tech_debt.md` to mark the "Shader Isolation" entry as resolved.

**Files modified:**
- `specs/adding_a_shader.md` — rewrite to describe the new process: (1) create
  `bdip_core/src/gpu/<name>.wgsl` following the bind group contract, (2) create
  `bdip_core/src/gpu/shaders/<name>.rs` with a params struct, `TransformShader`
  impl, and `inventory::submit!` call, (3) add `pub mod <name>;` to
  `shaders/mod.rs`. That's it — no other files to edit. Include a complete example
  for both parameterized (slider) and parameterless (toggle) shaders.
- `specs/tech_debt.md` — in the "Shader Isolation" entry, change the status to
  indicate it has been resolved and reference this refactor.

**Dependencies:** PR 5.

---

## PR Dependency Graph

```
PR 1  (trait + registry infrastructure)
  |
  v
PR 2  (port brightness)
  |
  v
PR 3  (port remaining 4 shaders)
  |
  v
PR 4  (generic pipeline dispatch — adds from_legacy bridge)
  |
  v
PR 5  (replace Transformation enum — removes from_legacy bridge)
  |
  v
PR 6  (documentation update)
```

All PRs are sequential. Each depends on the previous one. PRs 1-3 are purely
additive (no existing code is changed or broken). PR 4 rewrites `pipeline.rs`
internals and adds a temporary `from_legacy` bridge to keep the workspace
compiling. PR 5 is the cascading change that deletes `Transformation`,
`TransformOption`, and the bridge. PR 6 is documentation only.
