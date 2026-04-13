# Isolating Shaders: Trait-Based Transform Registration

## Goal

Refactor the GPU transform pipeline so that adding a new shader requires
creating a single self-contained module (struct + trait impl + `.wgsl`
file) and registering it in one place. Authors should not need to
understand `PipelineCache`, bind group layouts, or dispatch mechanics.

## Current State

The current process for adding a shader is documented in
`specs/adding_a_shader.md`. It requires modifying **4 locations** within
`pipeline.rs` (params struct, `TransformKind` variant, `compile()` arm,
`apply()` arm) plus the `.wgsl` file. While the wgpu plumbing is already
shared and the match arms are small, each new shader increases the size
of `pipeline.rs` and requires the author to read through unrelated
shader definitions to find the insertion points.

The `TransformKind::from()` and `Renderer::apply()` functions also
`panic!` at runtime on unhandled variants. A trait-based approach can
turn these into compile-time errors.

## Motivation

- **Scalability** — As the number of shaders grows (contrast, curves,
  HSL, tone mapping, sharpening, etc.), the match arms in `pipeline.rs`
  become a long list of mechanically similar blocks. Isolating each
  shader into its own module keeps file sizes manageable and diffs
  focused.
- **Discoverability** — A new contributor can look at one existing shader
  module as a template and replicate it without reading the pipeline
  internals.
- **Compile-time safety** — Eliminating the `panic!` paths in
  `TransformKind::from()` and `apply()` means a missing shader
  implementation is caught at build time, not at runtime.

## Constraints

- Ingest and present pipelines are structurally different from transform
  shaders (different bind group layouts, different output types). They
  remain eagerly initialized in `Renderer::new` and are out of scope for
  this refactor.
- All current transform shaders share the same bind group contract
  (group 0: src/dst textures, group 1: uniform buffer). The design
  should accommodate this common case simply while still allowing future
  shaders with different layouts if needed.
- The `Transformation` enum in `transformation.rs` is the public API for
  specifying transforms. It carries parameter values. The refactor should
  not require changing how callers construct `Transformation` values.

## Proposed Design

### Trait: `TransformShader`

```rust
/// Implemented by each shader's params struct. Provides everything
/// the pipeline needs to compile and dispatch the shader.
trait TransformShader: bytemuck::Pod + bytemuck::Zeroable {
    /// WGSL source code (typically via `include_str!`).
    const WGSL_SOURCE: &'static str;

    /// Human-readable name used for wgpu debug labels.
    const NAME: &'static str;

    /// Construct the params value from a `Transformation` variant.
    /// Returns `None` if this shader does not handle the given variant.
    fn from_transformation(t: &Transformation) -> Option<Self>;
}
```

### Per-Shader Module

Each shader lives in its own file under `bdip_core/src/gpu/shaders/`:

```
bdip_core/src/gpu/shaders/
    mod.rs              // re-exports + registration list
    brightness.rs       // BrightnessParams + TransformShader impl
    saturation.rs       // SaturationParams + TransformShader impl
    contrast.rs         // future
```

The `.wgsl` files remain alongside (or move into the `shaders/`
directory — either works since they are included at compile time via
`include_str!`).

Example shader module:

```rust
// bdip_core/src/gpu/shaders/brightness.rs

use crate::transformation::Transformation;
use super::TransformShader;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrightnessParams {
    pub brightness_offset: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for BrightnessParams {
    const WGSL_SOURCE: &'static str = include_str!("../brightness.wgsl");
    const NAME: &'static str = "Brightness";

    fn from_transformation(t: &Transformation) -> Option<Self> {
        match t {
            Transformation::Brightness(val) => Some(Self {
                brightness_offset: *val,
                _padding: [0.0; 3],
            }),
            _ => None,
        }
    }
}
```

### Generic Compilation and Dispatch

`PipelineCache` gains a generic `compile_for<T: TransformShader>()` and
`dispatch<T: TransformShader>()` that replace the current match-based
`compile()` and the params-buffer match in `apply()`. The shared bind
group layout creation, pipeline compilation, and compute pass dispatch
remain in `pipeline.rs` — but they operate on trait-provided data
instead of per-shader match arms.

### Registration

A single function (or macro) in `shaders/mod.rs` maps
`Transformation` variants to their shader implementations. This is the
one place a new shader author needs to add a line:

```rust
pub fn apply_transform(
    renderer: &mut Renderer,
    engine: &GpuEngine,
    src_texture: &wgpu::Texture,
    transformation: &Transformation,
) -> wgpu::Texture {
    // Try each shader in order; first match wins.
    try_apply::<BrightnessParams>(renderer, engine, src_texture, transformation)
        .or_else(|| try_apply::<SaturationParams>(/*...*/))
        .or_else(|| try_apply::<ContrastParams>(/*...*/))  // future
        .expect("no shader registered for transformation")
}
```

An alternative is a procedural or declarative macro that generates the
dispatch from a list of types, turning a missing entry into a
compile-time error when combined with exhaustive matching on
`Transformation`.

### What Changes for Shader Authors

**Before (current process, see `specs/adding_a_shader.md`):**
1. Write `.wgsl` file
2. Add params struct to `pipeline.rs`
3. Add `TransformKind` variant + `From` impl
4. Add `compile()` match arm
5. Add `apply()` match arm

**After:**
1. Write `.wgsl` file
2. Create `shaders/<name>.rs` with params struct + `TransformShader` impl
3. Add one line to the registration list in `shaders/mod.rs`

Steps 2 and 3 are self-contained — the author only needs to look at an
existing shader module as a template.

## Open Questions

- **`TransformKind` elimination** — With `TransformShader`, the
  `TransformKind` enum and its `From` impl may become unnecessary. The
  `PipelineCache` key could use `TypeId` instead. Worth evaluating
  whether this simplifies or complicates the cache lookup.
- **Non-uniform shaders** — Some future shaders (e.g., curves with a
  lookup table, or spatial filters with kernel buffers) may need more
  than a single uniform buffer in group 1. The trait could be extended
  with an optional `create_extra_bindings()` method, or those shaders
  could implement a richer trait variant. Decide when the first such
  shader is needed.
- **WGSL file location** — Keep `.wgsl` files in `gpu/` (current) or
  move them into `gpu/shaders/` alongside their Rust modules? Moving
  them keeps each shader's files together; keeping them preserves
  current paths.

## Implementation Strategy

This refactor is purely internal to `bdip_core::gpu`. The public API
(`Renderer::apply`, `Transformation` enum) does not change. It can be
done in a single PR:

1. Create `bdip_core/src/gpu/shaders/` module with the
   `TransformShader` trait.
2. Move `BrightnessParams` and `SaturationParams` into their own
   modules with trait impls.
3. Refactor `PipelineCache::compile()` to be generic over
   `TransformShader`.
4. Refactor `Renderer::apply()` to dispatch through the registration
   function.
5. Remove `TransformKind` if `TypeId`-based keying works cleanly.
6. Verify all existing tests pass unchanged.

## Priority

Low — the current architecture is functional and the per-shader cost is
small. This becomes worthwhile when the shader count reaches 4-5 or when
a non-engineer (e.g., a shader artist) needs to contribute shaders
without understanding the Rust pipeline machinery. See
`specs/tech_debt.md` for the tracking entry.
