# Adding a New Shader (Current Process)

This document describes the steps required to add a new GPU transform shader
to `bdip_core` under the current architecture. It serves as a practical
checklist and as a baseline for the future refactor described in
`specs/isolating_shaders_plan.md`.

---

## Prerequisites

- The new transformation variant already exists in `Transformation`
  (`bdip_core/src/transformation.rs`), or you are adding it as part of
  this work.
- The shader operates on linear-light `Rgba16Float` textures using the
  standard bind group contract (see below).

## Bind Group Contract

All transform shaders share the same two-group layout:

| Group | Binding | Resource |
|-------|---------|----------|
| 0 | 0 | Source texture (`texture_2d<f32>`, read) |
| 0 | 1 | Destination texture (`texture_storage_2d<rgba16float, write>`) |
| 1 | 0 | Uniform buffer (shader-specific params) |

This layout is enforced by `PipelineCache::compile()` in `pipeline.rs`.
All current shaders (brightness, saturation) follow it. New shaders must
also follow it unless the architecture is changed.

## Step-by-Step Checklist

### 1. Write the WGSL shader file

Create `bdip_core/src/gpu/<name>.wgsl`. Follow the existing pattern in
`brightness.wgsl` or `saturation.wgsl`:

- Declare bindings matching the bind group contract above.
- Define a params struct matching your Rust-side uniform (see step 2).
- Use `@workgroup_size(16, 16)` and an entry point named `main`.
- Operate in linear color space (input textures have already been
  ingested from sRGB).

### 2. Add a uniform params struct (`pipeline.rs`)

Near the top of `bdip_core/src/gpu/pipeline.rs`, add a `#[repr(C)]`
struct deriving `Copy`, `Clone`, `Debug`, `bytemuck::Pod`, and
`bytemuck::Zeroable`. WebGPU uniforms require 16-byte alignment, so
pad to 16 bytes:

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ExampleParams {
    value: f32,
    _padding: [f32; 3],
}
```

### 3. Add a `TransformKind` variant (`pipeline.rs`)

Add the new variant to the `TransformKind` enum and update the
`From<&Transformation>` impl to map from the corresponding
`Transformation` variant:

```rust
enum TransformKind {
    Brightness,
    Saturation,
    Example,       // <-- new
}

impl From<&Transformation> for TransformKind {
    fn from(t: &Transformation) -> Self {
        match t {
            Transformation::Brightness(_) => TransformKind::Brightness,
            Transformation::Saturation(_) => TransformKind::Saturation,
            Transformation::Example(_) => TransformKind::Example, // <-- new
            other => panic!("TransformKind not implemented for {:?}", other),
        }
    }
}
```

### 4. Add a match arm in `PipelineCache::compile()` (`pipeline.rs`)

In the `compile` function's match expression, add an arm that provides
the shader source and label strings:

```rust
TransformKind::Example => (
    include_str!("example.wgsl"),
    "Example Shader",
    "Example Pipeline",
    "Example Texture BGL",
    "Example Params BGL",
    "Example Pipeline Layout",
),
```

No other changes are needed in `compile()` — the bind group layout
creation, pipeline layout, and pipeline compilation are shared.

### 5. Add a match arm in `Renderer::apply()` (`pipeline.rs`)

In the `apply` method's `params_buffer` match, add an arm that packs
your params into a uniform buffer:

```rust
Transformation::Example(val) => {
    let p = ExampleParams {
        value: *val,
        _padding: [0.0; 3],
    };
    engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Apply Params Buffer"),
        contents: bytemuck::cast_slice(&[p]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}
```

### 6. Add the `Transformation` variant (if not already present)

In `bdip_core/src/transformation.rs`, add the new variant to the
`Transformation` enum.

### 7. Add CLI parsing (if applicable)

In the headless CLI (`bdip/src/main.rs`), add a match arm in the
transform parser to accept the new shader name and its parameter.

### 8. Write tests

At minimum, write unit tests for:

- **Identity case** — parameter value `0.0` (or equivalent) produces
  unchanged output.
- **Extreme values** — verify behavior at the parameter range boundaries.
- **Chaining** — applying the new shader in combination with existing
  shaders produces expected results.

---

## Files Modified (Summary)

| File | Change |
|------|--------|
| `bdip_core/src/gpu/<name>.wgsl` | New file |
| `bdip_core/src/gpu/pipeline.rs` | Params struct, `TransformKind` variant, `compile()` arm, `apply()` arm |
| `bdip_core/src/transformation.rs` | `Transformation` variant (if new) |
| `bdip/src/main.rs` | CLI parsing (if applicable) |

## Touch Points in `pipeline.rs`

Adding a shader currently requires modifying **4 locations** in
`pipeline.rs`:

1. Uniform params struct (top of file)
2. `TransformKind` enum + `From` impl
3. `PipelineCache::compile()` match arm
4. `Renderer::apply()` match arm

The future refactor in `specs/isolating_shaders_plan.md` aims to reduce
this to a single registration point per shader.
