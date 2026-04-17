# Adding a New Shader

Adding a shader requires two new files and one line in `shaders/mod.rs`. See
`brightness` (slider) and `grayscale` (toggle) in
`bdip_core/src/gpu/shaders/` for complete working examples.

---

## Prerequisites

- The shader operates on linear-light `Rgba16Float` textures using the standard bind
  group contract (see below).
- The shader ID (a short ASCII string like `"hsl_hue"`) must be unique across all
  registered shaders.

## Bind Group Contract

All transform shaders use this two-group layout:

| Group | Binding | Resource |
|-------|---------|----------|
| 0 | 0 | Source texture (`texture_2d<f32>`, read) |
| 0 | 1 | Destination texture (`texture_storage_2d<rgba16float, write>`) |
| 1 | 0 | Uniform buffer (shader-specific params, minimum 16 bytes) |

Parameterless shaders must still declare a 16-byte dummy uniform to satisfy the layout.
This should be defaulted to 0.

---

## Steps

### Step 1 — Create `bdip_core/src/gpu/shaders/<name>/mod.rs`

Define a params struct, implement `TransformShader`, and submit the registration.

**Parameterized shader (slider):**

```rust
use crate::gpu::shaders::{ParamKind, ShaderMeta, SliderDef, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExampleParams {
    pub value: f32,
    pub _padding: [f32; 3],   // pad to 16 bytes for WebGPU uniform alignment
}

impl TransformShader for ExampleParams {
    const META: ShaderMeta = ShaderMeta {
        id: "example",
        display_name: "Example",
        wgsl_source: include_str!("example.wgsl"),
        param: ParamKind::Sliders(&[SliderDef { name: "Amount", min: -1.0, max: 1.0, default: 0.0 }]),
    };

    fn from_values(values: &[f32]) -> Self {
        Self { value: values[0], _padding: [0.0; 3] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ExampleParams>());
```

**Parameterless shader (toggle):**

```rust
use crate::gpu::shaders::{ParamKind, ShaderMeta, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExampleParams {
    pub _unused: [f32; 4],
}

impl TransformShader for ExampleParams {
    const META: ShaderMeta = ShaderMeta {
        id: "example",
        display_name: "Example",
        wgsl_source: include_str!("example.wgsl"),
        param: ParamKind::Toggle,
    };

    fn from_values(_: &[f32]) -> Self {
        Self { _unused: [0.0; 4] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ExampleParams>());
```

### Step 2 — Create `bdip_core/src/gpu/shaders/<name>/<name>.wgsl`

Follow the existing shaders as a pattern. Key requirements:

- Use `@workgroup_size(16, 16)` and entry point `main`.
- Operate in linear color space — inputs are linear-light `rgba16float`.
- Declare a params struct matching your Rust-side uniform (same field layout and
  padding).

**Parameterized shader skeleton:**

```wgsl
struct ExampleParams {
    value: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ExampleParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    let out = vec4<f32>(
        clamp(pixel.rgb + params.value, vec3<f32>(0.0), vec3<f32>(1.0)),
        pixel.a,
    );
    textureStore(output_texture, coord, out);
}
```

**Parameterless shader skeleton:**

```wgsl
struct Params {
    _unused: vec4<f32>,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    let out = vec4<f32>(/* ... */, pixel.a);
    textureStore(output_texture, coord, out);
}
```

### Step 3 — Add `pub mod <name>;` to `shaders/mod.rs`

```rust
pub mod example;
```

### Step 4 — Write tests

Add a `#[cfg(test)] mod tests` block in your `mod.rs`. At minimum, include:

- `test_<name>_registry_entry_exists` — `registry_by_id("<id>")` returns `Some`.
- `test_<name>_registry_metadata` — verify `display_name` and `param` values.
- `test_<name>_make_uniform_known_value` — call `(reg.make_uniform)(&[val])` and assert
  the returned bytes match `bytemuck::bytes_of(&ExampleParams { value: val, .. })`. For
  multi-parameter shaders, pass all values: `(reg.make_uniform)(&[val1, val2, ...])` and
  construct the expected `Params` struct with each field set accordingly.
- **GPU roundtrip tests** covering: identity (no-op value), extreme parameter values,
  alpha preservation, and chaining with an existing shader.

Use the `make_solid_image` + `roundtrip` helpers from the existing tests. Each test
must cover a single isolated behavior (see `AGENTS.md` Unit Testing Standards).

Run `cargo test test_shader_registry_no_duplicate_ids` after adding your shader to
confirm the ID is unique.
