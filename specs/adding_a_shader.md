# Adding a New Shader

Adding a single-pass shader requires two new files and one line in `shaders/mod.rs`.
See `brightness` (slider) and `grayscale` (toggle) for complete working examples.
Multi-pass shaders follow the same pattern with additional `.wgsl` files; see
§ "Multi-pass shaders" below.

---

## Prerequisites

- The shader operates on linear-light `Rgba16Float` textures.
- The shader ID (a short ASCII string like `"hsl_hue"`) must be unique across all
  registered shaders.

---

## Single-pass shaders

### Step 1 — Create `bdip_core/src/gpu/shaders/<name>/mod.rs`

Define a params struct, implement `TransformShader`, and submit the registration.

**Parameterized shader (one or more sliders):**

```rust
use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, SliderDef, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExampleParams {
    pub value: f32,
    pub _padding: [f32; 3], // pad to 16 bytes for WebGPU uniform alignment
}

impl TransformShader for ExampleParams {
    const ID: &'static str = "example";
    const DISPLAY_NAME: &'static str = "Example";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Amount",
        min: -1.0,
        max: 1.0,
        default: 0.0,
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "example",
        wgsl_source: include_str!("example.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
    }];

    fn from_values(values: &[f32]) -> Self {
        Self { value: values[0], _padding: [0.0; 3] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ExampleParams>());
```

**Parameterless shader (toggle):**

```rust
use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExampleParams {
    pub _unused: [f32; 4],
}

impl TransformShader for ExampleParams {
    const ID: &'static str = "example";
    const DISPLAY_NAME: &'static str = "Example";
    const PARAM: ParamKind = ParamKind::Toggle;
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "example",
        wgsl_source: include_str!("example.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
    }];

    fn from_values(_: &[f32]) -> Self {
        Self { _unused: [0.0; 4] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ExampleParams>());
```

Each index in `from_values` corresponds to the `SliderDef` at the same index in
`PARAM`. The sidebar renders one labelled slider row per entry.

### Step 2 — Create `bdip_core/src/gpu/shaders/<name>/<name>.wgsl`

Single-pass shaders use the two-group layout:

| Group | Binding | Resource |
|-------|---------|----------|
| 0 | 0 | Source texture (`texture_2d<f32>`, read) |
| 0 | 1 | Destination texture (`texture_storage_2d<rgba16float, write>`) |
| 1 | 0 | Uniform buffer (shader params, minimum 16 bytes) |

Parameterless shaders must still declare a 16-byte dummy uniform.

**Parameterized shader skeleton:**

```wgsl
struct ExampleParams {
    value:    f32,
    _padding: vec3<f32>,
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

### Step 3 — Add `pub mod <name>;` to `shaders/mod.rs`

```rust
pub mod example;
```

### Step 4 — Write tests

Add a `#[cfg(test)] mod tests` block in your `mod.rs`. At minimum, include:

- `test_<name>_registry_entry_exists` — `registry_by_id("<id>")` returns `Some`.
- `test_<name>_registry_metadata` — verify `display_name`, `param`, and `passes.len()`.
- `test_<name>_make_uniform_known_value` — call `(reg.make_uniform)(&[val])` and assert
  the returned bytes match `bytemuck::bytes_of(&ExampleParams { value: val, .. })`.
- **GPU roundtrip tests** covering: identity (no-op value), extreme parameter values,
  alpha preservation, and chaining with an existing shader.

Use the `make_solid_image` + `roundtrip` helpers from `gpu::test_util`. Each test
must cover a single isolated behavior (see `AGENTS.md` § "Unit Testing Standards").

Run `cargo test test_shader_registry_no_duplicate_ids` after adding your shader to
confirm the ID is unique.

---

## Multi-pass shaders

A multi-pass shader is a single user-facing `Transform` that runs N sequential
compute dispatches internally. From the engine's perspective every shader is a pass
list — single-pass shaders are simply the length-1 case.

### Directory structure

```
bdip_core/src/gpu/shaders/<name>/
    mod.rs          # params struct + TransformShader impl + registration
    <name>_pass0.wgsl
    <name>_pass1.wgsl
    ...
```

### Bind-group contract (position-indexed)

Each pass's bind groups are derived entirely from its declared `inputs` slice:

| Group | Binding | Resource |
|-------|---------|----------|
| 0 | 0 … N-1 | Input textures in declared order (N = `inputs.len()`) |
| 0 | N | Destination storage texture (`rgba16float, write`) |
| 1 | 0 | Uniform buffer (same params struct for all passes) |

For a 1-input pass (all existing single-pass shaders), N = 1, so binding 0 is the
source and binding 1 is the destination — identical to the single-pass layout above.
For a 3-input pass, bindings 0–2 are inputs and binding 3 is the destination.

### Shared-uniform alignment rule

All passes in one shader share a single uniform buffer built from the params struct.
Every `.wgsl` file in the shader must declare the **full, identical** params struct,
even if a particular pass reads only some fields. WebGPU validates the uniform binding
size against the pipeline layout at creation time; a truncated struct produces a
byte-mismatch error.

### Implementing `PASSES`

```rust
use crate::gpu::shaders::{ParamKind, PassDef, PassInput, PassOutput, SliderDef, TransformShader};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TwoPassParams {
    pub amount: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for TwoPassParams {
    const ID: &'static str = "two_pass_example";
    const DISPLAY_NAME: &'static str = "Two-Pass Example";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Amount",
        min: 0.0,
        max: 1.0,
        default: 0.0,
    }]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "horizontal",
            wgsl_source: include_str!("two_pass_example_h.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("h"),
        },
        PassDef {
            label: "combine",
            wgsl_source: include_str!("two_pass_example_combine.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("h")],
            output: PassOutput::Final,
        },
    ];

    fn from_values(values: &[f32]) -> Self {
        Self { amount: values[0], _padding: [0.0; 3] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<TwoPassParams>());
```

`PassOutput::Scratch("h")` and `PassInput::Scratch("h")` resolve to the same scratch
texture. The engine allocates it from a shared pool and returns it after the shader
completes. A `PassInput::Scratch(s)` in pass `i` must always correspond to a
`PassOutput::Scratch(s)` in some pass `j < i`.

### Const-fn validator

`ShaderRegistration::new::<T>()` calls `validate_pass_list(T::PASSES)` in a const
context, so a malformed `PASSES` is a build error rather than a runtime crash:

```
error[E0080]: evaluation of `ShaderRegistration::new::<TwoPassParams>` failed
  |
  = note: validate_pass_list: PassInput::Scratch references a name not written
          by any earlier pass
```

The three enforced rules:

1. `PassOutput::Final` must appear exactly once, on the last pass.
2. Every `PassInput::Scratch(s)` must reference a prior `PassOutput::Scratch(s)`.
3. No two passes may declare `PassOutput::Scratch` with the same name.

### Data-dependent loop bounds (`RADIUS_CAP`)

Separable filter passes (blur, smooth) compute their kernel radius from
`textureDimensions` at dispatch time. Always pair the computed radius with a compile-
time cap to give the GPU compiler an upper bound for register allocation and to prevent
pathologically large images from ballooning the kernel:

```wgsl
const SIGMA_FRACTION: f32 = 0.02;
const RADIUS_CAP:     i32 = 360;

fn main(...) {
    let dims   = textureDimensions(input_texture);
    let sigma  = SIGMA_FRACTION * f32(max(dims.x, dims.y));
    let radius = min(i32(ceil(3.0 * sigma)), RADIUS_CAP);
    // ... loop over [-radius, +radius]
}
```

The cap value should safely accommodate the largest expected image without capping the
intended blur. At `SIGMA_FRACTION = 0.02` on a 24 MP image (~6000 px), radius ≈ 360.
