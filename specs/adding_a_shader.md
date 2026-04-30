# Adding a New Shader

Adding a single-pass shader requires two new files and one line in `shaders/mod.rs`.
Multi-pass shaders follow the same pattern with additional `.wgsl` files; see
§ "Multi-pass shaders" below.

---

## Reference implementations

Use these existing shaders as templates:

| Pattern | Example | Key features |
|---------|---------|--------------|
| Single-pass slider | [`brightness`](../bdip_core/src/gpu/shaders/brightness/mod.rs) | One slider, minimal params struct |
| Single-pass toggle | [`grayscale`](../bdip_core/src/gpu/shaders/grayscale/mod.rs) | No user-facing params (`ParamKind::Toggle`) |
| Multi-pass | [`clarity`](../bdip_core/src/gpu/shaders/clarity/mod.rs) | 5 passes, scratch textures, `PassScale::Down` |
| Auxiliary texture (LUT) | [`color_lut`](../bdip_core/src/gpu/shaders/color_lut/mod.rs) | 3D texture, `AuxTextureDef` |

---

## Prerequisites

- The shader operates on linear-light `Rgba16Float` textures.
- The shader ID (a short ASCII string like `"hsl_hue"`) must be unique across all
  registered shaders.
- **Identity Default:** The default values for all parameters **must** result in an
  identity transformation (no change to the image). This ensures that adding a shader
  to the pipeline has no immediate visual effect until the user modifies the sliders.

---

## Single-pass shaders

### Step 1 — Create `bdip_core/src/gpu/shaders/<name>/mod.rs`

Define a params struct, implement `TransformShader`, and submit the registration.

**Parameterized shader (one or more sliders):**

```rust
use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExampleParams {
    pub value: f32,
    pub _padding: [f32; 3], // pad to 16 bytes for WebGPU uniform alignment
}

impl TransformShader for ExampleParams {
    const ID: &'static str = "example";
    const DISPLAY_NAME: &'static str = "Example";
    const DESCRIPTION: &'static str = "One-sentence description for tooltips and CLI help.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Amount",
        min: -1.0,
        max: 1.0,
        default: 0.0,                   // MUST be an identity value (no-op)
        description: "Per-slider description shown in parameter help.",
    }]);
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "example",
        wgsl_source: include_str!("example.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
    }];

    fn from_values(values: &[f32]) -> Self {
        Self { value: values[0], _padding: [0.0; 3] }
    }
}

inventory::submit!(crate::gpu::shaders::ShaderRegistration::new::<ExampleParams>());
```

**Parameterless shader (toggle):**

```rust
use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExampleParams {
    pub _unused: [f32; 4],
}

impl TransformShader for ExampleParams {
    const ID: &'static str = "example";
    const DISPLAY_NAME: &'static str = "Example";
    const DESCRIPTION: &'static str = "One-sentence description for tooltips and CLI help.";
    const PARAM: ParamKind = ParamKind::Toggle;
    const PASSES: &'static [PassDef] = &[PassDef {
        label: "example",
        wgsl_source: include_str!("example.wgsl"),
        inputs: &[PassInput::Source],
        output: PassOutput::Final,
        output_scale: PassScale::Full,
        aux_textures: &[],
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

    // Apply transformation (do NOT clamp — preserve >1.0 headroom for later shaders).
    // Note: When params.value is its default (0.0), this is an identity transformation.
    let out = vec4<f32>(pixel.rgb + params.value, pixel.a);
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
- **GPU roundtrip tests** covering: identity (verify that the registered `default`
  values result in a no-op), extreme parameter values, alpha preservation, and
  chaining with an existing shader.

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
use crate::gpu::shaders::{
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TwoPassParams {
    pub amount: f32,
    pub _padding: [f32; 3],
}

impl TransformShader for TwoPassParams {
    const ID: &'static str = "two_pass_example";
    const DISPLAY_NAME: &'static str = "Two-Pass Example";
    const DESCRIPTION: &'static str = "Demonstrates a two-pass blur-then-combine pipeline.";
    const PARAM: ParamKind = ParamKind::Sliders(&[SliderDef {
        name: "Amount",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Blend strength between original and processed image.",
    }]);
    const PASSES: &'static [PassDef] = &[
        PassDef {
            label: "horizontal",
            wgsl_source: include_str!("two_pass_example_h.wgsl"),
            inputs: &[PassInput::Source],
            output: PassOutput::Scratch("h"),
            output_scale: PassScale::Full,
            aux_textures: &[],
        },
        PassDef {
            label: "combine",
            wgsl_source: include_str!("two_pass_example_combine.wgsl"),
            inputs: &[PassInput::Source, PassInput::Scratch("h")],
            output: PassOutput::Final,
            output_scale: PassScale::Full,
            aux_textures: &[],
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

### Output scale

Use `PassScale::Down(n)` for intermediate passes that operate at reduced resolution
(e.g., blur kernels). The engine allocates a scratch texture at `(width/n, height/n)`.
The final pass must use `PassScale::Full`.

See [`clarity`](../bdip_core/src/gpu/shaders/clarity/mod.rs) for a 5-pass shader that
downsamples 4×, blurs, upsamples, then combines.

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

---

## Auxiliary textures (LUTs, noise maps, overlays)

Some shaders need external textures — 3D color LUTs, 2D noise maps, paper textures.
These are declared via `AuxTextureDef` in `PassDef::aux_textures`.

### Declaring an auxiliary texture

```rust
use crate::gpu::shaders::{
    AuxSamplerFilter, AuxTextureDef, AuxTextureDimension,
    ParamKind, PassDef, PassInput, PassOutput, PassScale, SliderDef, TransformShader,
};

const PASSES: &[PassDef] = &[PassDef {
    label: "apply_lut",
    wgsl_source: include_str!("apply_lut.wgsl"),
    inputs: &[PassInput::Source],
    output: PassOutput::Final,
    output_scale: PassScale::Full,
    aux_textures: &[AuxTextureDef {
        name: "identity_lut_64",        // must match a registered AuxAssetRegistration
        dimension: AuxTextureDimension::D3,
        filter: AuxSamplerFilter::Linear,
    }],
}];
```

The `name` must match an `AuxAssetRegistration` in `gpu/assets.rs`. The engine
uploads the texture once and caches it for subsequent passes.

### WGSL bind-group layout with aux textures

Auxiliary textures are bound in Group 2. Group 0 remains inputs + output; Group 1
remains the uniform buffer.

| Group | Binding | Resource |
|-------|---------|----------|
| 0 | 0 … N-1 | Input textures |
| 0 | N | Destination storage texture |
| 1 | 0 | Uniform buffer |
| 2 | 0, 2, 4, … | Aux textures (even bindings) |
| 2 | 1, 3, 5, … | Samplers for each aux texture (odd bindings) |

**Example for a 3D LUT:**

```wgsl
@group(2) @binding(0) var lut_texture: texture_3d<f32>;
@group(2) @binding(1) var lut_sampler: sampler;
```

See [`color_lut`](../bdip_core/src/gpu/shaders/color_lut/mod.rs) for a complete
implementation using a 64³ identity LUT.

---

## Performance testing

Multi-pass and computationally heavy shaders should have a performance test in
[`bdip_core/tests/performance.rs`](../bdip_core/tests/performance.rs). These tests
run via `cargo perf-test` (not the regular `cargo test` suite) and assert wall-clock
budgets at 24 MP.

### Adding a perf test

```rust
#[test]
fn perf_gpu_roundtrip_24mp_example() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(PERF_WIDTH, PERF_HEIGHT, 32767, 32767, 32767);
    let uploaded = upload_texture(&engine.device, &engine.queue, &img);

    let transform = Transform {
        shader_id: "example",
        values: vec![0.5],
    };
    let result = bench_shader_roundtrip(
        &engine,
        &mut renderer,
        &uploaded,
        img.width(),
        img.height(),
        &transform,
    );

    let (label, pass_count) = shader_display_info(transform.shader_id);
    print_perf_report(label, pass_count, &result, PERF_WARM_TARGET_MS);

    assert!(
        result.warm.critical_path_ms() < PERF_WARM_TARGET_MS,
        "{label} warm critical path exceeded {PERF_WARM_TARGET_MS:.0} ms target: {:.2} ms",
        result.warm.critical_path_ms()
    );
}
```

### Timing model

The benchmark harness splits wall-clock time into three buckets:

| Bucket | What it measures |
|--------|------------------|
| `execute_ms` | CPU time to encode + submit GPU commands |
| `gpu_wait_ms` | Wall time blocked in `device.poll(Wait)` — true GPU compute |
| `readback_ms` | Download path (copy + map + memcpy to CPU) |

The `critical_path_ms()` sum is the user-visible latency. The current targets
(`PERF_WARM_TARGET_MS = 30 ms`, `PERF_COLD_TARGET_MS = 80 ms`) are defined at the top
of `performance.rs`.

### When to add a perf test

- Multi-pass shaders (3+ passes)
- Shaders with auxiliary textures (exercises Group 2 bind group setup)
- Shaders with `PassScale::Down` (exercises scratch-texture allocation)
- Any shader expected to be near the performance budget

Simple single-pass shaders (brightness, contrast, grayscale) are covered by the
baseline `perf_gpu_roundtrip_24mp` test and don't need individual benchmarks.
