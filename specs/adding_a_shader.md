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
All current shaders (brightness, saturation, contrast) follow it. New shaders
must also follow it unless the architecture is changed.

## Step-by-Step Checklist

### 1. Write the WGSL shader file

Create `bdip_core/src/gpu/<name>.wgsl`. Follow the existing pattern in
`brightness.wgsl` or `saturation.wgsl`:

- Declare bindings matching the bind group contract above.
- Define a params struct matching your Rust-side uniform (see step 2).
- Use `@workgroup_size(16, 16)` and an entry point named `main`.
- Operate in linear color space (input textures have already been
  ingested from sRGB).

For parameterless transforms (Grayscale, Invert), a dummy 16-byte uniform
must still be declared to satisfy the bind group layout (see step 2).

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

For parameterless transforms, use a dummy struct with no meaningful fields:

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GrayscaleParams {
    _unused: [f32; 4],
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
    Contrast,
    Example,       // <-- new
}

impl From<&Transformation> for TransformKind {
    fn from(t: &Transformation) -> Self {
        match t {
            Transformation::Brightness(_) => TransformKind::Brightness,
            Transformation::Saturation(_) => TransformKind::Saturation,
            Transformation::Contrast(_) => TransformKind::Contrast,
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

For parameterless transforms, the payload-less variant still needs an arm
that creates a zeroed dummy buffer:

```rust
Transformation::Grayscale => {
    let p = GrayscaleParams { _unused: [0.0; 4] };
    engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Apply Params Buffer"),
        contents: bytemuck::cast_slice(&[p]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}
```

### 6. Add the `Transformation` variant (if not already present)

In `bdip_core/src/transformation.rs`, add the new variant to the
`Transformation` enum and update the `Display` impl with its formatted
string. Parameterized variants should format as `"Name: {v:.2}"`;
parameterless variants as just `"Name"`.

### 7. Add CLI parsing (if applicable)

In the headless CLI (`bdip/src/main.rs`), add a match arm in the
`parse_transform` function to accept the new shader name and its
parameter:

```rust
"example" => {
    if parts.len() != 2 {
        return Err(anyhow::anyhow!(
            "Example requires a float value. E.g., example:0.5"
        ));
    }
    let val = parts[1].parse::<f32>()?;
    Ok(Transformation::Example(val))
}
```

For parameterless transforms, expect no colon-separated value:

```rust
"grayscale" => Ok(Transformation::Grayscale),
```

### 8. Add the variant to the UI pick list (`bdip/src/ui/sidebar.rs`)

Add the corresponding `TransformOption` variant to the `TRANSFORM_OPTIONS`
slice in `sidebar.rs`. The sidebar already handles the two control paths:

- **Parameterized transforms** (those matching the slider arm in
  `transform_view`): add the variant to the `pick_list` and the slider
  match arm in `sidebar.rs`. The slider range is `-1.0..=1.0` with step
  `0.01`. The `app.rs` `update()` handler converts `SliderReleased` into
  a `Transformation::Example(preview_value)` push to `HistoryManager`.
- **Parameterless transforms**: add the variant to the `pick_list` and the
  toggle match arm in `sidebar.rs`. The toggle displays an "Apply" label on
  the left and an `iced::widget::toggler` on the right. Its active state
  reflects whether the selected transform is the most recent entry in
  `HistoryManager`. The `ToggleParameterless` message in `app.rs` checks the
  current active state: if ON it calls `history.undo()`; if OFF it pushes the
  transform via `history.apply()`.

`TransformOption` is defined in `bdip/src/ui/message.rs`. Add the variant
there too, along with its `Display` and `from_transformation` arms.

**Slider/parameterless routing in `sidebar.rs`:**

```rust
const TRANSFORM_OPTIONS: &[TransformOption] = &[
    TransformOption::Brightness,
    TransformOption::Saturation,
    TransformOption::Contrast,
    TransformOption::Example,  // <-- new
];

fn transform_view(app: &BdipApp) -> Element<'_, Message> {
    let transform_control: Element<'_, Message> = match app.selected_transform {
        // parameterized → slider
        TransformOption::Brightness
        | TransformOption::Saturation
        | TransformOption::Contrast
        | TransformOption::Example => { /* slider widget */ }
        // parameterless → "Apply" label + toggler
        TransformOption::Grayscale | TransformOption::Invert => {
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
    // ...
}
```

**`update()` in `app.rs`:** the `SliderReleased` handler builds the
`Transformation` from `selected_transform` and `preview_value`. Add a
match arm there if the new variant is parameterized:

```rust
TransformOption::Example => Transformation::Example(self.preview_value),
```

For parameterless variants, the `ToggleParameterless` handler derives the
action from the current history state — no additional match arms are needed
there.

### 9. Write tests

At minimum, write unit tests in `pipeline.rs` for:

- **Identity case** — parameter value `0.0` (or equivalent no-op input)
  produces output that matches the input within f16 rounding tolerance (±64
  u16 units).
- **Extreme values** — verify clamping behavior at the parameter range
  boundaries (e.g., max positive pushes pixels in the expected direction,
  max negative collapses or flattens as expected).
- **Alpha preservation** — alpha channel is unchanged by the transform.
- **Chaining** — applying the new shader in combination with an existing
  shader (e.g., brightness then the new shader) produces numerically
  expected results.

Follow the `make_solid_image` + `roundtrip` helper pattern established in
the existing test suite. Each test must cover a single, isolated behavior
(see `AGENTS.md` Unit Testing Standards).

---

## Files Modified (Summary)

| File | Change |
|------|--------|
| `bdip_core/src/gpu/<name>.wgsl` | New file |
| `bdip_core/src/gpu/pipeline.rs` | Params struct, `TransformKind` variant, `compile()` arm, `apply()` arm, tests |
| `bdip_core/src/transformation.rs` | `Transformation` variant + `Display` arm (if new) |
| `bdip/src/main.rs` | `parse_transform` arm |
| `bdip/src/ui/message.rs` | `TransformOption` variant, `Display` arm, `from_transformation` arm |
| `bdip/src/ui/sidebar.rs` | `TRANSFORM_OPTIONS` slice, slider/button match arms |
| `bdip/src/ui/app.rs` | `SliderReleased` / `ApplyParameterless` match arms |

## Touch Points in `pipeline.rs`

Adding a shader currently requires modifying **4 locations** in
`pipeline.rs`:

1. Uniform params struct (top of file)
2. `TransformKind` enum + `From` impl
3. `PipelineCache::compile()` match arm
4. `Renderer::apply()` match arm

The future refactor in `specs/isolating_shaders_plan.md` aims to reduce
this to a single registration point per shader.
