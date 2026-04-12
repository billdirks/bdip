# User-Defined Shaders: Feasibility and Architectural Evaluation

## 1. Executive Summary

Supporting user-defined shaders (plugins) in `bdip` is highly achievable and aligns
well with the `wgpu` ecosystem. Because `wgpu` compiles WGSL at runtime (rather than
Ahead-Of-Time like SPIR-V in Vulkan), injecting a string loaded from a user's directory
directly into the engine is fundamentally supported by the underlying tech stack.

However, moving from the current "static" architecture to a "dynamic" one requires
significant structural changes, primarily around **Lazy Loading**, **Parameter
Definition**, **Dynamic UI Generation**, **Validation & Safety**, and **Shader
Authoring UX**.

**Difficulty:** Medium-High
**Performance Impact:** Zero (Shaders are compiled natively on the GPU, achieving
identical performance to built-in shaders).

---

## 2. Evaluation of Current Architecture

Currently, the GPU pipeline (`bdip_core/src/gpu/pipeline.rs`) is statically linked to
specific transformations:
1. **Monolithic Pipeline:** The `Renderer::new` method initializes pipeline layouts,
   bind groups, and the compute shader concurrently based on
   `include_str!("shader.wgsl")`.
2. **Fixed Uniforms:** The parameters passed to the GPU (like `ParamsUniform` containing
   `brightness_offset`) are defined as static Rust `structs` that implement
   `bytemuck::Pod`. WGPU requires these structs to follow strict 16-byte alignment
   rules.
3. **Static Enum Routing:** The `Transformation` enum dictates exactly what operations
   exist. The CLI `clap` parser and internal logic depend on these hardcoded variants.
4. **Three-Stage Pipeline:** The current architecture processes images through three
   stages: **ingest** (sRGB to linear conversion), **transform** (the actual image
   operation), and **present** (linear to sRGB conversion plus buffer packing). User
   shaders plug into the transform stage only; the ingest and present stages are
   engine-managed and must not be duplicated by plugin code.

To support "drop-in" custom shaders, the engine must safely transition from this
compiled structure to a data-driven model.

---

## 3. Implementation Outline (How to enable user shaders)

To allow a user to drop a shader into a folder and have it appear in the CLI/UI
seamlessly, the following implementation plan is required:

### Phase A: Architecture Refactoring (Addressing Tech Debt)
Before custom shaders can be loaded, the pipeline initialization must be refactored
to support lazy-loading (as identified in `specs/tech_debt.md`).
- **JIT Compilation Cache:** Create a system (e.g., a `PipelineCache` HashMap) that
  compiles a `wgpu::ComputePipeline` the *first time* an effect is requested, rather
  than at application boot. The cache grows unboundedly by design; GPU pipeline objects
  are small (kilobytes), and a desktop user will realistically load at most dozens of
  plugins in a session. LRU eviction is unnecessary for V1 but could be revisited if
  the plugin ecosystem grows large enough to warrant it.
- **Dynamic Transformation Variant:** Add a dynamic variant to the `Transformation`
  enum: `Transformation::Custom { id: String, params: HashMap<String, f32> }`.

**Tech Debt Interaction:** The lazy-loading refactor resolves the "Monolithic
Pipeline Initialization" item in `specs/tech_debt.md`. The remaining tech debt
items (parameter struct naming, shader file naming) should still be addressed
during Phase 5 as originally recommended — they apply to built-in transforms
independent of the plugin system. See Section 5 for the full interaction
summary.

### Phase B: The Plugin Format
Users need a way to define both the shader logic and the parameters the UI/CLI should
expose. This requires a standard plugin format, likely matching a directory structure
(`~/.config/bdip/plugins/`):
- `myshader.wgsl`: The WGSL logic (see Phase F for authoring approach).
- `myshader.toml`: Metadata required for parsing and UI rendering.

**Example `myshader.toml`:**
```toml
contract_version = 1
name = "My Custom CRT Filter"
description = "Adds scanlines and chromatic aberration."
id = "crt_filter"

[[parameters]]
name = "scanline_intensity"
type = "f32"
default = 0.5
min = 0.0
max = 1.0

[[parameters]]
name = "curvature"
type = "f32"
default = 0.1
min = 0.0
max = 0.5
```

The `contract_version` field declares which bind group layout and shader contract the
plugin targets. The engine rejects plugins whose `contract_version` does not match
a supported version, reporting a clear error message (e.g., "Plugin 'crt_filter'
targets contract v2, but this version of bdip supports v1"). This prevents silent
breakage when the engine's internal layout evolves.

**V1 Parameter Scope:** V1 plugins support `f32` parameters only. This keeps the
dynamic uniform buffer construction trivial (see Phase C). Support for additional
types (`i32`, `u32`, `vec2<f32>`, etc.) is a future consideration that requires a
full `std140` layout calculator.

### Phase C: Dynamic Uniform Buffer Construction
When `bdip_core` passes parameters to the GPU, it currently uses contiguous byte arrays
mapped to Rust structs (`bytemuck`). Since we won't have static structs for user
plugins, we must build the uniform buffer dynamically.
- The Engine must parse the `TOML` parameters, gather the floating-point values
  supplied by the user (or default values), and pack them into a contiguous `Vec<u8>`.
- **Alignment for `f32`-only parameters:** Each `f32` occupies 4 bytes with natural
  4-byte alignment, so N parameters pack contiguously into `N * 4` bytes. The total
  buffer size must be padded to a multiple of 16 bytes (the `std140` minimum struct
  alignment in WGSL). For example, 3 parameters = 12 bytes of data + 4 bytes padding
  = 16 bytes total. This padding is appended automatically by the engine.
- **Future complexity:** Supporting non-`f32` types (vectors, integers) would require
  a proper `std140` layout engine that handles per-field alignment rules (`vec2` to
  8-byte, `vec3`/`vec4` to 16-byte boundaries). This is well-understood but out of
  scope for V1.

### Phase D: Bind Group Layout Standardization
The engine needs a "contract" that all custom shaders adhere to in their WGSL:
- `@group(0) @binding(0)`: Source Texture (`texture_storage_2d<rgba16float, read>`)
- `@group(0) @binding(1)`: Destination Texture
  (`texture_storage_2d<rgba16float, write>`)
- `@group(1) @binding(0)`: Uniforms (Parameters)

**Color space contract:** Source textures are in **linear** `Rgba16Float` color space
(post-ingest). Destination textures must also be written in **linear** `Rgba16Float`
(pre-present). The engine handles all sRGB conversion automatically through its
ingest and present stages. User shaders must NOT apply their own gamma/sRGB
conversions, as doing so would result in double-conversion artifacts (washed out or
over-darkened output).

If a user shader fails to match this layout, `bdip_core` gracefully catches the
compilation error and reports it. See Phase E (Validation) for the full error
handling strategy.

### Phase E: Validation & Safety
User-supplied WGSL is untrusted input. The engine must validate and guard against
several failure modes:

1. **Syntax validation:** Before creating a `wgpu::ShaderModule`, run the WGSL source
   through `naga` (the shader compiler already used internally by `wgpu`) to parse and
   validate it. This catches syntax errors, type mismatches, and unsupported features
   with clear, line-numbered error messages rather than opaque GPU driver crashes.
2. **Entry point verification:** After parsing, inspect the `naga::Module` to confirm
   the shader declares a `@compute @workgroup_size(16, 16, 1)` entry point named
   `main`. Reject shaders missing this entry point or using non-standard workgroup
   sizes.
3. **Bind group layout matching:** Verify that the shader's declared bindings match the
   contract (Phase D). Mismatches should produce an error referencing the expected
   layout, not a generic pipeline creation failure.
4. **Uniform buffer size check:** Ensure the shader's declared uniform struct size
   matches the byte count derived from the TOML parameter list (plus padding). A
   mismatch indicates the shader and manifest are out of sync.
5. **GPU error recovery:** If a shader passes all static checks but still fails at
   runtime (e.g., due to a driver-level error), the engine should catch the
   `wgpu::Error` via the device error callback, remove the pipeline from the cache,
   and report the failure to the user without crashing the application.
6. **GPU hangs:** `wgpu` does not currently provide a per-dispatch timeout mechanism
   on all backends. A shader containing an infinite loop can hang the GPU. This is a
   known limitation of the WebGPU ecosystem. For V1, document this risk for shader
   authors. A future mitigation could involve running user shaders on a separate
   `wgpu::Device` so that a device-lost event from a hang does not affect the
   application's primary device.

### Phase F: Shader Authoring UX
The raw bind group contract (Phase D) requires users to write substantial WGSL
boilerplate: bind group declarations, workgroup dispatch layout, coordinate
calculations, and texture load/store calls. This is error-prone and creates a high
barrier to entry.

**Recommended approach — Wrapper injection:** The engine provides a **wrapper
template** that contains all boilerplate. The user writes only a single function:

```wgsl
fn transform(color: vec4<f32>, params: Params) -> vec4<f32> {
    // User logic here. 'color' is linear Rgba16Float.
    return vec4<f32>(color.rgb * params.scanline_intensity, color.a);
}
```

At load time, the engine:
1. Reads the user's `.wgsl` file containing only the `transform` function.
2. Generates the `Params` struct definition from the TOML parameter list.
3. Injects the user function into the wrapper template, which provides the
   `@compute` entry point, bind group declarations, coordinate math, texture
   load/store, and the call to `transform`.
4. Compiles the assembled WGSL string.

This approach eliminates most bind-group-mismatch errors, enforces the color space
contract automatically, and allows shader authors to focus purely on the color math.
Advanced users who need direct texture access (e.g., for spatial filters that sample
neighboring pixels) can opt out of the wrapper by setting `raw = true` in their
TOML manifest, in which case they provide a complete WGSL file matching the Phase D
contract directly.

### Phase G: UI and CLI Discovery
- **CLI:** A discovery phase scans the plugin directory on boot. The CLI can expose
  `--apply custom:crt_filter:scanline_intensity=0.8`.
- **UI:** The `iced` application loads the `TOML` definitions dynamically. Because
  `iced` layouts are built per-frame, it simply reads the plugin manifests and
  automatically generates sliders for every parameter defined in the array.

---

## 4. Deployment Timing Strategy

**Should this be done before or after Phase 4 (Full UI Integration)?**
**After Phase 4 is acceptable**, provided the UI is built data-driven from the
start. The risk the original analysis identified — hardcoding one slider widget
per transform — is real, but it is a UI design mistake to avoid regardless of
whether a plugin system exists. The Phase 4 UI should iterate over the
`Transformation` enum's metadata (parameter names, ranges, defaults) to generate
slider controls dynamically. This can be achieved with a simple trait or match
on the enum; it does not require TOML parsing, disk scanning, or any plugin
infrastructure. When the plugin system is added later, the UI gains an
additional source of parameter metadata (plugin manifests) without structural
changes.

**Should this be done before or after Phase 5 (Remaining V1 Shaders)?**
**After Phase 5 is acceptable.** The original analysis argued that implementing
Contrast, Saturation, Grayscale, and Invert as hardcoded pipelines would create
throwaway work. In practice, the rework cost is near zero:

- The WGSL shader math (the actual filter logic) is reusable regardless of how
  it is loaded. Whether `contrast.wgsl` is compiled via `include_str!()` or
  read from disk at runtime, the code is identical.
- Built-in transforms do not need to use the plugin system. They can remain as
  compiled-in code with typed Rust structs permanently, benefiting from
  compile-time type safety. The plugin system is for *user-defined* shaders.
- Each Phase 5 shader is a small unit of work: one `.wgsl` file, one uniform
  struct, one `apply_*` method, one match arm. If unification under the plugin
  system is later desired for consistency, migrating each shader is trivial.

**Prerequisite that should not be deferred:** The **lazy-loading refactor**
(Phase A of this document, tracked as "Monolithic Pipeline Initialization" in
`specs/tech_debt.md`) should be completed before Phase 5. Eagerly compiling
every pipeline at startup scales poorly as shader count grows. This refactor
is a small, self-contained task that does not require the full plugin system.
**Status (2026-04-12):** This prerequisite is being addressed in Phase 4
alongside the Saturation shader. See `specs/2shader_plan.md`.

**Recommended ordering:**
1. Lazy-loading refactor (Phase A) + Saturation shader — completed as Phase 4
   (see `specs/2shader_plan.md`). This resolves the monolithic pipeline
   initialization debt and validates multi-shader dispatch before UI work begins.
2. Phase 4 (UI) — build data-driven from the `Transformation` enum.
3. Phase 5 (remaining shaders) — implement as normal built-in transforms.
4. Plugin system (Phases B–G) — layer on top of the working pipeline. At this
   point, 5 working built-in shaders serve as reference implementations, and
   the data-driven UI is ready to accept plugin manifests as an additional
   parameter source.

---

## 5. Tech Debt Interaction Summary

The plugin architecture intersects with the following items in `specs/tech_debt.md`:

- **Generic Parameter Structs (`ParamsUniform`)** — *Still relevant.* Built-in
  transforms should use dedicated, typed structs (e.g., `BrightnessUniform`,
  `ContrastUniform`) as recommended by the tech debt entry. Only user-defined
  plugins use dynamic uniform buffers. The two approaches coexist.
- **Generic Shader Naming (`shader.wgsl`)** — *Still relevant.* Should be
  resolved during Phase 5 when per-transform `.wgsl` files are created.
- **Monolithic Pipeline Initialization** — *Resolved by Phase A.* Lazy-loading
  / `PipelineCache` is a prerequisite for both Phase 5 and the plugin system.
- **Missing High-Level `apply()` API** — *Must be extended.* The `apply()`
  method must route `Transformation::Custom` variants through the plugin
  system once it exists.
- **Synchronous Disk I/O Blocks UI** — *Unrelated* but relevant timing:
  plugin directory scanning should also be async.

---

## 6. Future Considerations

These items are explicitly out of scope for V1 of the plugin system but should be
planned for:

- **Hot-reload:** Watching the plugin directory for filesystem changes (`notify`
  crate) and recompiling modified shaders without restarting the application. This
  is a significant UX improvement for shader authors during development.
- **Non-`f32` parameter types:** Supporting `i32`, `u32`, `bool`, `vec2<f32>`,
  `vec3<f32>`, `vec4<f32>` parameters. Requires a full `std140` layout calculator
  for dynamic uniform buffer packing.
- **Pipeline cache eviction:** If the plugin ecosystem grows to hundreds of
  shaders, an LRU eviction strategy for compiled pipelines may become necessary.
- **Separate GPU device for untrusted shaders:** Running user shaders on an
  isolated `wgpu::Device` so that GPU hangs or device-lost events do not affect
  the primary rendering device.
- **Spatial filter support in wrapper mode:** The Phase F wrapper assumes
  per-pixel transforms. Spatial filters (blur, edge detection) need neighbor
  sampling, which requires either a different wrapper signature or opting into
  raw mode. A future "spatial wrapper" could provide a `fn transform_spatial(coord,
  params, sampler)` signature.
- **Plugin distribution:** A mechanism for sharing plugins (registry, Git
  repositories, zip bundles) is out of scope but worth considering if the user
  community grows.

---

## 7. Conclusion

Adding user-defined shaders natively into the `bdip` architecture is highly feasible
and would increase the tool's extensibility. The core difficulty lies on the Rust
host-side: **safely validating untrusted WGSL, converting dynamic parameter maps
into aligned byte buffers for the GPU, and providing an authoring experience that
abstracts away boilerplate.**

The wrapper-injection approach (Phase F) substantially lowers the barrier for shader
authors while the raw-mode escape hatch preserves full flexibility. Combined with
`naga`-based validation (Phase E) and an explicit color space contract (Phase D),
the system can accept user shaders safely without sacrificing the correctness of
the ingest/present pipeline.

As long as the pipeline initialization is decoupled from startup (resolving the
monolithic tech debt problem), user-created shaders will perform identically to
native, hardcoded transformations.
