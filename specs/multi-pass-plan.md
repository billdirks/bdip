# Multi-Pass Shader Support — Implementation Plan

This plan executes **Option C** from `specs/multi-pass-research.md` (declarative multi-pass
in `ShaderRegistration`) and ships two new transforms on top of it: **Clarity** and
**Cartoon**. Read `specs/multi-pass-research.md` first — this plan assumes the option
tradeoffs, migration-to-E analysis, and bind-group discipline discussed there.

Also required reading before coding:

- `specs/execution_model.md` — Clean Slate Replay invariant; no intermediate CPU readback.
- `specs/adding_a_shader.md` — the current single-pass shader registration pattern, which
  stays the authoritative doc for single-pass shaders. This plan adds a sibling pattern.
- `specs/some_shaders.md` — Clarity formula (`C_hp = C_in - C_blurred`, combined with a
  midtone weight).
- `specs/film_grain_plan.md` — FilmGrainFBM discussion (see "Scope" below for why FBM is
  deliberately *not* part of this plan).

---

## Goals

1. **Enable multi-pass compute for a single user-facing `Transform`** without breaking the
   Clean Slate Replay model, the zero-touch shader registry, or any history/undo invariant.
2. **Ship Clarity** as the first multi-pass shader — separable Gaussian blur + combine.
3. **Ship Cartoon** as the second multi-pass shader — smoothing + edge detection + 3-input
   combine. Cartoon is the concrete 3-input-combine case that validates the
   position-indexed binding discipline in production code.
4. **Unify single-pass and multi-pass internally, and keep the single-pass `ShaderMeta`
   literal identical to today.** Every shader is represented as a `&'static [PassDef]`
   in the engine — there is no `Single` vs `MultiPass` branch in `Renderer::apply`.
   Translation from the contributor's `ShaderMeta` (single-pass, with `wgsl_source`) to
   the engine's internal pass-list form happens inside the registration macro, not at
   `apply` time. Contributors to single-pass shaders write the same `ShaderMeta` literal
   they write today; they do not see `PassDef`, `PassInput`, `PassOutput`, or any
   single-pass-vs-multi-pass vocabulary. Multi-pass contributors use a parallel
   `MultiPassShaderMeta` type and a separate registration macro — these exist only for
   shaders that actually need multi-pass. Migrating each existing shader from
   `inventory::submit!` boilerplate to a `register_single_pass_shader!` macro call is
   mechanical and must not change behavior or require per-shader tuning.
5. **Preserve the 24 MP warm-path performance budget** (~20 ms critical path). Each new
   compute pass costs ~0.3–0.5 ms; Clarity adds ~1 ms, Cartoon ~2 ms. Both fit inside the
   readback-dominated frame. Single-pass shaders must not regress measurably under the
   unified path (validated by PR 0's prototype).

## Non-goals

- **No render graph.** A general DAG (Option E) is out of scope. The position-indexed
  binding discipline this plan enforces is the one structural choice that keeps a future
  migration to E mechanical; nothing else in this plan anticipates E.
- **No auxiliary-texture bindings.** Blue-noise textures, LUTs, and other external
  resources are tracked separately in `film_grain_plan.md` (FilmGrainBlue) and are
  orthogonal to multi-pass.
- **No multi-pass FilmGrainFBM.** Procedural FBM via an in-shader octave loop is
  mathematically equivalent to multi-pass octave summing when both use procedural noise.
  Multi-pass FBM would only raise quality if combined with baked-noise-texture sampling
  (auxiliary-texture territory), which is out of scope. FilmGrainFBM ships separately as a
  single-pass shader per `film_grain_plan.md` option A.
- **No dynamic pass counts.** Every multi-pass shader declares a fixed, compile-time pass
  list. Variable-depth pyramids (Bloom, Laplacian filters) are the first workloads that
  would force Option E, and they are not on this plan's roadmap.
- **No scratch-pool LRU.** The pool drops all scratch textures on image-size change. A
  smarter eviction strategy is a follow-up only if profiling shows it matters.

---

## Architecture decisions

Carried from `specs/multi-pass-research.md` § "Option C" and § "The one gotcha":

1. **A `Transform` may resolve to 1..N compute passes.** One shader_id, one history entry,
   one user-visible transform. Passes are an internal implementation detail.
2. **Passes are declared, not orchestrated.** Each multi-pass shader supplies a static
   `PassDef` array listing its passes in order. The engine walks the array and dispatches;
   shaders do not call into `Renderer`.
3. **Passes name their inputs and output with typed identifiers.** `PassInput::Source`
   and `PassOutput::Final` are special tokens for the Transform's boundaries —
   `Source` is the input texture (output of the previous Transform in the stack);
   `Final` is the output texture handed to the next Transform. Everything else is a
   named scratch: `PassInput::Scratch("h")` and `PassOutput::Scratch("h")` refer to the
   *same* scratch texture owned by the pool — one variant is the write handle (an
   earlier pass declares `PassOutput::Scratch("h")`), the other is the read handle (a
   later pass declares `PassInput::Scratch("h")`). The split exists only because input
   slots and output slots have different types in the bind group; the `"h"` identifier
   resolves to one underlying texture.
4. **Bindings are position-indexed, derived from declared arity.** A pass declaring N
   inputs binds them to `@group(0) @binding(0)` through `@group(0) @binding(N-1)`. The
   output storage texture binds to `@group(0) @binding(N)`. The uniform buffer stays on
   `@group(1) @binding(0)`. This is the single most important architectural commitment in
   the plan — it is what makes a future Option E migration touch zero existing WGSL.
5. **Scratch textures are owned by `Renderer` and recycled.** Keyed by
   `(shader_id, scratch_name, width, height)`. Pool is dropped on image-resize. Mirrors
   existing patterns (`present_tile_buffer`, `staging_buffer`).
6. **Every shader resolves to a pass list internally — no `Single`/`MultiPass` branch
   in the engine.** The engine works on a `RuntimeShader { passes: &'static [PassDef],
   ... }` that it retrieves from the registry. Single-pass shaders' pass lists are
   length-1 with `inputs: &[PassInput::Source]` and `output: PassOutput::Final`.
   `Renderer::apply` has exactly one execution path. Translation from the contributor's
   `ShaderMeta` (single-pass, carrying `wgsl_source`) to `RuntimeShader` happens inside
   the `register_single_pass_shader!` macro at submission time; the contributor writes
   the same `ShaderMeta` fields they write today and never sees `PassDef`.
7. **One WGSL file per pass.** A multi-pass shader is a directory with one `mod.rs` and N
   `.wgsl` files. `include_str!` picks each up at compile time. Single-pass shaders keep
   their existing single WGSL file.
8. **Scratch textures are `Rgba16Float`** — same format as the main pipeline. No precision
   loss between passes; full headroom preserved.
9. **Position-indexed bindings apply uniformly to single-pass and multi-pass.** Today's
   single-pass WGSL already uses `@binding(0)` for source, `@binding(1)` for destination,
   and `@group(1) @binding(0)` for uniforms — the exact layout the position-indexed
   contract produces for a 1-input pass. No WGSL file changes for existing shaders.

---

## Core abstractions

### New types in `bdip_core/src/gpu/shaders/mod.rs`

```rust
/// Which resource a pass reads. `Source` / `Scratch("name")` are the read-side view
/// of the same name-space as `PassOutput`. A pass reading `Scratch("h")` reads the
/// texture written by an earlier pass that declared `PassOutput::Scratch("h")`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassInput {
    /// The Transform's input texture (output of the previous Transform).
    Source,
    /// A scratch texture written by an earlier pass in this same Transform.
    Scratch(&'static str),
}

/// Where a pass writes its output. `Scratch("name")` is the write-side view of the same
/// name-space as `PassInput`. The same `"name"` resolves to the same texture in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassOutput {
    /// Write to the named scratch texture. A later pass reads it via
    /// `PassInput::Scratch("name")`. The engine allocates and recycles the texture.
    Scratch(&'static str),
    /// Write to the Transform's final output texture (handed to the next Transform).
    Final,
}

/// Declarative description of one compute pass.
#[derive(Debug, Clone, Copy)]
pub struct PassDef {
    /// Debug label (e.g., "blur_h"). Used for GPU object labels and error messages.
    pub label: &'static str,
    /// WGSL source for this pass. Must declare bindings per the position-indexed
    /// contract (see `specs/multi-pass-plan.md` § "Bind-group contract").
    pub wgsl_source: &'static str,
    /// Inputs in declared order. Index `i` binds to `@group(0) @binding(i)` in WGSL.
    /// An empty slice is allowed for passes that synthesize their output from
    /// uniforms only (rare; included for completeness).
    pub inputs: &'static [PassInput],
    /// Where this pass writes.
    pub output: PassOutput,
}
```

### `ShaderMeta` stays as today; new `MultiPassShaderMeta` for multi-pass

`ShaderMeta` is unchanged from today's definition — this is the shape single-pass
contributors already write:

```rust
pub struct ShaderMeta {
    pub id: &'static str,
    pub display_name: &'static str,
    pub wgsl_source: &'static str,
    pub param: ParamKind,
}
```

A parallel type is added for multi-pass shaders:

```rust
pub struct MultiPassShaderMeta {
    pub id: &'static str,
    pub display_name: &'static str,
    pub passes: &'static [PassDef],
    pub param: ParamKind,
}
```

### Internal `RuntimeShader`

The registry stores and the engine walks this internal form. Contributors never write
or read it directly.

```rust
pub(crate) struct RuntimeShader {
    pub id: &'static str,
    pub display_name: &'static str,
    pub passes: &'static [PassDef],
    pub param: ParamKind,
}
```

`registry_by_id(id)` returns `&'static RuntimeShader`. Both `ShaderMeta` and
`MultiPassShaderMeta` are translated to `RuntimeShader` at registration-macro expansion
time, so the registry holds a homogeneous pool and `Renderer::apply` never branches on
shader kind.

### Registration macros

Two thin macros hide the `inventory::submit!` boilerplate and the single-pass-to-pass-list
translation.

**Single-pass (what all 11 existing shaders use):**

```rust
register_single_pass_shader! {
    meta: ShaderMeta {
        id: "brightness",
        display_name: "Brightness",
        wgsl_source: include_str!("brightness.wgsl"),
        param: ParamKind::Sliders(&[...]),
    },
    constructor: |values| Box::new(BrightnessParams::from_values(values)),
}
```

The macro expands to `inventory::submit!` of a `ShaderRegistration` whose `RuntimeShader`
has `passes = &[PassDef { label: "brightness", wgsl_source: <from meta>,
inputs: &[PassInput::Source], output: PassOutput::Final }]`. The `ShaderMeta` literal the
contributor writes has the exact four fields it had before — `id`, `display_name`,
`wgsl_source`, `param`.

**Multi-pass (new, used by Clarity and Cartoon):**

```rust
register_multi_pass_shader! {
    meta: MultiPassShaderMeta {
        id: "clarity",
        display_name: "Clarity",
        passes: &[
            PassDef { label: "blur_h",  wgsl_source: include_str!("blur_h.wgsl"),
                      inputs: &[PassInput::Source], output: PassOutput::Scratch("h") },
            PassDef { label: "blur_v",  wgsl_source: include_str!("blur_v.wgsl"),
                      inputs: &[PassInput::Scratch("h")], output: PassOutput::Scratch("v") },
            PassDef { label: "combine", wgsl_source: include_str!("combine.wgsl"),
                      inputs: &[PassInput::Source, PassInput::Scratch("v")],
                      output: PassOutput::Final },
        ],
        param: ParamKind::Sliders(&[...]),
    },
    constructor: |values| Box::new(ClarityParams::from_values(values)),
}
```

The macro forwards `passes` unchanged and fills in `RuntimeShader`. All multi-pass
vocabulary lives in this single type and this single macro — single-pass contributors
never import or reference them.

### Note on macro vs. `const fn`

Using a declarative macro for the translation avoids the const-fn limitations around
constructing `&'static [T; N]` literals inside a `const fn`. The macro expands at the
submit site to a literal array expression, which the compiler handles without
complaint. PR 0's evaluation criterion #1 verifies the macro expansion produces a
shape `inventory::submit!` accepts.

### Bind-group contract (multi-pass passes)

Each pass's bind group 0 is built from the declared `inputs` slice plus one output slot.
Group 1 continues to carry the uniform buffer.

| Group | Binding    | Resource                                                            |
|-------|------------|---------------------------------------------------------------------|
| 0     | 0..N-1     | Input textures in declared order (N = `inputs.len()`)               |
| 0     | N          | Destination storage texture (`rgba16float, write`)                  |
| 1     | 0          | Uniform buffer (same shader-wide params for all passes)             |

**All passes in one shader share the same uniform buffer.** Parameters are shader-level,
not pass-level. A Clarity pass and Cartoon pass each have one params struct; internal
passes read whichever fields they need. Keeping uniforms shader-scoped (not pass-scoped)
avoids a second uniform-buffer allocation per pass for data that never changes between
passes within a single Transform.

### Single-pass representation in the engine

A single-pass shader's `RuntimeShader.passes` is a slice of length 1 whose sole pass
has `inputs: &[PassInput::Source]` and `output: PassOutput::Final`. Single-pass WGSL
files stay exactly as they are today (source at `@binding(0)`, dest at `@binding(1)`,
uniform at `@group(1) @binding(0)`). The engine has one execution path — see the
"Renderer changes" section.

---

## Renderer changes

All changes are in `bdip_core/src/gpu/pipeline.rs`. Nothing else in `bdip_core` needs to
change.

### `PipelineCache`

Today: `HashMap<&'static str, CachedPipeline>`.
Change to: `HashMap<&'static str, Vec<CachedPipeline>>` (indexed by pass index; single-pass
shaders have `Vec` of length 1).

Compilation is still lazy per shader_id. When compiling a multi-pass shader, each
`PassDef` gets its own `ComputePipeline`, its own bind-group layout (derived from
`inputs.len()`), and its own shader module (from the pass's `wgsl_source`).

### Scratch pool

New field on `Renderer`:

```rust
// Keyed by (shader_id, scratch_name, width, height). All scratch textures are
// Rgba16Float — same format as the main pipeline, so no precision loss.
scratch_pool: HashMap<(&'static str, &'static str, u32, u32), wgpu::Texture>,
```

Scratch textures are allocated on first miss and retained for subsequent invocations at
the same `(shader_id, scratch_name, dims)` key. When `Renderer::apply` sees a new image
size for a key already in the pool at different dims, the old entry is dropped and
replaced. For V1, **any `apply` call at dims different from the majority of pool entries
triggers a full pool reset** (simpler than per-key tracking, and image resize is rare).

#### Design choice: per-shader pool vs. shared scratch textures

The pool is partitioned per `shader_id`. Clarity's `"h"` and `"v"` textures are distinct
entries from Cartoon's `"sh"` / `"smooth"` / `"quant"` / `"edges"`, even though nothing
would go wrong *physically* if both shaders shared one set of scratch textures at the same
dims. We chose per-shader partitioning for V1 over a shared pool for the following
reasons; a later optimization pass may revisit this.

**Why per-shader is the right V1 choice**

- **Simpler lifetime reasoning.** Each shader's scratches are sized and named
  independently. Within a `shader_id`, pass order implies the read/write windows
  automatically — no liveness analysis needed. A shared pool would have to track which
  scratch texture is currently "in use" during a single `apply` call to avoid aliasing a
  still-live scratch to a new pass's output.
- **Naturally correct across Transforms.** Multiple multi-pass Transforms in a Clean Slate
  Replay (e.g., Clarity followed by Cartoon) each look up their own textures by
  `shader_id` with no cross-shader coordination. A shared pool is also correct here —
  Clarity's scratches are fully consumed before Cartoon starts, so the same physical
  texture *could* be reused — but it requires the engine to understand that invariant
  rather than fall out of the key shape.
- **Better debugging.** `wgpu` texture labels like `"clarity::h"` and `"cartoon::smooth"`
  appear directly in RenderDoc / Xcode GPU captures. A shared pool would produce generic
  labels (`"scratch_0"`) and lose that affordance at the exact moment debugging needs it.
- **VRAM is fine in realistic stacks.** A 24 MP `Rgba16Float` scratch is ~185 MB. Clarity
  uses 2 scratches, Cartoon uses 4 — a stack of both simultaneously is ~1.1 GB. Comfortable
  on discrete GPUs and on the primary target hardware (Apple Silicon unified memory).
  Tight but survivable on low-end integrated GPUs.

**When a shared pool would be worth building**

The win from sharing is strictly VRAM reduction. A shared pool keyed by `(width, height)`
alone caps scratch footprint at `max_scratches_needed_by_any_one_shader × texture_size` —
roughly 4 × 185 MB = ~740 MB for a 24 MP workload on today's shaders, vs. ~1.1 GB with
per-shader partitioning. The ratio worsens as more multi-pass shaders get added:
partitioning scales linearly with `sum(scratches_per_shader)` while sharing scales with
`max(scratches_per_shader)`.

Concretely, consider building the shared variant if:

1. A user-reported OOM occurs on a realistic stack (e.g., ≥3 multi-pass shaders active at
   24 MP+ on integrated GPUs).
2. The multi-pass roadmap grows to the point where `sum(scratches)` crosses a GPU's safe
   working-set budget on the primary target hardware.
3. Telemetry (or perf profiling) shows scratch-pool memory dominates the app's total VRAM
   footprint and forces texture eviction elsewhere.

**Migration path**

The change is entirely internal to `Renderer` — `PassDef` and every existing WGSL file
stay untouched because the name `"h"` is still just an identifier, but now resolved
against a pool keyed by `(width, height)` with a per-pass liveness check. The work is:

1. Add liveness analysis to `apply_multi_pass`: before each pass, mark as free any scratch
   whose last reader is strictly below the current pass index.
2. Introduce a "scratch allocator" that, for each pass's output, returns either a freshly
   allocated texture or a previously freed one at matching dims.
3. Key the pool by `(width, height)` only.

Labels can preserve the `shader_id::scratch_name` naming by updating the label each time
a shared texture is reassigned — slightly ugly in RenderDoc but no worse than generic
names.

The shift from `(shader_id, scratch_name, w, h)` to `(w, h)` is the entire structural
change. No shader, no WGSL file, no test outside the pool itself needs to move.

### `Renderer::apply` dispatch

Today `Renderer::apply` is a single compute dispatch. Under the unified model it becomes
a single pass-list loop — no branch on shader kind:

```rust
pub fn apply(&mut self, engine: &GpuEngine, src_texture: &wgpu::Texture, transform: &Transform) -> wgpu::Texture {
    let reg = registry_by_id(transform.shader_id).expect(...);
    self.apply_passes(engine, src_texture, transform, reg, reg.meta.passes)
}
```

`apply_passes`:

1. Allocate the `Final` destination texture (same size as input) — this matches today's
   single-pass destination allocation exactly.
2. For every `PassOutput::Scratch(name)` referenced by the pass list, look up or lazily
   create the scratch texture in the pool. (Single-pass shaders never hit this step —
   their pass list contains no `Scratch` output.)
3. Build the uniform buffer once from `transform.values`.
4. For each pass in declaration order:
   - Resolve each `PassInput` to a concrete `wgpu::TextureView` (Source = input texture,
     Scratch = pool entry).
   - Resolve `PassOutput` to a concrete destination view (Scratch = pool entry, Final =
     output texture).
   - Build bind group 0 from `inputs.len()` input views plus the destination view.
   - Build bind group 1 from the shared uniform buffer.
   - Dispatch the pass's pipeline.
5. Submit one command encoder containing all passes (single submission per Transform —
   matches single-pass today).
6. Return the `Final` texture.

All scratch textures stay in the pool after the call returns; the next Clarity invocation
at the same dims reuses them.

**Single-pass fast path (speculative; add only if PR 0 measures a regression).**
The generic loop is correct for single-pass shaders: the pass list has one entry, the
only `PassOutput` is `Final`, and **no scratch texture is allocated** (the scratch pool
is only touched for `PassOutput::Scratch(...)`, which single-pass shaders never declare).
The destination `Final` texture is allocated per-call, exactly as today's single-pass
`apply` allocates its destination. So scratch-pool cost is a non-issue.

What *might* be measurable is per-call CPU overhead from the generic loop: iterating a
length-1 `passes` slice, resolving each `PassInput`/`PassOutput` via `match`, and the
extra `Vec` indirection in the pipeline cache (`HashMap<&str, Vec<CachedPipeline>>` vs
today's `HashMap<&str, CachedPipeline>`). On the 24 MP warm path, GPU dispatch dominates
`execute` — the loop overhead is likely low microseconds and below measurement noise.

If PR 0 measures a real regression (outside the +5% budget in PR 0's evaluation
criteria), the remediation is a short-circuit inside `apply_passes`:

```rust
fn apply_passes(&mut self, ..., passes: &[PassDef]) -> wgpu::Texture {
    if let [only] = passes {
        if only.inputs == &[PassInput::Source] && only.output == PassOutput::Final {
            return self.dispatch_pass_direct(..., only);  // straight-line
        }
    }
    // generic loop
}
```

`dispatch_pass_direct` is a single helper — essentially today's `apply` body — reachable
from the unified entry point. It is shape-preserving: no `Single`/`MultiPass` enum
resurfaces in the public API, and no shader author sees a difference.

**Add this short-circuit only if PR 0's measurement says to.** Speculatively adding it
reintroduces the second code path that unification was meant to eliminate, for a cost
that may be below measurement noise.

---

## Migrating existing single-pass shaders

Every existing shader swaps its `inventory::submit! { ShaderRegistration { ... } }`
boilerplate for a `register_single_pass_shader!` macro call. The `ShaderMeta` literal
inside is unchanged — same four fields, same `include_str!`, same `ParamKind`:

```rust
// Before
inventory::submit! {
    ShaderRegistration {
        meta: &ShaderMeta {
            id: "brightness",
            display_name: "Brightness",
            wgsl_source: include_str!("brightness.wgsl"),
            param: ParamKind::Sliders(&[...]),
        },
        constructor: |values| Box::new(BrightnessParams::from_values(values)),
    }
}

// After
register_single_pass_shader! {
    meta: ShaderMeta {
        id: "brightness",
        display_name: "Brightness",
        wgsl_source: include_str!("brightness.wgsl"),
        param: ParamKind::Sliders(&[...]),
    },
    constructor: |values| Box::new(BrightnessParams::from_values(values)),
}
```

The contributor writes no `PassDef`, imports no pass vocabulary, and does not learn that
an internal translation to a pass list is happening. This is mechanical across:

```
bdip_core/src/gpu/shaders/
├── brightness/mod.rs
├── contrast/mod.rs
├── exposure/mod.rs
├── grayscale/mod.rs
├── highlights/mod.rs
├── invert/mod.rs
├── saturation/mod.rs
├── shadows/mod.rs
├── temperature/mod.rs
├── tint/mod.rs
└── vignette/mod.rs
```

No WGSL file changes. No test changes. No behavioral change. Every existing test passes
unchanged.

---

## Motivating examples

### Clarity

Per `specs/some_shaders.md`:

```
C_hp = C_in - C_blurred
C_out = C_in + (C_hp * u_Clarity * W_mid)
```

where `W_mid` is a midtone luminance weight (peaks at mid-gray, falls off toward both
endpoints), and `C_blurred` is a Gaussian-smoothed copy of `C_in`.

**Pass list (3 passes):**

| Index | Label       | Inputs                            | Output           | Purpose                              |
|-------|-------------|-----------------------------------|------------------|--------------------------------------|
| 0     | `blur_h`    | `[Source]`                        | `Scratch("h")`   | Horizontal separable Gaussian blur   |
| 1     | `blur_v`    | `[Scratch("h")]`                  | `Scratch("v")`   | Vertical separable Gaussian blur     |
| 2     | `combine`   | `[Source, Scratch("v")]`          | `Final`          | High-pass extract + midtone-weighted |

**Params (single shared uniform across all 3 passes):**

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClarityParams {
    pub amount: f32,          // u_Clarity ∈ [-1.0, 1.0]
    pub blur_sigma: f32,      // fixed default e.g., 2.0% of image diagonal
    pub _padding: [f32; 2],
}
```

For V1, `blur_sigma` is not exposed — it is a fixed value set by `from_values(amount)`.
Exposing it as a second slider is a later, separate decision.

**Blur kernel size.** A 1D Gaussian with σ=2% of image width (e.g., ~100 px on a 24 MP
image) truncated at 3σ gives ~600 taps per invocation per pass. This is the professional
range. On 24 MP M4 Pro, ~0.4 ms per pass × 2 blur passes + ~0.05 ms combine = ~0.85 ms.
Verifiable against the warm perf test.

### Cartoon

Standard toon-filter pipeline: edge-preserving smoothing → quantization → edge detection →
combine.

**Pass list (5 passes):**

| Index | Label       | Inputs                                                   | Output             | Purpose                                  |
|-------|-------------|----------------------------------------------------------|--------------------|------------------------------------------|
| 0     | `smooth_h`  | `[Source]`                                               | `Scratch("sh")`    | Horizontal blur (larger σ than Clarity)  |
| 1     | `smooth_v`  | `[Scratch("sh")]`                                        | `Scratch("smooth")`| Vertical blur → smoothed image           |
| 2     | `quantize`  | `[Scratch("smooth")]`                                    | `Scratch("quant")` | Posterize to N levels per channel        |
| 3     | `edges`     | `[Source]`                                               | `Scratch("edges")` | Sobel magnitude → single-channel mask    |
| 4     | `combine`   | `[Source, Scratch("quant"), Scratch("edges")]`           | `Final`            | Mix `quant` with `Source` by `u_Strength`; overlay edges |

The `combine` pass is the 3-input-combine case. It binds three input textures to
`@binding(0)`, `@binding(1)`, `@binding(2)`, writes output to `@binding(3)`, reads
uniforms from `@group(1) @binding(0)`.

**Params:**

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CartoonParams {
    pub strength: f32,         // [0.0, 1.0] — 0 = original, 1 = full cartoon
    pub levels: f32,           // posterization levels per channel ∈ [2.0, 16.0]
    pub edge_threshold: f32,   // Sobel-magnitude cutoff ∈ [0.0, 1.0]
    pub edge_darkness: f32,    // how black the overlaid edges are ∈ [0.0, 1.0]
}
```

Four sliders: Strength, Levels, Edge Threshold, Edge Darkness. Matches the 4 sliders other
multi-param shaders (Shadows, Highlights) already expose.

### FilmGrainFBM — why it's not here

Procedural FBM = sum of N Perlin octaves with halving amplitude, doubling frequency.
Whether you sum them by looping in one shader invocation or by running N shader passes that
each add to an accumulator, the *result is identical* when the noise source is procedural
Perlin. Multi-pass gives zero quality improvement.

Multi-pass FBM raises quality only when combined with baked-noise-texture sampling (at
different mip levels), which requires auxiliary-texture binding — a separate architectural
axis owned by `film_grain_plan.md` (FilmGrainBlue). FilmGrainFBM continues to ship as a
single-pass shader via the in-shader octave loop called out as "Option A" in that spec.

---

## Test strategy

### Infrastructure tests (PR 1)

Tests live in `bdip_core/src/gpu/pipeline.rs` and `bdip_core/src/gpu/shaders/mod.rs`.

- `test_single_pass_macro_round_trips` — brightness registered via
  `register_single_pass_shader!` produces bit-identical output to its pre-migration
  baseline (captured as a golden byte array; compare to the existing roundtrip
  assertion).
- `test_single_pass_skips_scratch_pool` — after running a single-pass shader, the
  scratch pool has zero entries for that `shader_id`. Confirms single-pass shaders do
  not allocate scratch.
- `test_multi_pass_scratch_recycling` — a 2-pass test shader whose second pass copies
  its scratch input to `Final`. Run `apply` twice at the same dims; assert the scratch
  texture handle is reused (read back pool state for the assertion, or an observable
  side effect via labels).
- `test_multi_pass_image_resize_drops_pool` — `apply` once at 4×4, then once at 8×8;
  assert the pool has no lingering 4×4 entries (check pool size).
- `test_multi_pass_final_output_correctness` — a 2-pass identity shader (pass 0: Source
  → Scratch, pass 1: Scratch → Final, both plain copies) returns pixel-identical output
  to the input after `ingest`→`apply`→`present` roundtrip.
- `test_pipeline_cache_compiles_per_pass` — `get_or_create` on a multi-pass shader
  returns a `Vec<CachedPipeline>` whose length equals the pass count; on a single-pass
  shader it returns a length-1 vec. A second call returns the same vec entries by
  pointer equality.
- `test_position_indexed_bindings_three_inputs` — a test shader with a 3-input pass
  (`@binding(0)`, `@binding(1)`, `@binding(2)` for inputs; `@binding(3)` for output)
  correctly reads all three and writes the expected combination. This is the explicit
  regression guard against reverting to hardcoded binding slots.
- `test_single_pass_macro_synthesizes_one_pass_def` —
  `register_single_pass_shader!` produces a `RuntimeShader` whose `passes` slice has
  length 1 with `inputs == &[PassInput::Source]` and `output == PassOutput::Final`, and
  `wgsl_source` copied through from the `ShaderMeta` literal. Locks the single-pass
  translation so future edits to the macro cannot silently change the shape.

Each test follows `AGENTS.md` single-behavior rule.

### Shader-level tests (PRs 2 & 3)

**Clarity** (mirrors existing shader-test style from `vignette/mod.rs`):

| Test name                                      | Setup                                              | Assertion                                                        |
|------------------------------------------------|----------------------------------------------------|------------------------------------------------------------------|
| `test_clarity_registry_entry_exists`           | —                                                  | `registry_by_id("clarity").is_some()`                            |
| `test_clarity_registry_metadata`               | —                                                  | display_name, `Sliders`, `passes.len() == 3`                     |
| `test_clarity_make_uniform_known_value`        | `reg.make_uniform(&[0.5])`                         | bytes equal `bytemuck::bytes_of(&ClarityParams { amount: 0.5, .. })` |
| `test_clarity_zero_amount_is_identity`         | 16×16 solid mid-gray, `amount=0.0`                 | every pixel within ±64 of input                                  |
| `test_clarity_positive_amount_increases_contrast_on_edge` | synthetic step image (left half gray, right half white); `amount=0.5` | pixels just inside the edge on each side diverge further from the mean than with `amount=0.0` |
| `test_clarity_negative_amount_softens_edge`    | same step image; `amount=-0.5`                     | edge transition is softer than at `amount=0.0`                   |
| `test_clarity_alpha_preserved`                 | 4×4 solid mid-gray, `amount=0.5`                   | every output pixel's alpha == 65535                              |
| `test_clarity_deterministic`                   | 16×16 solid mid-gray; run twice at `amount=0.5`    | outputs pixel-identical                                          |
| `test_clarity_scratch_pool_reuses_across_runs` | run Clarity twice at same dims                     | pool has 2 entries ("h", "v") both times; same texture pointers  |

**Cartoon** (mirrors the same structure):

| Test name                                        | Setup                                             | Assertion                                                        |
|--------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------|
| `test_cartoon_registry_entry_exists`             | —                                                 | `registry_by_id("cartoon").is_some()`                            |
| `test_cartoon_registry_metadata`                 | —                                                 | name, 4 sliders, `passes.len() == 5`                             |
| `test_cartoon_make_uniform_known_value`          | `reg.make_uniform(&[0.5, 8.0, 0.2, 0.8])`         | bytes equal `bytemuck::bytes_of(&CartoonParams { .. })`          |
| `test_cartoon_zero_strength_is_identity`         | solid gradient image, `strength=0.0`              | output within ±128 of input per channel                          |
| `test_cartoon_full_strength_reduces_unique_colors` | smooth gradient, `strength=1.0`, `levels=4`     | unique pixel values in output < unique values in input (posterization works) |
| `test_cartoon_edges_darken_high_gradient_pixels` | image with sharp black/white edge, `edge_darkness=1.0` | pixels along the edge are darker in the output than in the input|
| `test_cartoon_no_edges_below_threshold`          | smooth gradient with no edges, `edge_threshold=0.1` | no pixel differs from the pure-posterized version by more than ±64 |
| `test_cartoon_alpha_preserved`                   | 4×4 solid mid-gray                                | every output pixel's alpha == 65535                              |
| `test_cartoon_deterministic`                     | same image, run twice with same params            | outputs pixel-identical                                          |
| `test_cartoon_three_input_combine_pass_binds_correctly` | synthetic inputs distinguishable per channel | combine output shows contributions from all 3 inputs (regression guard on binding positions) |

### Cross-shader integration tests (PR 4)

Add to `bdip_core/src/gpu/shaders/cross_shader_tests.rs`:

- `test_brightness_then_clarity` — Brightness 0.2 → Clarity 0.5 produces a deterministic
  output; mean pixel brightness > mean of Brightness-alone output (Clarity does not cancel
  Brightness).
- `test_clarity_then_vignette` — stacking Clarity with Vignette composes without crashing
  and preserves alpha.
- `test_cartoon_then_saturation` — Cartoon then Saturation composes; unique-color count
  after the stack is close to the Cartoon-alone count (Saturation does not restore colors
  that Cartoon quantized away).

### Performance budget test

Extend `test_perf_gpu_roundtrip_24mp` (or add `test_perf_gpu_roundtrip_24mp_clarity`) to
measure Clarity and Cartoon cold + warm critical paths on the 24 MP synthetic image, with
assertions:

- Clarity warm critical path: < 22 ms (20 ms baseline + ~1 ms Clarity overhead + ~1 ms
  slack).
- Cartoon warm critical path: < 24 ms (20 ms baseline + ~2 ms Cartoon overhead + ~2 ms
  slack).

These assertions are *soft* ceilings — if the measurement drifts, the assertion fires
before regressions reach production. The benchmark is `#[ignore]`-gated, same as today's
perf test.

---

## PR breakdown

Each PR is a discrete, reviewable unit. No PR leaves the codebase in a broken state.

### PR 0 — Prototype: unified single-pass path via `register_single_pass_shader!`

**Scope:** A throwaway spike on a feature branch (not merged to `main` as-is). Its purpose
is to validate the three unknowns that would force a plan redesign if they don't resolve
as expected. Outcome is a go/no-go signal for PR 1 plus — if green — a diff that can be
cleaned up and opened as PR 1.

**Rationale.** The unified model (no `Single`/`MultiPass` branch in the engine) is a
readability and migration-path win *only* if three concrete mechanisms hold up. PR 0
exists because those mechanisms are cheap to check in isolation and expensive to unwind
after PR 1 ships them to `main`.

**What the prototype builds**

1. Adds `PassInput`, `PassOutput`, `PassDef` to `bdip_core/src/gpu/shaders/mod.rs` exactly
   as defined in "Core abstractions" above. Keeps `ShaderMeta` unchanged.
2. Adds `MultiPassShaderMeta` and the internal `RuntimeShader` type.
3. Adds the `register_single_pass_shader!` and `register_multi_pass_shader!` macros.
4. Migrates **one** existing single-pass shader — `brightness` — to
   `register_single_pass_shader!`. The `ShaderMeta` literal inside is identical to what
   it contains today. Leaves the other 10 single-pass shaders on their pre-migration
   `inventory::submit!` form via a temporary compat shim, solely so PR 0 compiles and
   tests pass during the spike.
5. Reworks the registry so `registry_by_id` returns `&'static RuntimeShader`.
6. Reworks `Renderer::apply` into the unified `apply_passes` pass-list loop. All current
   single-pass shaders go through it.

**Evaluation criteria (all three must be green before opening PR 1)**

1. **`register_single_pass_shader!` macro expands and registers correctly.**
   - The macro accepts a `ShaderMeta { ... }` literal (unchanged from today's four-field
     shape) plus a constructor expression, expands to an `inventory::submit!` of a
     `ShaderRegistration` carrying a `RuntimeShader` with a synthesized 1-element
     `passes` slice, and compiles on stable Rust.
   - Brightness registers via the macro and is retrievable through
     `registry_by_id("brightness")`, returning a `RuntimeShader` whose `passes` has the
     expected shape (`inputs=[Source], output=Final`, `wgsl_source` copied through).
   - The single-pass contributor's `mod.rs` contains no `PassDef`, `PassInput`, or
     `PassOutput` references (verify by grep on the migrated `brightness/mod.rs`).
2. **Single-pass WGSL is a zero-diff migration.** Brightness's `brightness.wgsl` file is
   not modified at all. The existing `@binding(0)` (source), `@binding(1)` (dest),
   `@group(1) @binding(0)` (uniform) match the layout the position-indexed contract
   produces for `inputs=[Source], output=Final`. If any existing single-pass shader would
   need a WGSL binding edit to run under the unified path, PR 0 documents that and PR 1
   adjusts scope.
3. **Single-pass performance does not regress on the 24 MP warm path.**
   - Run `test_perf_gpu_roundtrip_24mp` before and after PR 0's changes (same commit
     parent, same hardware, same 20-iteration warm/cold sample policy).
   - Warm-path `execute` must stay within +5% of the pre-prototype baseline
     (today's ~0.35 ms → no worse than ~0.37 ms). Cold path is not a gate (pipeline
     compilation dominates and is not in the interactive critical path).
   - Also measure brightness specifically (the migrated shader) vs. one unmigrated
     shader (e.g., exposure) to rule out a per-shader anomaly.
   - If the generic loop is measurably slower than today's single-pass code, PR 0
     implements the "single-pass fast path" described in "Renderer changes" and
     re-measures. The fast path must be enough to restore parity; if not, escalate.

**Exit criteria**

- All three evaluation criteria green → clean up the prototype into PR 1 (migrate all 11
  single-pass shaders, remove the compat shim, finalize tests). PR 1's diff is
  essentially PR 0 plus the other 10 migrations plus the full infrastructure-test suite.
- Any criterion red → open an issue documenting the failure, update this plan, and
  reconsider. The most likely failure mode (criterion 1 with a const-fn limitation) is
  repaired by the macro fallback without changing the contributor surface or the plan
  structure; the other failure modes would warrant a plan revision.

**Files touched (spike only):**

- `bdip_core/src/gpu/shaders/mod.rs` (add types, modify `ShaderMeta`)
- `bdip_core/src/gpu/shaders/brightness/mod.rs` (switch to constructor)
- `bdip_core/src/gpu/pipeline.rs` (unified `apply_passes`)
- Temporary compat glue for the other 10 shaders (discarded before PR 1)

**Reporting.** The prototype's result is captured as a short note in the PR 1 description
(not a separate doc): "PR 0 prototype measured X ms warm-path execute for brightness under
the unified loop, Y ms baseline; const-fn constructor works / required macro fallback."

### PR 1 — Multi-pass infrastructure + existing-shader migration

**Scope:** All architectural changes needed to support multi-pass, with zero new shaders.
Every existing shader migrates to `register_single_pass_shader!` in the same PR so
`main` always has a consistent shape. Assumes PR 0's three evaluation criteria came back
green.

**Files to add:** none.

**Files to modify:**

- `bdip_core/src/gpu/shaders/mod.rs`:
  - Add `PassInput`, `PassOutput`, `PassDef`, `MultiPassShaderMeta`, and the internal
    `RuntimeShader` type. `ShaderMeta` stays exactly as today.
  - Add the `register_single_pass_shader!` and `register_multi_pass_shader!` macros.
  - Rework the registry so `registry_by_id` returns `&'static RuntimeShader`.
- `bdip_core/src/gpu/pipeline.rs`:
  - `PipelineCache` map value → `Vec<CachedPipeline>` (length 1 for single-pass).
  - `Renderer::scratch_pool` field.
  - `Renderer::apply` is a single unified dispatcher (`apply_passes`), with the
    single-pass short-circuit from "Renderer changes" only if PR 0 measured a
    regression.
  - New private `PassBindGroupLayout` helper that builds a layout from declared input
    arity.
- `bdip_core/src/gpu/shaders/{brightness,contrast,exposure,grayscale,highlights,invert,saturation,shadows,temperature,tint,vignette}/mod.rs`:
  swap `inventory::submit! { ShaderRegistration { ... } }` for
  `register_single_pass_shader! { meta: ShaderMeta { ... }, constructor: ... }` — 11
  files, mechanical. The `ShaderMeta` literal is unchanged; the only new thing is the
  macro name wrapping it. No WGSL file changes.
- `specs/adding_a_shader.md`: single-pass guidance is updated to show the new macro
  name (but the `ShaderMeta` fields shown inside are identical to today). Add a
  "Multi-pass shaders" section that introduces `MultiPassShaderMeta`,
  `register_multi_pass_shader!`, and the position-indexed binding contract — this
  section is where all pass-related vocabulary is introduced for contributors who need
  it.

**Tests shipped:** the Infrastructure tests listed above, plus a test-only 2-pass "copy
shader" fixture in `pipeline.rs` so the infrastructure has an integration surface without
needing a real new shader. All ~50+ existing shader tests continue to pass unchanged.

**Review focus:** the `PassDef` / `MultiPassShaderMeta` shape and the
`register_single_pass_shader!` / `register_multi_pass_shader!` macro expansions (these
are the public contract); confirm the single-pass `ShaderMeta` literal is byte-identical
to today's shape; position-indexed bind-group construction; scratch-pool lifecycle; and
the single-pass short-circuit (if present) — confirm it is a shape-preserving
optimization inside `apply_passes` and not a re-emergence of a `Single`/`MultiPass`
branch.

**Rollback characteristics:** if this PR is reverted, nothing on `main` ships multi-pass.
No user-visible change either way.

### PR 2 — Clarity shader

**Scope:** First real multi-pass shader on top of PR 1's infrastructure.

**Files to add:**

- `bdip_core/src/gpu/shaders/clarity/mod.rs`
- `bdip_core/src/gpu/shaders/clarity/blur_h.wgsl`
- `bdip_core/src/gpu/shaders/clarity/blur_v.wgsl`
- `bdip_core/src/gpu/shaders/clarity/combine.wgsl`

**Files to modify:**

- `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod clarity;`.
- `specs/some_shaders.md` — update the Clarity row to note it ships as multi-pass
  (separable Gaussian + combine) and reference this plan.

**Tests shipped:** the Clarity shader-level test matrix above.

**Review focus:** WGSL correctness of the Gaussian kernel (σ, kernel width, normalization),
midtone-weight formula, visual result on a real photo (include before/after screenshots
in the PR description).

### PR 3 — Cartoon shader

**Scope:** Second multi-pass shader; exercises the 3-input combine path.

**Files to add:**

- `bdip_core/src/gpu/shaders/cartoon/mod.rs`
- `bdip_core/src/gpu/shaders/cartoon/smooth_h.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/smooth_v.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/quantize.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/edges.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/combine.wgsl`

**Files to modify:**

- `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod cartoon;`.
- `specs/transformations_reference.md` — add a Cartoon section under a new "Stylization"
  heading referencing the XDoG paper and this plan.

**Tests shipped:** the Cartoon shader-level test matrix above, including the explicit
3-input binding regression guard.

**Review focus:** Sobel kernel, posterize math, combine formula, and **the WGSL binding
indices in `combine.wgsl`** — this is the first 3-input pass on `main` and is the
production validation of the position-indexed discipline.

### PR 4 — Cross-shader integration + performance guardrails

**Scope:** Stitch multi-pass into the broader test story. Small, safe.

**Files to modify:**

- `bdip_core/src/gpu/shaders/cross_shader_tests.rs` — add the three cross-shader chain
  tests listed above.
- `bdip_core/src/gpu/pipeline.rs` — extend `test_perf_gpu_roundtrip_24mp` (or add
  siblings) with Clarity + Cartoon perf assertions.

**Tests shipped:** cross-shader chain tests + perf guardrails.

**Rollback characteristics:** tests-only PR; reverting loses coverage but not behavior.

---

## Risks and open questions

- **Blur kernel portability across GPUs.** Large Gaussian kernels loop many texture reads;
  shader compilers may unroll differently. Validated on M4 Pro in PR 2's perf test; must
  also be spot-checked on a discrete NVIDIA GPU before V1 ship (not blocking for PR
  merges but must be on the V1 checklist).
- **Scratch pool growth if users stack multiple multi-pass shaders.** Each multi-pass
  shader contributes 1–4 scratch textures per image size. 24 MP `Rgba16Float` = ~185 MB
  per texture. A stack of 5 multi-pass shaders = ~1 GB scratch VRAM. Acceptable on
  typical discrete GPUs; tight on integrated GPUs. Mitigation (shared pool keyed by dims
  with per-pass liveness analysis) is described in "Renderer changes" §
  "Design choice: per-shader pool vs. shared scratch textures" — defer until profiling or
  a user OOM shows it matters.
- **Parameter coupling for Clarity.** V1 hardcodes blur sigma; a later PR may expose it as
  a slider. Decision is not blocking.
- **Cartoon parameter ranges.** The ranges above are informed estimates; will need tuning
  on real photos during PR 3 review. Not an architecture question.
- **`rustfmt` interaction with `register_single_pass_shader!` calls containing
  `include_str!(...)`.** Long `include_str!` calls sometimes cause awkward line breaks
  inside macro invocations. Non-blocking; can use `#[rustfmt::skip]` if needed on the
  affected lines.
- **Macro hygiene around the synthesized `PassDef` slice.** The macro emits a
  `&[PassDef { ... }]` literal bound to the caller's crate. PR 0 confirms that
  `inventory::submit!` accepts the expansion on stable Rust.
