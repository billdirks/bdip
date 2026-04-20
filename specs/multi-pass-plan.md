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
5. **Scratch textures are owned by `Renderer` and recycled via a shared free list.**
   Pool is keyed by `(width, height)` only. `apply_passes` borrows one texture per
   distinct scratch name on entry and returns them on exit, so the same physical
   textures are reused across passes-within-a-shader *and* shaders-within-a-stack. Pool
   is dropped on image-resize. Mirrors existing patterns (`present_tile_buffer`,
   `staging_buffer`).
10. **All passes of a multi-pass shader run at the Transform's input resolution.** Every
    scratch texture is allocated at `(input_width, input_height)` and every pass
    dispatches `ceil(input_width / 16)` × `ceil(input_height / 16)` × 1 workgroups with
    `@workgroup_size(16, 16)` — the same sizing today's single-pass `Renderer::apply`
    uses for its lone dispatch. No pass declares or receives a resolution different
    from the Transform's input. Downsampled/upsampled intermediates (pyramids, variable
    mip chains) are explicitly out of scope and are the first thing that would force an
    Option E render graph (see "Non-goals").
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

### Registration-time validation of multi-pass pass lists

Malformed pass lists (typos in scratch names, `Final` in the wrong position, duplicate
scratch outputs) are caught as early as possible. The plan uses a two-tier strategy:
**try const-fn first** so the error fires at `cargo build`; **always ship a registry-walk
test** as the guaranteed safety net so shaders without dispatch tests still get
validated in CI.

**Tier 1 — `const fn` validator invoked by `register_multi_pass_shader!`.**

The macro emits, next to the `inventory::submit!` call, a `const _: () = validate_pass_list(PASSES);`
block where `validate_pass_list` is a `const fn` in `bdip_core/src/gpu/shaders/mod.rs`.
The validator enforces:

1. **Exactly one `PassOutput::Final`, at the last pass.** No earlier pass may output
   `Final`.
2. **Every `PassInput::Scratch(s)` resolves to a prior write.** Walk the pass list in
   order; for each input `Scratch(s)` at index `i`, assert some pass at index `j < i`
   declared `PassOutput::Scratch(s)`.
3. **No duplicate `PassOutput::Scratch(name)` across the pass list.** Reusing a name as
   an output in two passes is forbidden — the pool borrow is per-name, and a second
   write would silently overwrite the first without the engine knowing to allocate a
   second texture.

On violation, the validator invokes `panic!` in const context (stabilized Rust 1.79+),
which the compiler reports at the build site referencing the offending shader's
`mod.rs`. Error messages name the shader id, the pass index, and the offending scratch
name.

Const-fn string equality uses stable byte-by-byte comparison on
`&'static str::as_bytes()`. The validator is ~30–50 lines of const code; complexity
stays localized to `shaders/mod.rs`.

**Single-pass is trivially valid by construction.** `register_single_pass_shader!`
synthesizes a 1-element pass list with `inputs: &[PassInput::Source]` and
`output: PassOutput::Final`, which satisfies all three rules without the contributor
seeing the validator.

**Tier 2 — `test_all_registered_pass_lists_validate` (in `shaders/mod.rs`).**

Regardless of whether tier 1 catches a given class of error, a dedicated test walks
`inventory::iter::<ShaderRegistration>()` and runs the same validation logic on every
registered `RuntimeShader.passes`. This:

- Catches any validation failure even if a shader ships without dispatch tests.
- Exercises the validator on every contributor's shader automatically — no per-shader
  opt-in.
- Provides the safety net if the const-fn validator has to be disabled (e.g., a future
  Rust regression around `panic!` in const, or complexity in the validator grows beyond
  the ~50-line budget).

**Fallback plan.** If const-fn string comparison turns out awkward during PR 1
implementation, the macro-level check degrades to a purely structural subset ("last
pass is `Final`, only one `Final`") that does not touch strings, and tier 2 becomes the
sole check for scratch-name resolution. Strictly worse than dual-tier but still catches
the error before any dispatch runs.

**What is not validated here.** WGSL binding *types* and binding counts matching the
declared arity are validated at pipeline-creation time (first `apply`, caught in
`test_pipeline_cache_compiles_per_pass` and the per-shader dispatch tests). Adding a
build-time naga compile step is out of scope for this plan.

### Bind-group contract (multi-pass passes)

Each pass's bind group 0 is built from the declared `inputs` slice plus one output slot.
Group 1 continues to carry the uniform buffer.

| Group | Binding    | Resource                                                            |
|-------|------------|---------------------------------------------------------------------|
| 0     | 0..N-1     | Input textures in declared order (N = `inputs.len()`)               |
| 0     | N          | Destination storage texture (`rgba16float, write`)                  |
| 1     | 0          | Uniform buffer (same shader-wide params for all passes)             |

**All passes in one shader share the same uniform buffer.** Parameters are shader-level,
not pass-level — this is the intended design, not a V1 deferral. A Clarity pass and a
Cartoon pass each have one params struct; internal passes read whichever fields they
need and ignore the rest. WGSL compilers dead-code-eliminate unreferenced struct
members at compile time, so an unused field costs neither CPU nor GPU cycles. The
"waste" is at worst ~20 bytes of unread uniform memory per Transform — strictly
cheaper than the alternative (N separate uniform structs, N buffer allocations, N
bind-group layouts, N `make_uniform` functions, new per-pass uniform field on
`PassDef`).

**Alignment rule: every `.wgsl` file in the shader declares the full, identical params
struct.** Not just the fields its pass reads. WebGPU validates the uniform binding's
size against the pipeline layout at creation time; a pass whose WGSL declares a
truncated struct fails pipeline creation with a cryptic byte-mismatch error rather
than silently reading the wrong offsets.

Concretely for Cartoon: all five WGSL files (`smooth_h.wgsl`, `smooth_v.wgsl`,
`quantize.wgsl`, `edges.wgsl`, `combine.wgsl`) declare:

```wgsl
struct CartoonParams {
    strength:       f32,
    levels:         f32,
    edge_threshold: f32,
    edge_softness:  f32,
    edge_darkness:  f32,
    _padding0:      f32,
    _padding1:      f32,
    _padding2:      f32,   // pad to 32 bytes, matching Rust-side #[repr(C)]
}

@group(1) @binding(0) var<uniform> params: CartoonParams;
```

— even though `smooth_h` and `smooth_v` reference none of these fields (they derive
sigma from `textureDimensions`, not uniforms), and `quantize` only reads `levels`. The
Rust-side `CartoonParams` with its `_padding: [f32; 3]` is the source of truth; every
WGSL declaration must match its byte layout exactly. A single-pass shader already
follows this rule trivially (one WGSL file, one declaration).

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
// Pool of reusable Rgba16Float scratch textures keyed only by (width, height).
// Textures are borrowed by an `apply_passes` invocation on entry and returned on exit,
// so the peak footprint equals `max(scratches_needed_by_any_one_shader) × texture_size`
// regardless of how many multi-pass Transforms stack in one Clean Slate Replay.
scratch_pool: HashMap<(u32, u32), Vec<wgpu::Texture>>,
```

Semantics:

- The pool is a free list per `(width, height)`. Textures are never freed mid-session;
  they are re-used across both passes-within-a-shader and shaders-within-a-stack.
- `apply_passes` **borrows** one texture per distinct `PassOutput::Scratch(name)`
  declared in its pass list, pulling from the free list or allocating on miss. Borrows
  stay checked out for the duration of that `apply_passes` call.
- When `apply_passes` returns, every borrowed texture is **returned** to the free list.
  The next Transform's `apply_passes` (Clarity → Cartoon, for example) finds those same
  textures waiting and reuses them.
- On image-size change, the whole pool is dropped and rebuilt. For V1, **any `apply`
  call at dims different from the majority of pool entries triggers a full pool reset**
  — simpler than per-key tracking, and image resize is rare.

This design relies on two invariants already present in the engine:

1. **Transforms execute strictly sequentially.** `Renderer::apply` is `&mut self`; there
   is no concurrent `apply` on one `Renderer`. A Transform's `apply_passes` submits its
   command encoder before returning, so the next Transform's passes are queued after
   everything the previous Transform wrote.
2. **Within one `apply_passes`, each scratch name is bound to its own borrowed texture
   for the whole call.** No intra-shader aliasing — a pass that declares
   `PassInput::Scratch("h")` always sees the texture the earlier pass wrote to `"h"`.
   Liveness analysis inside a single shader is therefore unnecessary; the cost is one
   borrowed texture per distinct scratch name, which caps at the per-shader maximum.

#### Why this beats per-shader partitioning

- **Bounded peak VRAM.** 24 MP `Rgba16Float` scratch ≈ 185 MB. Cartoon (4 scratches)
  caps the pool at ~740 MB regardless of how many multi-pass shaders stack above it.
  Per-shader partitioning would scale with `sum(scratches_per_shader)` and reach
  ~1.1 GB for Clarity + Cartoon today, worse as the multi-pass roadmap grows.
- **No reliance on Clean Slate Replay as a load-bearing correctness invariant.** The
  borrow/return discipline makes the pool correct under any call pattern that respects
  `&mut self` — we are not encoding "Transforms never overlap" into the pool's key shape.
- **Simpler code.** One free list, no `shader_id` threading into the key, no per-shader
  bookkeeping.

#### Mitigating the debugging-label cost

A naive shared pool loses `"clarity::h"` / `"cartoon::smooth"` labels in RenderDoc /
Xcode GPU captures, since one physical texture gets handed to multiple shaders over the
course of a session. The mitigations, in order of effort:

1. **Relabel on borrow (V1 default).** When `apply_passes` borrows a texture, it calls
   `texture.set_label(Some(&format!("{}::{}", shader_id, scratch_name)))` (or whatever
   the wgpu-equivalent re-labeling mechanism is on the current wgpu version — fall back
   to labeling only at allocation if the runtime does not support re-labeling). The
   label in a GPU capture reflects **the most recent borrower**, which matches what a
   developer is actually debugging at capture time.
2. **Debug-build label suffixes.** Under `#[cfg(debug_assertions)]`, append a monotonic
   borrow counter (`"clarity::h#17"`) so a capture spanning multiple `apply_passes`
   calls disambiguates which Transform wrote which frame. Release builds keep the clean
   name.
3. **Opt-in "no-reuse" mode for GPU captures.** Behind an env var (e.g.
   `BDIP_SCRATCH_NO_REUSE=1`), the pool skips the free list and allocates fresh per
   borrow. Captures taken with this flag set produce the per-shader label hierarchy that
   per-shader partitioning would have given for free, at the VRAM cost per-shader would
   have imposed anyway. Intended only for the narrow case where someone is hunting a
   cross-shader scratch-aliasing bug.

V1 ships (1) and (2). (3) is a follow-up if we actually encounter a debugging scenario
that (1) + (2) cannot resolve.

#### Test-only accessor for pool introspection

The pool is private to `Renderer`, but the infrastructure and shader tests need a way
to observe recycling without relying on GPU-capture labels or pointer-bit reinterpret.
`Renderer` exposes:

```rust
#[cfg(test)]
impl Renderer {
    /// Number of textures currently in the free list for the given dims.
    /// Does NOT count textures currently checked out by an in-flight `apply_passes`
    /// — since `Renderer::apply` is `&mut self`, tests only observe the pool at
    /// rest (between calls).
    pub(crate) fn scratch_pool_len(&self, dims: (u32, u32)) -> usize { ... }

    /// Pointer-equality handle on a specific pool slot for same-texture assertions
    /// across runs. Returns `None` if `index >= scratch_pool_len(dims)`.
    pub(crate) fn scratch_pool_handle(&self, dims: (u32, u32), index: usize) -> Option<*const wgpu::Texture> { ... }
}
```

`scratch_pool_len` is the assertion surface for "how many scratch textures exist for
these dims"; `scratch_pool_handle` lets `test_clarity_scratch_pool_reuses_across_runs`
prove the *same physical texture* came back on the second run, not merely "a texture
of the same shape."

Both accessors are `#[cfg(test)]`-gated so they do not appear in release builds and
cannot be called from outside the crate. This replaces the earlier hand-wavy "read
back pool state or observe via labels" language in the test plan with a concrete,
testable API.

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

1. Read `(width, height)` from the input texture. These are the **Transform dims**;
   every scratch allocation and every pass dispatch in this call uses them.
2. Allocate the `Final` destination texture at Transform dims — this matches today's
   single-pass destination allocation exactly.
3. For every **distinct** `PassOutput::Scratch(name)` referenced by the pass list,
   **borrow** one texture at Transform dims from the pool's `(width, height)` free
   list (allocating on miss). Each borrow is associated with its `name` for the
   duration of this `apply_passes` call, and — under `#[cfg(debug_assertions)]` plus on
   allocation — its wgpu label is set to `"{shader_id}::{scratch_name}"` (see
   "Mitigating the debugging-label cost"). Single-pass shaders never hit this step —
   their pass list contains no `Scratch` output.
4. Build the uniform buffer once from `transform.values`.
5. For each pass in declaration order:
   - Resolve each `PassInput` to a concrete `wgpu::TextureView` (Source = input texture,
     Scratch = the borrow mapped to that name).
   - Resolve `PassOutput` to a concrete destination view (Scratch = the borrow mapped to
     that name, Final = output texture).
   - Build bind group 0 from `inputs.len()` input views plus the destination view.
   - Build bind group 1 from the shared uniform buffer.
   - Dispatch the pass's pipeline at **Transform dims**:
     `dispatch_workgroups(ceil(width / 16), ceil(height / 16), 1)` with the shader's
     `@workgroup_size(16, 16)`. Every pass uses the same dispatch — no per-pass
     resolution override exists.
6. Submit one command encoder containing all passes (single submission per Transform —
   matches single-pass today).
7. **Return every borrowed texture to the pool's free list.** The next Transform's
   `apply_passes` (same dims) reuses them.
8. Return the `Final` texture.

**Invariant: no variable-resolution scratch.** Every texture allocated or dispatched
against during `apply_passes` is at Transform dims. A future shader that needs a
downsampled intermediate (Bloom, Laplacian pyramid) cannot be expressed under this
model — that is the feature that forces Option E and is explicitly out of scope per
"Non-goals".

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
    pub _padding: [f32; 3],
}
```

**Blur sigma is derived in the shader, not the uniform.** `ClarityParams` carries only
`amount`. The "2% of image diagonal" sigma rule lives inside each blur-pass WGSL file,
where `textureDimensions(input_texture)` is already available:

```wgsl
const SIGMA_FRACTION: f32 = 0.02;
const RADIUS_CAP: i32 = 256;

@compute @workgroup_size(16, 16)
fn main(...) {
    let dims = textureDimensions(input_texture);
    let sigma = SIGMA_FRACTION * f32(max(dims.x, dims.y));
    let radius = min(i32(ceil(3.0 * sigma)), RADIUS_CAP);
    // ... separable Gaussian loop over [-radius, +radius]
}
```

Rationale: the registry-wide `from_values(&[f32]) -> Self` signature stays unchanged
(no shader needs image dims on the CPU side), and sigma's unit ("texels") is honest
where it is consumed. Cartoon's smooth passes follow the same pattern with a larger
`SIGMA_FRACTION`.

**Data-dependent loop bound.** `radius` is computed from `dims` at dispatch time, so
the kernel loop is not statically bounded. WGSL permits this, but the `RADIUS_CAP`
min gives the compiler a compile-time upper bound for unrolling / register-allocation
decisions and prevents a pathological 100+ MP image from silently ballooning the
kernel. The cap is only reached for images beyond the 24 MP target envelope anyway.

**Exposing sigma later.** If a "Radius" slider is ever added, it becomes a second
`ClarityParams` field that scales `SIGMA_FRACTION` — purely additive, no shape change
to the registration or the pass list.

**Blur kernel size.** At `SIGMA_FRACTION = 0.02` on a 24 MP image (~6000 px wide),
σ ≈ 120 texels, radius ≈ 360 texels per tap direction. Separable: 2 × 360 = 720 taps
per output pixel (H pass + V pass combined). On 24 MP M4 Pro, ~0.4 ms per pass × 2
blur passes + ~0.05 ms combine = ~0.85 ms. Verifiable against the warm perf test.

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

**Edges come from `Source`, not `smoothed`.** The smooth pass intentionally erases the
edges the user wants outlined. Computing Sobel on the original input preserves the
faithful edge structure of the photograph, consistent with the Kyprianidis/XDoG toon
literature.

**Params:**

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CartoonParams {
    pub strength: f32,        // [0.0, 1.0] — 0 = original, 1 = full cartoon
    pub levels: f32,          // posterization levels per channel ∈ [2.0, 16.0]
    pub edge_threshold: f32,  // Sobel-magnitude cutoff ∈ [0.0, 1.0]
    pub edge_softness: f32,   // width of smoothstep ramp above threshold ∈ [0.01, 0.5]
    pub edge_darkness: f32,   // how black the overlaid edges are ∈ [0.0, 1.0]
    pub _padding: [f32; 3],   // pad to 32 bytes (WebGPU uniform alignment)
}
```

Five sliders: Strength, Levels, Edge Threshold, Edge Softness, Edge Darkness. Defaults:

| Slider         | Default | Rationale                                                         |
|----------------|---------|-------------------------------------------------------------------|
| Strength       | 0.0     | Zero-amount identity, consistent with every other shader.         |
| Levels         | 8.0     | Mid-range; visible banding without looking alien.                 |
| Edge Threshold | 0.15    | Above typical Sobel noise floor on clean photos.                  |
| Edge Softness  | 0.10    | 10% ramp above threshold — crisp but not aliased.                 |
| Edge Darkness  | 1.0     | Cartoon without edges is just posterize — 1.0 makes it obvious.   |

**Locked pass math:**

```
SIGMA_FRACTION_SMOOTH = 0.015     // larger than Clarity's 0.02? No — see note below
RADIUS_CAP = 256

// smooth_h / smooth_v — separable Gaussian, Rec.709 luma independent per-channel
sigma  = SIGMA_FRACTION_SMOOTH * f32(max(dims.x, dims.y))
radius = min(i32(ceil(3.0 * sigma)), RADIUS_CAP)

// quantize — linear-light per-channel floor-quantization.
// IMPORTANT: quantization runs in linear-light space (same as the rest of the pipeline).
// Banding boundaries therefore fall at energy-uniform intervals, which differs visibly
// from sRGB-gamma quantization (e.g., Photoshop Posterize). For a variant with
// sRGB-space quantization, see specs/tech_debt.md "Cartoon (sRGB-quantization variant)."
let L = floor(clamp(params.levels, 2.0, 16.0))
let quantized_rgb = clamp(floor(smoothed.rgb * L) / (L - 1.0), 0.0, 1.0)

// edges — 3x3 Sobel on Rec.709 luma of Source, with user-controlled threshold + ramp
let luma = dot(sample.rgb, vec3<f32>(0.2126, 0.7152, 0.0722))
let mag  = length(vec2<f32>(sobel_x(luma_3x3), sobel_y(luma_3x3)))
let ramp_end = clamp(params.edge_threshold + params.edge_softness, 0.0, 2.83)
let edge = smoothstep(params.edge_threshold, ramp_end, mag)
// Store as single-channel mask in .r; alpha = 1, other channels unused.

// combine — 3-input final pass
let color_base = mix(src.rgb, quant.rgb, params.strength)
let darken     = 1.0 - params.edge_darkness * edges.r
let out_rgb    = clamp(color_base * darken, 0.0, 1.0)
textureStore(output_texture, coord, vec4<f32>(out_rgb, src.a))
```

Note: Cartoon's smooth σ at 1.5% is *smaller* than Clarity's 2.0%. Clarity's blur is an
intermediate used to extract high-frequency detail; Cartoon's is the final color base,
so overshooting its radius erases too much structure the user wanted to keep. The
numbers were selected independently for what each pass is for, not scaled from each
other.

**Slider-extrema behavior (locked, asserted in tests):**

| Slider state                                      | Output                                            |
|---------------------------------------------------|---------------------------------------------------|
| `strength=0, edge_darkness=0`                     | Pixel-identical to input (identity).              |
| `strength=1, edge_darkness=0`                     | Pure posterized smoothed image, no edges overlaid.|
| `edge_darkness=1` on a strong edge (`edge_mask=1`)| `darken=0` → output pure black at that pixel.     |
| `edge_threshold=1.0`                              | Sobel magnitude never reaches threshold → no edges darkened. |
| `levels=2, strength=1`                            | 2 bands per channel on the smoothed image.        |

The `test_cartoon_zero_strength_is_identity` row in the shader-level test matrix
tightens accordingly: set both `strength=0.0` and `edge_darkness=0.0`, assert
pixel-identical to input (drop the ±128 tolerance — the locked formula is exact
identity at these parameters).

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
  scratch pool is empty at that shader's dims. Confirms single-pass shaders do not
  allocate scratch.
- `test_multi_pass_scratch_recycling_within_shader` — a 2-pass test shader whose second
  pass copies its scratch input to `Final`. Run `apply` twice at the same dims.
  Assertions: `scratch_pool_len(dims) == 1` after both runs (free list holds one
  texture that was returned and then re-borrowed); `scratch_pool_handle(dims, 0)` is
  the same raw pointer after run 1 and after run 2 (proves the physical texture was
  re-used, not reallocated).
- `test_multi_pass_scratch_shared_across_shaders` — run two distinct test multi-pass
  shaders back-to-back at the same dims, each needing 2 scratches. Assertions:
  `scratch_pool_len(dims) == 2` after both runs (peak footprint does not grow when the
  second shader runs); the two `scratch_pool_handle(dims, i)` pointers captured after
  run 1 match the pointers after run 2 — the second shader borrowed the same physical
  textures the first shader returned. Regression guard on the shared-pool invariant.
- `test_multi_pass_image_resize_drops_pool` — `apply` once at 4×4, then once at 8×8;
  assert the pool has no lingering 4×4 entries (check `scratch_pool_len((4,4)) == 0`).
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
- `test_all_registered_pass_lists_validate` — walks
  `inventory::iter::<ShaderRegistration>()` and runs `validate_pass_list` on every
  registered `RuntimeShader.passes`. Safety net for the const-fn validator: catches
  malformed pass lists even in shaders that ship without dispatch tests, and guarantees
  coverage if the const-fn tier is ever relaxed. (See "Registration-time validation of
  multi-pass pass lists" for the rules it enforces.)
- `test_validate_pass_list_rejects_final_in_middle`,
  `test_validate_pass_list_rejects_missing_scratch_write`,
  `test_validate_pass_list_rejects_duplicate_scratch_output` — direct unit tests on the
  `validate_pass_list` `const fn` driven by fixture pass lists. One test per violation
  class, following the single-behavior rule. Complements tier 2 by proving each rule
  independently without relying on a malformed shader being registered.

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
| `test_clarity_scratch_pool_reuses_across_runs` | run Clarity twice at same dims                     | `scratch_pool_len(dims) == 2` both times; same two `wgpu::Texture` pointers re-borrowed on the second run |

**Cartoon** (mirrors the same structure):

| Test name                                        | Setup                                             | Assertion                                                        |
|--------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------|
| `test_cartoon_registry_entry_exists`             | —                                                 | `registry_by_id("cartoon").is_some()`                            |
| `test_cartoon_registry_metadata`                 | —                                                 | name, 5 sliders (Strength, Levels, Edge Threshold, Edge Softness, Edge Darkness), `passes.len() == 5` |
| `test_cartoon_make_uniform_known_value`          | `reg.make_uniform(&[0.5, 8.0, 0.2, 0.1, 0.8])`    | bytes equal `bytemuck::bytes_of(&CartoonParams { strength: 0.5, levels: 8.0, edge_threshold: 0.2, edge_softness: 0.1, edge_darkness: 0.8, _padding: [0.0; 3] })` |
| `test_cartoon_zero_strength_and_zero_edge_darkness_is_identity` | solid gradient, `strength=0.0`, `edge_darkness=0.0` | output **pixel-identical** to input (every channel equal, no tolerance) |
| `test_cartoon_full_strength_reduces_unique_colors` | smooth gradient, `strength=1.0`, `levels=4`, `edge_darkness=0.0` | unique pixel values in output < unique values in input (posterization works; edges disabled so only `levels` matters) |
| `test_cartoon_edges_darken_high_gradient_pixels` | sharp black/white edge, `edge_darkness=1.0`, `strength=0.0`, `edge_threshold=0.1`, `edge_softness=0.1` | pixels along the edge are darker in the output than in the input |
| `test_cartoon_higher_edge_softness_widens_edge_band` | sharp edge, `edge_threshold=0.5`, compared at `edge_softness=0.05` vs `edge_softness=0.3` | the count of pixels where `darken > 0` is strictly greater in the higher-softness run (ramp widens the edge band) |
| `test_cartoon_no_edges_below_threshold`          | smooth gradient with no edges, `edge_threshold=1.0` | output equals pure-posterized version (no edge darkening applied at any pixel) |
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

### How to use this section

An agent invoked with "Implement PR X from `specs/multi-pass-plan.md`" should be able
to execute the numbered PR end-to-end from the sections below. Each PR has the same
shape:

1. **Prerequisites** — what must already be merged / true.
2. **Required reading** — specific sections of this plan and other specs that define
   the contracts the PR depends on. Read these *before* editing.
3. **Files** — exhaustive add / modify list with per-file scope.
4. **Implementation details** — concrete type signatures, macro expansion shapes,
   formulas, and code skeletons. If a decision has been locked, it is recorded here —
   do not re-open it.
5. **Tests** — exact test names, setup, and assertions. Each test follows `AGENTS.md`'s
   single-behavior rule.
6. **Acceptance commands** — literal shell commands that must pass before the PR is
   considered done.
7. **Out of scope** — explicit guardrails so scope does not drift.

Before any PR: read `AGENTS.md` and `specs/execution_model.md` § 2 ("Clean Slate
Replay"). Every PR must end with `cargo fmt --all`, `cargo clippy --all-targets`, and
`cargo test` all clean.

---

### PR 0 — Prototype: validate the unified path on one shader

**Prerequisites:** branched off `main`. No other PRs depend on this one — PR 0 is a
throwaway spike that produces a go/no-go signal for PR 1.

**Required reading:**

- This whole plan, with particular attention to:
  - § "Core abstractions" (`PassInput`, `PassOutput`, `PassDef`, `MultiPassShaderMeta`,
    `RuntimeShader`, the two registration macros, const-fn validator).
  - § "Renderer changes" (`PipelineCache` change, scratch pool, `apply_passes` steps,
    single-pass fast path).
  - § "Architecture decisions" items 1–10.
- `specs/multi-pass-research.md` § "Option C" and § "The one gotcha".
- `specs/adding_a_shader.md` (current single-pass pattern).
- `bdip_core/src/gpu/shaders/brightness/mod.rs` and
  `bdip_core/src/gpu/shaders/brightness/brightness.wgsl` (the migration target).
- `bdip_core/src/gpu/pipeline.rs` (`Renderer::apply`, `PipelineCache`).
- `bdip_core/src/gpu/shaders/mod.rs` (current `ShaderMeta`, `ShaderRegistration`,
  `registry_by_id`).

**Files (spike; discarded after PR 1 lands):**

Modify:

- `bdip_core/src/gpu/shaders/mod.rs`:
  - Add `PassInput`, `PassOutput`, `PassDef`, `MultiPassShaderMeta`, `RuntimeShader`.
    Definitions are copied verbatim from § "Core abstractions".
  - Keep `ShaderMeta` exactly as today.
  - Add `register_single_pass_shader!` and `register_multi_pass_shader!` macros; only
    the single-pass one is exercised this PR.
  - Change `registry_by_id` to return `&'static RuntimeShader`.
  - Add the `validate_pass_list` `const fn` and have
    `register_multi_pass_shader!` invoke it via `const _: () = ...`. Not exercised by
    the single-pass migration but must compile.
- `bdip_core/src/gpu/shaders/brightness/mod.rs`: replace the `inventory::submit!` call
  with `register_single_pass_shader! { meta: ShaderMeta { ... }, constructor: |values|
  Box::new(BrightnessParams::from_values(values)) }`. The `ShaderMeta` literal is
  unchanged — same four fields (`id`, `display_name`, `wgsl_source`, `param`).
- `bdip_core/src/gpu/pipeline.rs`: rewrite `Renderer::apply` as the unified
  `apply_passes` pass-list loop per § "Renderer changes" § "Renderer::apply dispatch".
  Add the `scratch_pool` field (typed as in § "Scratch pool"), the free-list
  borrow/return discipline, and the relabel-on-borrow mitigation. Do not implement the
  single-pass fast path yet (speculative — add only if criterion #3 fails).
- Temporary compat glue for the other 10 shaders — the minimum needed so the crate
  compiles and existing tests pass. The concrete shape (e.g., a `submit_legacy!` macro
  that wraps the old `inventory::submit!` form into a `RuntimeShader`) is left to the
  implementer since it is discarded in PR 1.

Do **not** modify `brightness.wgsl` or any other `.wgsl` file.

**Implementation details:**

- `PassInput` and `PassOutput` definitions — copy verbatim from § "Core abstractions".
- `PassDef` / `MultiPassShaderMeta` / `RuntimeShader` — copy verbatim from § "Core
  abstractions".
- `register_single_pass_shader!` expansion — per § "Core abstractions" § "Registration
  macros". The expansion synthesizes a `RuntimeShader` whose `passes` is a 1-element
  slice: `&[PassDef { label: META.id, wgsl_source: META.wgsl_source,
  inputs: &[PassInput::Source], output: PassOutput::Final }]`.
- `apply_passes` steps — implement exactly the 8 steps in § "Renderer::apply dispatch".
- `PipelineCache` — `HashMap<&'static str, Vec<CachedPipeline>>` per § "PipelineCache".
  Single-pass compiles into a length-1 vec.

**Evaluation criteria (all three must be green before opening PR 1):**

1. **Macro expands and registers correctly.**
   - `cargo build -p bdip_core` succeeds on stable Rust.
   - `registry_by_id("brightness").unwrap().passes` has length 1, with
     `inputs == &[PassInput::Source]` and `output == PassOutput::Final`.
   - `brightness/mod.rs` contains no `PassDef`, `PassInput`, or `PassOutput` tokens
     (verify via `grep -nE 'PassDef|PassInput|PassOutput' bdip_core/src/gpu/shaders/brightness/mod.rs`
     returning nothing).
2. **Zero WGSL diff.** `git diff` on
   `bdip_core/src/gpu/shaders/brightness/brightness.wgsl` is empty.
3. **Warm-path 24 MP performance within +5% of baseline.**
   - Baseline: run `cargo test --release -p bdip_core -- --ignored
     test_perf_gpu_roundtrip_24mp` on the parent commit, record 20-iteration warm-path
     `execute` mean.
   - Post-change: run the same command on the spike commit.
   - Assertion: warm mean ≤ 1.05 × baseline mean.
   - If the regression exceeds 5%, add the single-pass fast path from § "Renderer
     changes" § "Single-pass fast path" inside `apply_passes` and re-measure. The fast
     path must restore parity or the criterion fails.

**Acceptance commands:**

```
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test -p bdip_core
cargo test --release -p bdip_core -- --ignored test_perf_gpu_roundtrip_24mp
```

**Exit:**

- All three criteria green → the spike becomes the starting point for PR 1 (migrate
  the remaining 10 shaders, drop the compat shim, add the full infrastructure test
  suite).
- Any criterion red → stop. Document the failure in an issue and revisit this plan;
  do not open PR 1.

**Out of scope:**

- Multi-pass shaders (Clarity, Cartoon) — PRs 2 and 3.
- Migrating shaders other than brightness — PR 1.
- Infrastructure test suite beyond what criterion #1 needs — PR 1.
- Spec updates to `adding_a_shader.md` — PR 1.

**Reporting:** capture the warm-path mean numbers and whether the fast path was
needed, in the PR 1 description (not as a separate doc).

---

### PR 1 — Multi-pass infrastructure + existing-shader migration

**Prerequisites:** PR 0 green (all three evaluation criteria passed). PR 0's spike diff
is the starting point; PR 1 finishes the migration and adds tests.

**Required reading:**

- This whole plan.
- `bdip_core/src/gpu/shaders/*/mod.rs` for all 11 existing single-pass shaders (each is
  a mechanical migration target).
- `specs/adding_a_shader.md` — this is the spec being rewritten as part of this PR.

**Files:**

Add: none.

Modify:

- `bdip_core/src/gpu/shaders/mod.rs` — finalized versions of the types and macros from
  PR 0. Remove the temporary compat shim. Ensure `registry_by_id` returns
  `&'static RuntimeShader` and nothing else.
- `bdip_core/src/gpu/pipeline.rs`:
  - `PipelineCache` = `HashMap<&'static str, Vec<CachedPipeline>>`.
  - `Renderer::scratch_pool: HashMap<(u32, u32), Vec<wgpu::Texture>>`.
  - `Renderer::apply` is the single unified `apply_passes` dispatcher.
  - Single-pass fast path: include only if PR 0 measured a regression and required it.
    If included, document inline that it is a shape-preserving optimization and not a
    resurrected `Single`/`MultiPass` branch.
  - New private helper `build_pass_bind_group_layout(device, input_count) ->
    wgpu::BindGroupLayout` that derives group 0 from `input_count`.
  - `#[cfg(test)]` accessors: `scratch_pool_len((u32, u32)) -> usize` and
    `scratch_pool_handle((u32, u32), usize) -> Option<*const wgpu::Texture>` per
    § "Test-only accessor for pool introspection".
- `bdip_core/src/gpu/shaders/{brightness,contrast,exposure,grayscale,highlights,invert,saturation,shadows,temperature,tint,vignette}/mod.rs`
  — swap each `inventory::submit! { ShaderRegistration { ... } }` for the
  `register_single_pass_shader! { meta: ShaderMeta { ... }, constructor: |values|
  Box::new(<Params>::from_values(values)) }` form. The `ShaderMeta` literal is
  unchanged. No WGSL file modifications.
- `specs/adding_a_shader.md`:
  - Update the single-pass example to show `register_single_pass_shader! { ... }` (the
    `ShaderMeta` fields inside are unchanged from today).
  - Add a new "Multi-pass shaders" section covering `MultiPassShaderMeta`,
    `register_multi_pass_shader!`, the position-indexed binding contract (input
    bindings at `@binding(0..N-1)`, destination at `@binding(N)`, uniform at
    `@group(1) @binding(0)`), the shared-uniform alignment rule (every `.wgsl` file
    declares the full struct), the data-dependent loop bound convention
    (`RADIUS_CAP`), and a pointer to the const-fn validator's error messages.

**Implementation details:**

- Types — final versions of `PassInput`, `PassOutput`, `PassDef`, `MultiPassShaderMeta`,
  `RuntimeShader` per § "Core abstractions".
- `validate_pass_list` — `const fn` enforcing the three rules in § "Registration-time
  validation of multi-pass pass lists". Panics in const context on violation. Write
  `validate_pass_list` as pure `const fn` taking `&[PassDef]` and returning `()`; use
  byte-level comparison on `s.as_bytes()` for scratch-name equality.
- `register_multi_pass_shader!` expansion emits `const _: () =
  validate_pass_list(PASSES);` next to the `inventory::submit!` call, so misuse fails
  `cargo build` at the shader's own `mod.rs`.
- `apply_passes` — implement all 8 steps from § "Renderer::apply dispatch". Pay
  particular attention to:
  - Step 3: distinct `PassOutput::Scratch(name)` set (use a small `Vec<&'static str>`
    or `heapless` ish structure — 4 scratches is the current maximum).
  - Step 5: `dispatch_workgroups(ceil(width / 16), ceil(height / 16), 1)` at the
    Transform's input dims for every pass.
  - Step 7: every borrowed texture is returned to the pool's free list on exit, even
    on early `?` returns (use a guard pattern or explicit return before yielding the
    `Final` texture).
- Label mitigations (§ "Mitigating the debugging-label cost") — V1 ships tiers (1)
  (relabel on borrow) and (2) (debug-build `#name` counter suffix). Tier (3) is
  deferred.

**Tests shipped (add or verify each; see § "Infrastructure tests"):**

- `test_single_pass_macro_round_trips` (pre/post migration byte-identical for
  brightness).
- `test_single_pass_skips_scratch_pool`.
- `test_multi_pass_scratch_recycling_within_shader` (2-pass copy fixture; uses
  `scratch_pool_len` + `scratch_pool_handle`).
- `test_multi_pass_scratch_shared_across_shaders` (two distinct 2-pass fixtures;
  asserts same pointers re-borrowed).
- `test_multi_pass_image_resize_drops_pool`.
- `test_multi_pass_final_output_correctness` (2-pass identity copy returns
  pixel-identical input).
- `test_pipeline_cache_compiles_per_pass`.
- `test_position_indexed_bindings_three_inputs` (test fixture shader with 3-input
  pass, explicit binding regression guard).
- `test_single_pass_macro_synthesizes_one_pass_def`.
- `test_all_registered_pass_lists_validate` (walks
  `inventory::iter::<ShaderRegistration>()`).
- `test_validate_pass_list_rejects_final_in_middle`,
  `test_validate_pass_list_rejects_missing_scratch_write`,
  `test_validate_pass_list_rejects_duplicate_scratch_output` (direct unit tests on the
  `const fn`).

All existing per-shader tests (≥50 of them) continue to pass unchanged. `cargo test`
must be fully green; no test is moved to `#[ignore]`.

**Acceptance commands:**

```
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test -p bdip_core
cargo test --release -p bdip_core -- --ignored test_perf_gpu_roundtrip_24mp
# Sanity: confirm no shader mod.rs references pass vocabulary
grep -rn "PassDef\|PassInput\|PassOutput" bdip_core/src/gpu/shaders/ \
    | grep -v mod.rs:  # allowed only in shaders/mod.rs
```

**Review focus (capture in PR description):**

- `PassDef` / `MultiPassShaderMeta` shape and macro expansions (the public contract).
- `register_single_pass_shader!` expansion proves byte-identical `ShaderMeta` fields.
- Position-indexed bind-group construction handles N=1, 2, 3 inputs correctly.
- Scratch-pool borrow/return is leak-free (check via `scratch_pool_len` before/after
  tests).
- Const-fn validator compile-error messages are actionable.
- Single-pass fast path, if present, is localized inside `apply_passes` and not a
  separate public entry point.

**Rollback characteristics:** reverting this PR removes multi-pass infrastructure but
leaves the single-pass shaders in the old `inventory::submit!` form. No user-visible
change either way.

**Out of scope:**

- Real multi-pass shaders (Clarity, Cartoon) — PRs 2 and 3.
- Cross-shader integration tests — PR 4.
- Perf assertions beyond the PR 0 criterion — PR 4.

---

### PR 2 — Clarity shader

**Prerequisites:** PR 1 merged. Multi-pass infrastructure, the const-fn validator, and
the updated `adding_a_shader.md` guidance exist on `main`.

**Required reading:**

- This plan, § "Clarity" (pass list, params, locked sigma formula, blur kernel size,
  extrema behavior).
- `specs/some_shaders.md` — the Clarity row with the canonical
  `C_hp = C_in - C_blurred`, `C_out = C_in + C_hp * u_Clarity * W_mid` formulas and
  the midtone-weight description.
- `specs/adding_a_shader.md` § "Multi-pass shaders" (as written in PR 1).
- `bdip_core/src/gpu/shaders/vignette/mod.rs` as a reference for shader-test style
  (single-behavior tests using `make_solid_image` + `roundtrip` helpers).

**Files:**

Add:

- `bdip_core/src/gpu/shaders/clarity/mod.rs` — `ClarityParams`, `TransformShader` impl
  (or whatever PR 1's macro expects), `register_multi_pass_shader!` block, test module.
- `bdip_core/src/gpu/shaders/clarity/blur_h.wgsl` — separable Gaussian, horizontal.
- `bdip_core/src/gpu/shaders/clarity/blur_v.wgsl` — separable Gaussian, vertical.
- `bdip_core/src/gpu/shaders/clarity/combine.wgsl` — 2-input combine pass (reads
  `Source` and `Scratch("v")`).

Modify:

- `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod clarity;`.
- `specs/some_shaders.md` — update the Clarity row to note it ships as multi-pass
  (separable Gaussian + combine) and reference this plan.

**Implementation details:**

`ClarityParams`:

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClarityParams {
    pub amount: f32,          // u_Clarity ∈ [-1.0, 1.0]
    pub _padding: [f32; 3],   // pad to 16 bytes
}
```

Slider: `SliderDef { name: "Amount", min: -1.0, max: 1.0, default: 0.0 }`.

`PassDef` list (3 passes):

```
blur_h:  inputs=[Source],                  output=Scratch("h")
blur_v:  inputs=[Scratch("h")],            output=Scratch("v")
combine: inputs=[Source, Scratch("v")],    output=Final
```

WGSL — all three files declare:

```wgsl
struct ClarityParams {
    amount:   f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}
@group(1) @binding(0) var<uniform> params: ClarityParams;
```

Locked sigma / kernel code (identical in `blur_h.wgsl` and `blur_v.wgsl`, differing
only in tap direction):

```wgsl
const SIGMA_FRACTION: f32 = 0.02;
const RADIUS_CAP: i32 = 256;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let sigma  = SIGMA_FRACTION * f32(max(dims.x, dims.y));
    let radius = min(i32(ceil(3.0 * sigma)), RADIUS_CAP);
    let two_sigma_sq = 2.0 * sigma * sigma;

    var accum: vec4<f32> = vec4<f32>(0.0);
    var weight_sum: f32 = 0.0;
    let coord = vec2<i32>(gid.xy);
    for (var t: i32 = -radius; t <= radius; t = t + 1) {
        let offset = vec2<i32>(t, 0);   // blur_v uses vec2<i32>(0, t)
        let s = textureLoad(input_texture, clamp(coord + offset, vec2<i32>(0), vec2<i32>(dims) - 1), 0);
        let w = exp(-f32(t * t) / two_sigma_sq);
        accum = accum + s * w;
        weight_sum = weight_sum + w;
    }
    let out = accum / weight_sum;
    textureStore(output_texture, coord, vec4<f32>(out.rgb, textureLoad(input_texture, coord, 0).a));
}
```

Alpha is copied through from the input pixel — the blur does not smear alpha.

Combine pass (`combine.wgsl`) — 2-input, per `some_shaders.md` Clarity row:

```wgsl
@group(0) @binding(0) var input_source:  texture_2d<f32>;
@group(0) @binding(1) var input_blurred: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ClarityParams;

fn midtone_weight(luma: f32) -> f32 {
    // Peaks at 0.5 mid-gray; falls smoothly to 0 at 0.0 and 1.0.
    // One standard form: 1 - (2*luma - 1)^2  ∈ [0, 1].
    let t = 2.0 * luma - 1.0;
    return clamp(1.0 - t * t, 0.0, 1.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord   = vec2<i32>(gid.xy);
    let src     = textureLoad(input_source, coord, 0);
    let blurred = textureLoad(input_blurred, coord, 0);

    let c_hp    = src.rgb - blurred.rgb;
    let luma    = dot(src.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let w_mid   = midtone_weight(luma);
    let out_rgb = src.rgb + c_hp * params.amount * w_mid;

    textureStore(output_texture, coord, vec4<f32>(clamp(out_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), src.a));
}
```

Note: Clarity's output is clamped to `[0, 1]` because the formula can legitimately
exceed the range on saturated pixels. `Rgba16Float` still stores any overflow losslessly
until the readback clamp, but clamping here matches the reference formula's intent.

**Tests shipped (exact names from § "Shader-level tests (PRs 2 & 3)"):**

- `test_clarity_registry_entry_exists`
- `test_clarity_registry_metadata` — asserts `display_name == "Clarity"`,
  `param == Sliders([{"Amount", -1.0, 1.0, 0.0}])`, `passes.len() == 3`.
- `test_clarity_make_uniform_known_value` — `reg.make_uniform(&[0.5])` returns bytes
  equal to `bytemuck::bytes_of(&ClarityParams { amount: 0.5, _padding: [0.0; 3] })`.
- `test_clarity_zero_amount_is_identity` — 16×16 solid mid-gray, `amount = 0.0`; every
  output pixel within ±64 u16 of input. (Clarity is never *bit*-exact at amount=0 due
  to the blur roundtrip; ±64 matches other shader tests' tolerance.)
- `test_clarity_positive_amount_increases_contrast_on_edge` — step image; pixels just
  inside the edge diverge more from the mean at `amount=0.5` than at `amount=0.0`.
- `test_clarity_negative_amount_softens_edge` — same step image; edge transition at
  `amount=-0.5` is softer than at `amount=0.0` (compare inter-band pixel differences).
- `test_clarity_alpha_preserved` — 4×4 solid mid-gray, `amount=0.5`; every output
  alpha == 65535.
- `test_clarity_deterministic` — same inputs run twice produce pixel-identical output.
- `test_clarity_scratch_pool_reuses_across_runs` — run Clarity twice at same dims;
  `scratch_pool_len(dims) == 2` both times; `scratch_pool_handle(dims, 0)` and
  `(dims, 1)` produce the same raw pointers on the second run.

**Acceptance commands:**

```
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test -p bdip_core
cargo test test_shader_registry_no_duplicate_ids
```

**Review focus (capture in PR description with a 24 MP before/after screenshot):**

- Gaussian kernel math (σ derivation, normalization by `weight_sum`, 3σ truncation,
  `RADIUS_CAP`).
- Combine formula matches `some_shaders.md` Clarity row exactly.
- Midtone-weight curve visibly attenuates strong highlights/shadows (see the step-image
  test behavior).
- Warm-path timing on 24 MP — reportable in the PR description but not asserted here
  (perf assertion is PR 4).

**Out of scope:**

- Exposing `blur_sigma` / radius as a second slider.
- Cross-shader tests (PR 4).
- `specs/transformations_reference.md` update — Clarity already exists there.

---

### PR 3 — Cartoon shader

**Prerequisites:** PR 2 merged. Clarity exists as a working reference for the
multi-pass pattern. `adding_a_shader.md` § "Multi-pass shaders" exists.

**Required reading:**

- This plan, § "Cartoon" (pass list, `CartoonParams`, defaults, locked pass math,
  slider-extrema behavior).
- This plan, § "Bind-group contract (multi-pass passes)" — especially the shared-uniform
  alignment rule.
- `specs/tech_debt.md` entry "Cartoon (sRGB-quantization variant)" — `quantize.wgsl`
  must carry an inline comment pointing at it.
- `bdip_core/src/gpu/shaders/clarity/` (reference for the pattern).

**Files:**

Add:

- `bdip_core/src/gpu/shaders/cartoon/mod.rs` — `CartoonParams`, `register_multi_pass_shader!`
  block, test module.
- `bdip_core/src/gpu/shaders/cartoon/smooth_h.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/smooth_v.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/quantize.wgsl` — with inline comment explaining
  linear-light quantization and pointing at `specs/tech_debt.md` "Cartoon
  (sRGB-quantization variant)".
- `bdip_core/src/gpu/shaders/cartoon/edges.wgsl`
- `bdip_core/src/gpu/shaders/cartoon/combine.wgsl` — 3-input pass, first in the tree.

Modify:

- `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod cartoon;`.
- `specs/transformations_reference.md` — add a "Stylization" heading with a Cartoon
  entry that references this plan.

**Implementation details:**

`CartoonParams`:

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CartoonParams {
    pub strength:       f32,
    pub levels:         f32,
    pub edge_threshold: f32,
    pub edge_softness:  f32,
    pub edge_darkness:  f32,
    pub _padding:       [f32; 3],   // 32 bytes total
}
```

Sliders (in this order — tests depend on it):

```
SliderDef { name: "Strength",       min: 0.0,  max: 1.0,  default: 0.0 }
SliderDef { name: "Levels",         min: 2.0,  max: 16.0, default: 8.0 }
SliderDef { name: "Edge Threshold", min: 0.0,  max: 1.0,  default: 0.15 }
SliderDef { name: "Edge Softness",  min: 0.01, max: 0.5,  default: 0.10 }
SliderDef { name: "Edge Darkness",  min: 0.0,  max: 1.0,  default: 1.0 }
```

`PassDef` list (5 passes):

```
smooth_h: inputs=[Source],                                          output=Scratch("sh")
smooth_v: inputs=[Scratch("sh")],                                   output=Scratch("smooth")
quantize: inputs=[Scratch("smooth")],                               output=Scratch("quant")
edges:    inputs=[Source],                                          output=Scratch("edges")
combine:  inputs=[Source, Scratch("quant"), Scratch("edges")],      output=Final
```

All five WGSL files declare the full `CartoonParams` struct verbatim (see § "Bind-group
contract (multi-pass passes)" for the exact WGSL).

Locked pass math — see § "Cartoon" § "Locked pass math" for the formulas. Key points
for the implementer:

- `SIGMA_FRACTION_SMOOTH = 0.015`, `RADIUS_CAP = 256`.
- Quantize in **linear-light** space: `floor(smoothed.rgb * L) / (L - 1.0)` where
  `L = floor(clamp(params.levels, 2.0, 16.0))`. Include this exact comment in
  `quantize.wgsl`:

```wgsl
// Quantization runs in linear-light space (consistent with the rest of the pipeline).
// Bands fall at energy-uniform intervals, which differs visibly from sRGB-gamma
// quantization (e.g., Photoshop Posterize). An sRGB-space Cartoon variant is tracked
// in specs/tech_debt.md "Cartoon (sRGB-quantization variant)".
```

- Edges: Sobel on Rec.709 luma of **Source** (not `smoothed`). Write single-channel
  mask as `vec4<f32>(edge, 0.0, 0.0, 1.0)`.
- Edge shaping: `smoothstep(params.edge_threshold,
  clamp(params.edge_threshold + params.edge_softness, 0.0, 2.83), mag)`.
- Combine: `mix(src.rgb, quant.rgb, strength) * (1.0 - edge_darkness * edges.r)`,
  clamped to `[0, 1]`, alpha from `src.a`.

Bindings for `combine.wgsl` (the 3-input pass):

```wgsl
@group(0) @binding(0) var input_source:   texture_2d<f32>;   // Source
@group(0) @binding(1) var input_quant:    texture_2d<f32>;   // Scratch("quant")
@group(0) @binding(2) var input_edges:    texture_2d<f32>;   // Scratch("edges")
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CartoonParams;
```

The binding indices above must match the `inputs` order in `PassDef` exactly — this is
the production validation of the position-indexed discipline.

**Tests shipped (exact names from § "Shader-level tests (PRs 2 & 3)"):**

- `test_cartoon_registry_entry_exists`
- `test_cartoon_registry_metadata` — 5 sliders in declared order, `passes.len() == 5`.
- `test_cartoon_make_uniform_known_value` — `reg.make_uniform(&[0.5, 8.0, 0.2, 0.1,
  0.8])` returns bytes equal to
  `bytemuck::bytes_of(&CartoonParams { strength: 0.5, levels: 8.0, edge_threshold: 0.2,
  edge_softness: 0.1, edge_darkness: 0.8, _padding: [0.0; 3] })`.
- `test_cartoon_zero_strength_and_zero_edge_darkness_is_identity` — solid gradient,
  `strength=0.0`, `edge_darkness=0.0`; output **pixel-identical** to input (no
  tolerance — the formula is exact identity at these parameters).
- `test_cartoon_full_strength_reduces_unique_colors` — smooth gradient,
  `strength=1.0`, `levels=4`, `edge_darkness=0.0`; unique output values < unique input
  values.
- `test_cartoon_edges_darken_high_gradient_pixels` — sharp black/white edge,
  `edge_darkness=1.0`, `strength=0.0`, `edge_threshold=0.1`, `edge_softness=0.1`;
  edge-pixel luma in output < edge-pixel luma in input.
- `test_cartoon_higher_edge_softness_widens_edge_band` — sharp edge,
  `edge_threshold=0.5`; pixel-count where darken applied is strictly greater at
  `edge_softness=0.3` than at `edge_softness=0.05`.
- `test_cartoon_no_edges_below_threshold` — smooth gradient, `edge_threshold=1.0`;
  output equals pure-posterized version (no edge darkening anywhere).
- `test_cartoon_alpha_preserved` — 4×4 solid mid-gray; every output alpha == 65535.
- `test_cartoon_deterministic` — same params run twice → pixel-identical.
- `test_cartoon_three_input_combine_pass_binds_correctly` — test helper drives a
  synthetic 3-input scenario (different channel per input) and asserts the combine
  output contains contributions from all three (regression guard on binding positions).

**Acceptance commands:**

```
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test -p bdip_core
cargo test test_shader_registry_no_duplicate_ids
```

**Review focus (capture in PR description with a 24 MP before/after screenshot):**

- The 3-input combine's `@binding(0..3)` in WGSL matches the `PassDef` `inputs` order.
- Sobel kernel operates on `Source` (not smoothed).
- Linear-quantization comment present in `quantize.wgsl`.
- Slider-extrema tests pass with the *exact* locked formula — if any of them require
  tolerance tuning, pause and reconcile with § "Cartoon" § "Slider-extrema behavior"
  rather than loosening the assertion.

**Out of scope:**

- sRGB-quantization variant (tech-debt entry).
- Edge-detection on smoothed image.
- Cross-shader tests and perf assertions (PR 4).

---

### PR 4 — Cross-shader integration + performance guardrails

**Prerequisites:** PRs 2 and 3 merged. Clarity and Cartoon exist.

**Required reading:**

- This plan, § "Cross-shader integration tests (PR 4)" and § "Performance budget test".
- `bdip_core/src/gpu/shaders/cross_shader_tests.rs` — the current file to extend.
- `bdip_core/src/gpu/pipeline.rs` § `test_perf_gpu_roundtrip_24mp` — the perf test to
  extend or clone.

**Files:**

Modify:

- `bdip_core/src/gpu/shaders/cross_shader_tests.rs` — add the three chain tests below.
- `bdip_core/src/gpu/pipeline.rs` — extend the perf test (or add siblings) with the
  Clarity and Cartoon assertions below.

**Implementation details — exact tests to add:**

Cross-shader chain tests (extend `cross_shader_tests.rs`):

- `test_brightness_then_clarity` — apply Brightness(+0.2) then Clarity(+0.5) on a
  synthetic image; assertion: mean pixel brightness > mean pixel brightness after
  Brightness(+0.2) alone (Clarity does not cancel Brightness's lift).
- `test_clarity_then_vignette` — apply Clarity(+0.5) then Vignette (default) on a 16×16
  solid mid-gray image; assertion: no panic, `apply` returns an image of the expected
  dims, every output pixel has alpha == 65535.
- `test_cartoon_then_saturation` — apply Cartoon (defaults) then Saturation(1.0) on a
  smooth gradient; assertion: `unique_output_colors` is within ±5% of
  `unique_output_colors_cartoon_alone` (Saturation at 1.0 does not restore colors that
  Cartoon quantized away).

Performance assertions (extend / add siblings to `test_perf_gpu_roundtrip_24mp`):

- `test_perf_gpu_roundtrip_24mp_clarity` — 24 MP synthetic image, Clarity at
  `amount=0.5`, warm-path critical path mean over 20 iterations; assert `mean < 22.0
  ms` (20 ms readback baseline + ~1 ms Clarity + ~1 ms slack).
- `test_perf_gpu_roundtrip_24mp_cartoon` — same shape; assert `mean < 24.0 ms` (20 ms
  + ~2 ms Cartoon + ~2 ms slack).

Both new perf tests are `#[ignore]`-gated, same convention as the existing
`test_perf_gpu_roundtrip_24mp`. Their purpose is drift detection — they are soft
ceilings that fire before regressions reach production.

**Acceptance commands:**

```
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test -p bdip_core
cargo test --release -p bdip_core -- --ignored test_perf_gpu_roundtrip_24mp
cargo test --release -p bdip_core -- --ignored test_perf_gpu_roundtrip_24mp_clarity
cargo test --release -p bdip_core -- --ignored test_perf_gpu_roundtrip_24mp_cartoon
```

**Review focus:**

- Chain tests assert *relative* quantities (comparisons against "shader-alone"
  baselines), not absolute pixel values — this keeps them resilient to minor formula
  tweaks.
- Perf ceilings are labeled as soft guardrails in their test comments.
- No assertion is stronger than what the shader actually guarantees (e.g., the cartoon
  → saturation test allows ±5% drift).

**Rollback characteristics:** tests-only PR; reverting loses coverage but does not
change runtime behavior.

**Out of scope:**

- New shaders.
- Additional perf test matrices (e.g., 100 MP). A V1-checklist entry for NVIDIA
  portability spot-check already lives in § "Risks and open questions"; this PR does
  not implement it.

---

## Risks and open questions

- **Blur kernel portability across GPUs.** Large Gaussian kernels loop many texture reads;
  shader compilers may unroll differently. Validated on M4 Pro in PR 2's perf test; must
  also be spot-checked on a discrete NVIDIA GPU before V1 ship (not blocking for PR
  merges but must be on the V1 checklist).
- **Scratch pool growth if users stack multiple multi-pass shaders.** The shared pool
  (see "Renderer changes" § "Scratch pool") caps peak footprint at
  `max(scratches_per_shader) × texture_size`, so stacking more multi-pass Transforms at
  the same image size does not grow the pool beyond Cartoon's 4-scratch high-water mark
  (~740 MB at 24 MP). The remaining risk is a future shader that alone needs many more
  scratches than Cartoon; that is a per-shader design review, not a pool-level issue.
- **Parameter coupling for Clarity.** V1 hardcodes blur sigma; a later PR may expose it as
  a slider. Decision is not blocking.
- **Cartoon parameter defaults.** The defaults locked above (Strength 0.0, Levels 8.0,
  Edge Threshold 0.15, Edge Softness 0.10, Edge Darkness 1.0) are informed estimates
  and may be tuned on real photos during PR 3 review. Locked formula and slider set
  are not up for re-negotiation at that point; only numeric defaults.
- **`rustfmt` interaction with `register_single_pass_shader!` calls containing
  `include_str!(...)`.** Long `include_str!` calls sometimes cause awkward line breaks
  inside macro invocations. Non-blocking; can use `#[rustfmt::skip]` if needed on the
  affected lines.
- **Macro hygiene around the synthesized `PassDef` slice.** The macro emits a
  `&[PassDef { ... }]` literal bound to the caller's crate. PR 0 confirms that
  `inventory::submit!` accepts the expansion on stable Rust.
