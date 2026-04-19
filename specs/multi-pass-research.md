# Multi-Pass Shader Support — Research & Options

## Context

`bdip_core` currently supports **single-pass** shader transforms. Each `Transform` maps 1:1
onto a single WGSL compute dispatch (`Renderer::apply`) with a fixed bind-group layout:

| Group | Binding | Resource                                            |
|-------|---------|-----------------------------------------------------|
| 0     | 0       | Source texture (`texture_2d<f32>`, read)            |
| 0     | 1       | Destination storage texture (`rgba16float, write`)  |
| 1     | 0       | Uniform buffer (shader-specific params)             |

A shader is registered (decentralized, via `inventory::submit!`) by declaring a params struct
that implements `TransformShader`. `Renderer::apply` walks the registry by `shader_id`, lazily
compiles a `ComputePipeline`, allocates a fresh `Rgba16Float` destination texture the same
size as the input, and dispatches once. See `bdip_core/src/gpu/pipeline.rs` and
`specs/adding_a_shader.md`.

Three pending transforms don't fit this 1:1 model:

- **Clarity** (`specs/some_shaders.md`): `C_hp = C_in - C_blurred`, then
  `C_out = C_in + (C_hp * amount * W_mid)`. The `C_blurred` term requires a low-pass filter.
  A quality blur at radii that give useful "midtone local contrast" is a separable Gaussian
  (horizontal pass → vertical pass, two dispatches) — single-pass with a 2D kernel is O(r²)
  per pixel and infeasible for the radii Clarity needs (tens of pixels).
- **FilmGrainFBM** (`specs/film_grain_plan.md`): multi-octave gradient noise. Shippable as
  either an in-shader loop (single pass, 3–5× Perlin cost) or a true multi-pass approach
  sampling noise textures at different scales. The in-shader variant is viable today; the
  texture-sampled variant requires both multi-pass *and* auxiliary texture binding, which
  is a separate architectural axis covered in `film_grain_plan.md` (FilmGrainBlue).
- **Cartoon / toon filter** (new — not yet in an individual spec): stylization that makes
  a photograph look hand-drawn. The standard pipeline is (1) edge-preserving smoothing to
  flatten color regions, (2) color quantization/posterization of the smoothed image,
  (3) edge detection (Sobel or Difference-of-Gaussians), and (4) a combine pass that
  overlays the detected edges onto the quantized colors. Steps 1 and 3 are neighborhood
  operations that require real blur passes; the combine pass reads three inputs (original
  or quantized color layer, blurred layer for color regions, edges), which exercises the
  multi-input arity of the `PassDef` contract directly.

This document evaluates ways to express **multi-pass compute** for a single user-facing
`Transform`, and recommends an approach.

## Constraints carried from the existing design

1. **Clean Slate Replay.** Every slider interaction re-runs the full transform stack from the
   Pristine Texture. Any multi-pass support must be driven entirely on the GPU with no
   intermediate CPU readback — `download_presentation_buffer` clamps to `[0, 1]` and destroys
   `Rgba16Float` headroom. See `specs/execution_model.md` §2.
2. **History granularity.** `HistoryManager` stores one entry per user action, and
   `collapsing_adjacent_runs` reduces consecutive same-kind entries to the latest. A Clarity
   adjustment is *one* user intent. Splitting it into multiple history entries (one per internal
   pass) would break undo/redo semantics and the "absolute slider values" invariant.
3. **Zero-touch shader registration.** Shaders register themselves with `inventory::submit!`
   — no central list to edit. Any multi-pass extension must preserve this so contributors
   keep adding shaders by touching one directory.
4. **Performance envelope.** On the 24 MP warm path, `execute` is ~0.35 ms and `readback` is
   ~15.6 ms (see `test_perf_gpu_roundtrip_24mp`). Download is the bottleneck by ~45×. Adding
   2–5 compute dispatches per transform costs low single-digit milliseconds at most and
   does not move the interactive-latency needle. The correctness and readability costs of
   each option below therefore dominate.

## Options

### Option A — Decompose into multiple `Transform` entries at the call site

Treat "Clarity" as a macro that expands at UI time into a sequence of existing, simpler
`Transform` entries: e.g., `[BlurH(r), BlurV(r), ClarityCombine(amount)]`. No engine change;
the UI/history layer is responsible for grouping them.

**How it would work**

- Register `blur_h`, `blur_v`, `clarity_combine` as three ordinary single-pass shaders.
- In `BdipApp`, when the user picks "Clarity", push three `Transform` entries in one
  `HistoryEntry` batch.
- `HistoryManager` and collapsing logic are extended to understand "grouped" entries —
  either as a new variant (`Entry::Group(Vec<Transform>)`) or via a sentinel.

**Pros**

- Reuses the existing single-pass machinery verbatim. No changes in `pipeline.rs`.
- Each sub-shader is independently testable with the existing helpers.

**Cons — multiple, severe**

- **Breaks the collapsing invariant.** Adjacent-run collapsing assumes each history entry
  is atomic. A Clarity group interleaved with anything else cannot be collapsed by type,
  and adjacent Clarity groups would need group-aware collapsing logic.
- **Intermediate passes leak into the user model.** `BlurH` and `BlurV` of a Gaussian are
  not meaningful images on their own. Exposing them in the registry pollutes the sidebar's
  transform picker unless a new "internal / not-user-selectable" flag is added to
  `ShaderRegistration`. That flag is another pieces of hidden state that future contributors
  must remember.
- **Intermediate texture lifetime is wrong.** `Renderer::apply` allocates a fresh
  `Rgba16Float` destination texture per call. A three-transform Clarity would allocate three
  full-image textures (several hundred MB on a 24 MP image) instead of recycling one scratch
  texture. This is not a correctness bug but a measurable VRAM hit and an allocation churn
  problem that we don't have today.
- **Parameter coupling is clumsy.** Clarity's "amount" lives on the combine pass but the blur
  radius lives on the blur passes. Changing amount should not re-run the blur; with this model
  there's no easy way to express "these three transforms share a logical parameter source"
  without ad-hoc wiring.
- **Cross-cutting semantics for undo/redo.** A single Ctrl-Z should undo Clarity as a whole.
  The history layer would need to know each group is atomic, which re-introduces a bespoke
  notion of "compound transform" at a higher layer than the one where it belongs.

### Option B — Hardcoded per-shader branches in `Renderer`

Add `Renderer::apply_clarity(...)`, `Renderer::apply_fbm(...)` methods that bypass the
registry entirely and orchestrate their own multi-pass flow internally.

**Pros**

- Simplest possible code for the first multi-pass shader: one function, no abstraction.

**Cons**

- **Kills the `inventory`-based registry model for these shaders.** `Transform { shader_id:
  "clarity", ... }` would need special-cased dispatch inside `Renderer::apply`. The neat
  property that "adding a shader means adding a directory and one line in `shaders/mod.rs`"
  is lost for anything multi-pass.
- **Non-uniform shape.** `registry_by_id("clarity")` would return something but running it
  requires a different call path. Consumers (CLI, tests, UI) would need to branch.
- **Doesn't scale.** Every future multi-pass shader (separable blur for Bloom, unsharp mask,
  bilateral filter, chromatic aberration) adds another hand-coded method. We're already
  naming two on day one.

### Option C — Declarative multi-pass in `ShaderRegistration`

Treat a shader as an ordered list of passes, each of which is a standard-looking compute
dispatch with a named input/output. The engine owns scratch texture allocation and the
registry still drives everything.

**Shape of the extension**

```rust
pub enum PassInput {
    Source,              // the input to the Transform (output of previous Transform)
    Scratch(&'static str), // output of a prior pass in this same Transform
}

pub enum PassOutput {
    Scratch(&'static str), // intermediate — engine allocates/recycles scratch texture
    Final,                 // this pass writes the Transform's output texture
}

pub struct PassDef {
    pub label: &'static str,           // e.g., "blur_h"
    pub wgsl_source: &'static str,     // one WGSL file per pass
    pub input: PassInput,
    pub output: PassOutput,
    // (optional, for later: a second input binding, e.g., Clarity combine needs
    // both Source AND Scratch("blur_v"))
}

pub enum ShaderProgram {
    Single(&'static str),        // existing single-pass path; wgsl_source as today
    MultiPass(&'static [PassDef]),
}

pub struct ShaderMeta {
    pub id: &'static str,
    pub display_name: &'static str,
    pub program: ShaderProgram,
    pub param: ParamKind,
}
```

For Clarity, `MultiPass` would declare three passes:

1. `blur_h`: input=`Source`, output=`Scratch("h")`
2. `blur_v`: input=`Scratch("h")`, output=`Scratch("v")`
3. `combine`: inputs=[`Source`, `Scratch("v")`], output=`Final`

The combine pass reads two textures, so the bind-group contract grows to allow an optional
second source-texture binding for passes that need it. This is a small extension to the
current layout — still a fixed shape per pass kind (1-source or 2-source), still just
textures + one uniform buffer. No arbitrary bind-group configuration.

**Renderer changes**

`Renderer::apply` becomes a dispatcher over the program type:

- `ShaderProgram::Single`: existing code path.
- `ShaderProgram::MultiPass`: walk `passes`, allocate scratch textures from a per-`Renderer`
  pool keyed by `(width, height)`, run each compute pass with the declared bindings, return
  the `Final` texture.

Scratch textures are recycled across invocations: a Clarity apply at the same image size
reuses its `Scratch("h")` and `Scratch("v")` allocations on every slider tick — no
allocation churn. When the image size changes, the pool is dropped and rebuilt. This matches
how `present_tile_buffer` and `staging_buffer` are already managed in `pipeline.rs`.

The `PipelineCache` keys on shader_id today; for multi-pass it keys on `(shader_id,
pass_index)` or equivalently stores `Vec<CachedPipeline>` per shader. Either is a minor
adjustment.

**Registration**

Each multi-pass shader is a directory containing one `mod.rs` (params + `PassDef` array) and
one WGSL file per pass. The shader registers itself with the same `inventory::submit!`
macro. `specs/adding_a_shader.md` gains a "Multi-pass" section with the pattern.

**Pros**

- **Single Transform, single history entry.** `HistoryManager`, collapsing, and undo/redo
  are untouched.
- **Preserves the registry ergonomics.** Adding a multi-pass shader still means adding one
  directory; only the internal shape differs. `inventory::submit!` still works; no central
  list to update.
- **No intermediate readback.** All passes run on the GPU against `Rgba16Float` textures,
  preserving headroom across passes.
- **Correctly scoped scratch allocation.** Scratch textures are owned by the engine and
  recycled — not leaked as "fake transforms" the user model has to understand.
- **Extends cleanly to FilmGrainFBM-multi-pass** (if we ever want it in its "proper" form
  rather than the in-shader octave loop), to **Cartoon** (blur + edge-detect + combine
  chain), and to future multi-pass work (Bloom, unsharp mask, large-radius noise, bilateral
  filter). Declaring a pass list scales where hardcoded methods do not.
- **Single-pass shaders are unaffected.** They keep the `wgsl_source` one-liner; no
  migration required.

**Cons**

- The `TransformShader` trait / `ShaderRegistration` struct grow. `ShaderProgram` is a new
  variant surface. Slight readability cost at the registration layer, offset by the fact
  that the *extension point itself is declarative and self-documenting.*
- The bind-group contract grows to support 2-source passes. This is necessary for Clarity's
  combine pass anyway — there is no single-pass alternative.
- More moving parts in `Renderer::apply` (scratch pool, pass loop). The complexity is
  well-contained and mirrors patterns already present (`present_tile_buffer` management).

### Option D — In-shader fusing (no multi-pass; one fat WGSL)

Write Clarity as a single WGSL file that samples a neighborhood per pixel and approximates
the blur inline. For FBM, loop octaves within a single shader.

**Pros**

- Zero architectural change. Ships today.
- **Genuinely appropriate for FilmGrainFBM.** 3–5 Perlin octaves summed in one pass is a
  standard implementation and fits the current single-pass model perfectly. The
  `film_grain_plan.md` "option A" already acknowledges this. **FBM does not actually need
  multi-pass.**

**Cons for Clarity**

- **Quality ceiling.** A single-pass 2D blur large enough to give professional Clarity
  behavior (radius ≈ 30–60 px on a 24 MP image) costs thousands of texture reads per pixel.
  Even with a separable approximation done inside a loop, you pay for the full 2D kernel
  because one compute invocation can only read — not write — the image (no data sharing
  across invocations within a single pass). This is the reason separable Gaussian is
  universally implemented as two passes in production code.
- **Performance falls behind the competitors.** Lightroom / Capture One / darktable all do
  Clarity with a real blur pass (or downsampled blur). A single-pass approximation would
  either be slow (large kernel in one pass) or visibly lower quality (small kernel).

### Option E — Full render-graph abstraction

A general-purpose DAG where nodes declare inputs/outputs by name, the engine topologically
sorts, allocates transient resources, and dispatches. wgpu has a `wgpu-rs`-adjacent
precedent in `rend3`/`wgpu_hal` experiments.

**Pros**

- Most flexible long-term design.
- Handles arbitrary DAGs, not just linear multi-pass chains. Would accommodate aux-texture
  reads (blue noise), multi-output passes (e.g., bloom with bright-pass + blur chain + add).

**Cons**

- **Significant architectural investment** for a codebase that currently runs a fixed
  linear pipeline. The needs today are linear: for every multi-pass transform on the
  near-term roadmap (Clarity, optional FBM-multi-pass, future Bloom / unsharp mask), an
  ordered list of passes is sufficient.
- **Readability hit.** A render graph has intrinsic complexity (resource aliasing, lifetime
  analysis) that cannot be avoided even when the graphs are trivial. The project's current
  tech-debt log (`specs/tech_debt.md`) already flags over-complication at the public API
  surface; adding a render graph now would worsen that.
- YAGNI — build the DAG when a non-linear shader actually appears.

## Recommendation

**Adopt Option C (declarative multi-pass in `ShaderRegistration`) for Clarity and Cartoon.**
Clarity is the simplest real use (2-input combine); Cartoon is the concrete 3-input-combine
stress test that validates the position-indexed binding discipline from day one.

**Use Option D (in-shader octaves) for FilmGrainFBM.** FBM does not require multi-pass; the
octave loop is the standard implementation and it fits the existing registry with zero
architectural change. `specs/film_grain_plan.md` already identifies this (option A). Multi-pass
FBM only becomes relevant if we later add auxiliary-texture sampling (FilmGrainBlue), which
is an orthogonal blocker.

### Why Option C wins for Clarity

- **Correctness.** It preserves the Clean Slate Replay model byte-for-byte — passes are
  internal to a Transform, intermediate data never crosses the CPU boundary, and `Rgba16Float`
  headroom is maintained across all passes.
- **History/undo invariants are untouched.** A Clarity slider change is a single history
  entry, collapses correctly with other Clarity entries, and undoes atomically.
- **Readability.** Adding a multi-pass shader still means adding one directory. The
  declaration is structural (`PassDef` array) rather than imperative — a contributor reads
  the pass list top-to-bottom and understands the data flow without reading `pipeline.rs`.
  The engine-side complexity is localized to `Renderer::apply` and stays proportional to
  the feature's actual needs.
- **Performance matches the competition.** Separable Gaussian is the canonical Clarity blur
  in Lightroom, Capture One, darktable, and RawTherapee — implemented as two passes, same
  as Option C. We run at the same asymptotic cost, with the same quality ceiling. Readback
  dominates total latency (~15 ms vs. ~0.5 ms per compute pass on 24 MP M4 Pro), so adding
  two passes for Clarity costs ~1 ms in a 16 ms budget — negligible.
- **Scratch texture recycling prevents allocation churn.** The pattern mirrors existing
  cached buffers in `Renderer` (`present_tile_buffer`, `staging_buffer`). No new allocation
  strategy — just a third cached resource.
- **Forward-compatible.** If/when multi-pass FBM, Bloom, or unsharp mask lands, the same
  registration shape works with zero new infrastructure.

### What we lose by not picking Option E

A render graph would, in principle, let a shader consume two arbitrary prior-pass outputs
or fan out to multiple consumers. None of the transforms on the near-term roadmap need that.
If a future shader does (the most likely candidate is a layered tonemapper with shared
bright-pass intermediate, or bloom with a downsample chain), the `PassDef` array can be
upgraded incrementally — it's a strict subset of a DAG, and the migration path is laid out
in the next section.

### Migration path from C to E (if we ever need it)

Option C is a strict subset of Option E. A future migration is mechanical, not
architectural, because the core abstractions in C already map 1:1 onto render-graph
concepts:

- **Linear pass list → DAG is a superset relationship.** A `PassDef` array with
  `PassInput::Source | Scratch(name)` already encodes a trivial DAG (a chain). Upgrading to
  a general DAG means accepting multiple named inputs per pass and topologically sorting —
  the existing declarations are valid DAG nodes unchanged.
- **Named scratch resources already exist.** The `Scratch("h")` / `Scratch("v")` naming in
  Option C is exactly the "virtual resource" concept a render graph uses. The pool that
  allocates scratch textures keyed by `(shader_id, scratch_name, dims)` becomes the graph's
  transient-resource allocator with the same key shape.
- **Renderer dispatch loop is the right shape.** `Renderer::apply` walks passes in
  declared order and dispatches each. Swapping "declared order" for "topologically sorted
  order" is a local change to the loop driver; the per-pass dispatch code is unchanged.
- **Bind-group contract is extendable.** Widening from "1-source or 2-source" to "N named
  inputs" is additive — existing 1- and 2-source passes keep working.

**What actually changes in a C→E migration**

- `PassDef.input: PassInput` grows to `inputs: &[NamedInput]` (a slice instead of an enum
  with two variants).
- Engine gains topological sort + transient-resource aliasing (reusing a scratch texture's
  memory across non-overlapping lifetimes in the graph). This is the real work, but it is
  *new* work enabled by E — not rework undoing C.
- `PassOutput::Final` is retained; the graph just identifies it as the root consumer.

**What does not need rewriting in a C→E migration**

Every Clarity pass, every single-pass shader, every existing test, every registration
macro, the `inventory` mechanism, history/undo semantics, and the Clean Slate Replay model
all survive untouched.

### The one gotcha — and how to avoid it now

If C ships with the second source texture hardcoded at a fixed WGSL binding (e.g.,
`@group(0) @binding(2)` baked into the bind-group layout), then a future graph shader with
3+ inputs forces a bind-group-layout migration across every multi-pass shader already in
the tree. That is the only rework a naive C implementation would create.

**We avoid this upfront by binding by *declared position* rather than by hardcoded slot
number.** Concretely:

- `PassDef` declares its inputs as an ordered list of `PassInput` values (even if today's
  passes use only 1 or 2).
- The engine binds input `i` to `@group(0) @binding(i)` for every pass, using the
  pass-declared arity to build the bind-group layout. The output storage texture moves to
  a predictable slot after the inputs (e.g., `@binding(N)` where `N` is the input count) or
  to a separate bind group — either choice is forward-compatible as long as it is
  *derived*, not hardcoded.
- Pass WGSL files declare their input bindings by index matching their declared position
  in `PassDef`. Passes that need more inputs tomorrow just declare more positions; older
  passes are unaffected.

With this discipline, widening from 2 inputs to N is a zero-diff change to every existing
pass file. Without it, every multi-pass shader landed under C would need its WGSL
`@binding(...)` attributes rewritten when E lands.

**Net:** Picking C does not buy down future optionality. It buys *time* on a render graph
until a shader actually needs non-linear structure, and the accumulated code is reusable
when that day comes — provided we enforce position-indexed input bindings from day one.

### Performance comparison to leading apps

| Tool                        | Clarity implementation                              | Multi-pass? |
|-----------------------------|-----------------------------------------------------|-------------|
| Adobe Lightroom / Camera Raw| Unsharp-mask variant on a Gaussian-blurred copy     | Yes         |
| Capture One                 | Gaussian + luminance mask                           | Yes         |
| darktable ("local contrast")| Bilateral / Gaussian pyramid                        | Yes         |
| RawTherapee                 | Wavelet decomposition (many passes)                 | Yes         |

None of the leading apps implement Clarity as a single-pass operation. Option C matches the
prevailing implementation strategy, so we are not absorbing a performance disadvantage; we
are closing a quality gap that Option D would leave open.

## Implementation sketch (not a plan — just for cost calibration)

A single PR could land the infrastructure + Clarity together; Cartoon follows as a second
shader on top of the same infrastructure:

1. Extend `ShaderMeta` with `ShaderProgram { Single, MultiPass }`. Existing single-pass
   shaders migrate by wrapping their `wgsl_source` in `ShaderProgram::Single(...)` —
   mechanical, ~11 files.
2. Add `PassDef`, `PassInput`, `PassOutput` to `shaders/mod.rs`. No wire-level change to
   the existing single-pass bind group.
3. Extend the bind-group contract to optionally bind a **second** source texture at
   `@group(0) @binding(2)` when a pass declares two inputs. Single-input passes ignore the
   extra binding slot (WGSL permits unused bindings).
4. In `pipeline.rs`: grow `CachedPipeline` to `Vec<CachedPipeline>` keyed per pass; add a
   `scratch_pool: HashMap<(&'static str, u32, u32), wgpu::Texture>` field on `Renderer`;
   in `apply`, branch on `ShaderProgram` and loop through passes when `MultiPass`.
5. Add `bdip_core/src/gpu/shaders/clarity/` with `mod.rs`, `blur_h.wgsl`, `blur_v.wgsl`,
   `combine.wgsl`. Clarity's tests mirror existing roundtrip patterns.
6. Update `specs/adding_a_shader.md` with a "Multi-pass shaders" section describing the
   `PassDef` pattern.
7. Update `specs/some_shaders.md` Clarity row to note it is multi-pass (separable Gaussian
   + combine).
8. (Follow-up PR) Add `bdip_core/src/gpu/shaders/cartoon/` with `mod.rs`, `smooth_h.wgsl`,
   `smooth_v.wgsl`, `edges.wgsl`, `combine.wgsl` — exercises the 3-input combine path and
   validates the position-indexed binding discipline in production code.

The perf regression budget for Clarity on 24 MP is: ~2 additional compute passes at
~0.3–0.5 ms each = ~1 ms total. Cartoon adds ~4–5 compute passes at the same cost per pass
(~2 ms). Both are well inside the ~15 ms readback-dominated frame budget.

## Addendum — Transforms by option

This is a reference list of common image-processing transforms categorized by which
option is the minimum needed to implement them correctly and competitively. It is not a
roadmap — it exists to help decide which transforms are unlocked by paying each
architectural cost. Entries marked *(roadmap)* are already on this project's spec lists
(`specs/some_shaders.md`, `specs/film_grain_plan.md`, `specs/transformations_reference.md`);
the rest are common features in darkroom-class apps that we may or may not pursue.

### Single-pass (already supported today)

Per-pixel operations that depend only on the input pixel and the uniform parameters. No
multi-pass infrastructure needed.

- Exposure, Brightness, Contrast, Saturation, Temperature, Tint, Invert, Grayscale
  *(roadmap — several already shipped)*
- Shadows, Highlights (the "smart mask" formula is still per-pixel — luminance is computed
  from the same pixel) *(roadmap)*
- Vignette (distance from uv center) *(shipped)*
- FilmGrainWhite (PCG3d hash), FilmGrainPerlin, FilmGrainFBM (in-shader octave loop)
  *(roadmap)*
- Per-channel curves and LUTs (1D or 3D LUTs sampled per pixel)
- Tone mapping operators with purely local math (Reinhard, ACES, Filmic)
- Kelvin-accurate white balance (matrix multiply)
- Chromatic aberration correction (per-channel sub-pixel shift)
- Gamma / transfer-function conversions

### Needs Option C (linear multi-pass, fixed pass count, same-resolution intermediates)

Transforms whose output at a pixel depends on a *neighborhood* (blur-like) but whose data
flow is a fixed, shallow chain. Each pass runs at the full image resolution.

- **Clarity** — separable Gaussian blur (H + V) + combine with original *(roadmap)*
- **Gaussian blur** — H + V separable pair. Standalone or as a building block.
- **Box blur** — same shape as Gaussian (H + V), cheaper kernel.
- **Unsharp mask / general sharpening** — blur + high-pass extraction + add back
- **Surface blur** — edge-preserving single pass, but a quality version uses blur-then-combine
- **Bilateral filter** (single-scale, approximated) — blur with spatial+range weights, then
  combine; a quality implementation benefits from separable approximation
- **Glow / bloom (fixed-radius variant)** — threshold bright pixels, blur, add back. The
  "fixed-radius" qualifier matters — true bloom uses a pyramid (see Option E).
- **Dehaze (simple variant)** — estimate airlight from a blurred minimum, subtract
- **FilmGrainBlue** — uses a pre-baked blue-noise texture; technically single-pass but
  requires auxiliary texture binding (an orthogonal axis, tracked in `film_grain_plan.md`)
- **Local contrast enhancement (single-scale)** — Clarity's cousin at larger blur radius
- **Median filter** — often implementable as a single large pass, but larger radii are
  commonly decomposed
- **Simple HDR local tonemap (Durand-Dorsey style)** — log → bilateral blur → subtract
  detail → compress base → recombine; 3–4 passes, all at full resolution
- **Noise reduction (non-wavelet)** — blur luminance, preserve chroma, combine
- **Directional / motion blur** — separable along an arbitrary axis
- **Cartoon / toon filter** *(roadmap — pending, alongside Clarity)* — edge detection
  (Sobel single-pass, or Difference-of-Gaussians using two separable Gaussian pairs) +
  color posterization/quantization + overlay. Simple variants ship in 3–5 passes; the
  higher-quality XDoG (Kyprianidis & Winkenbach) variant used in research-grade
  stylization is still a linear chain of ~6–8 passes, all at full resolution. See note
  below on input arity.

All of these fit a pass list of 2–6 `PassDef` entries where each pass consumes `Source`
and/or prior `Scratch` outputs at the same dimensions.

**A note on input arity for C.** Most of the Option C transforms above need at most 2
inputs per pass (Source + one blurred scratch, for the canonical "combine" pass). A few —
Cartoon being the clearest example — naturally want 3 inputs on their final combine
(`Source`, a blurred version for color regions, and an edge-detection result). This is
still firmly inside Option C as long as the bind-group contract is derived from the
per-pass declared arity (see "The one gotcha" above). It is *not* a signal to escalate to
Option E. A pass-local cap of 4–8 inputs covers every transform in this bucket
comfortably.

### Needs Option E (render graph — variable-resolution intermediates, pyramids, or dynamic pass counts)

Transforms whose data flow requires **resizing intermediates**, a **pass count that depends
on image size**, or fan-out patterns that cannot be linearized without large fixed
assumptions. These are the cases a flat `PassDef` list cannot express cleanly.

- **Bloom (quality variant)** — downsample the image log₂(size) times, blur each level,
  upsample and accumulate. Pass count scales with resolution; each pass operates at a
  different size. This is the canonical render-graph use case.
- **Gaussian / Laplacian pyramid methods** — build a pyramid, edit each band, collapse
  back. Variable-size intermediates at every level.
- **Local Laplacian filtering** (Paris et al.) — edge-aware detail enhancement built on
  Laplacian pyramids; quality local contrast that sits above Clarity.
- **Wavelet decomposition / reconstruction** — noise reduction and sharpening in the
  wavelet domain (RawTherapee-style). Multi-level, multi-band, size-varying.
- **Exposure fusion** — multiple exposures processed into weight maps, blended in a
  pyramid. DAG with independently computed weight and Laplacian pyramids joined at the
  end.
- **Guided filter (multi-scale)** — edge-preserving smoother that generalizes the
  bilateral filter; fast versions use downsampled guidance images.
- **Depth-of-field / lens blur with near/far layers** — foreground and background
  segregated, processed at different blur radii, composited. Multiple parallel branches
  converging.
- **Frequency separation (retouching-style)** — low- and high-frequency layers edited
  independently, then recombined. Technically expressible in C with care, but natural in E
  because each branch may have further internal structure.
- **Sky replacement / masked compositing with generated masks** — a mask-generation
  subgraph feeding a separate compositing subgraph.
- **Panorama / HDR merge from multiple source images** — fundamentally multi-input at the
  top level, not just multi-pass.
- **Tone mapping operators with detail preservation** (Mantiuk, Reinhard-local) — built on
  multi-scale decompositions.

### Borderline cases (expressible in C but cleaner in E)

These can be crammed into a linear `PassDef` list but start to feel awkward. If several of
them land in the same release, that is a signal that E has become worth building.

- **Dual-tone color grading with shared luminance mask** — luminance extracted once, used
  by shadows-tint and highlights-tint passes, then combined. 4 passes, linear, fits C; E
  would express the shared extraction more naturally.
- **Fixed-depth bloom (e.g., 3 mip levels)** — a hardcoded 3-level pyramid can be spelled
  as 6–9 passes in C with per-pass dims in the `PassDef`, but the variable-dim support and
  the pass count make it want to be E.
- **Chromatic aberration with blur** — per-channel shift + per-channel blur, three parallel
  chains combined; expressible as a straight-line pass sequence if the chains do not
  interleave.

### How this informs transform selection

- If the transforms you want are all in the first two buckets, **Option C covers them for
  the foreseeable future.** The in-place migration path to E (laid out above) preserves
  that work.
- If the transforms you want include **two or more from the Option E bucket**, start
  budgeting for a render graph. Build Clarity under C first (it's a sunk cost either way —
  C-era Clarity ports to E unchanged), then build E when the first truly non-linear
  transform actually needs it.
- The **borderline list is the warning track.** If you find yourself adding a third
  borderline shader and the `PassDef` arrays are growing unreadable, that's the signal to
  migrate to E — not before.

## Open questions (not blocking a decision)

- **Scratch texture pool eviction.** The simplest policy — drop all scratch when the image
  size changes — is fine for V1. If users flip between multiple images, a small LRU keyed
  by `(shader_id, pass, dims)` would avoid re-allocation. Not worth building until profiling
  shows it matters.
- **Blur kernel parameterization for Clarity.** The spec exposes only `u_Clarity` (amount).
  A fixed internal blur radius (e.g., 1.5% of image width) is the typical Lightroom default;
  we can ship that and decide later whether to expose it. Not an architecture question.
- **Two-source binding generalization.** The proposed contract allows 1-source or 2-source
  passes. If a future shader needs 3+ source textures, we extend the enum. Keeping the
  contract narrow (named inputs, fixed shapes) preserves readability; going fully generic
  is the render-graph rabbit hole.
