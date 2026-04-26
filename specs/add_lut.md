# Auxiliary Textures for Shaders — Plan

## Problem

Many filters in the `fun_shaders.txt` reference list require external textures: LUTs
(Look-Up Tables) for color grading, noise maps for grain/grit, paper/border textures for
artistic overlays, and gradient maps for false-color effects. The `[TX]` tag marks 37 of
the 100 filters. Additionally, `FilmGrainBlue` (deferred in `specs/film_grain_plan.md`)
is blocked specifically because the pipeline cannot bind auxiliary textures.

The current shader pipeline (`bdip_core/src/gpu/image_pipeline.rs`) hardcodes two bind
groups per pass:

- **Group 0** — texture inputs (source/scratch) + one write-only storage output.
- **Group 1** — one uniform buffer for shader parameters.

There is no mechanism for a shader to declare that it needs an additional read-only
texture (a LUT, noise map, etc.). `ShaderRegistration` carries only `meta` and
`make_uniform`; `PassDef` declares only `inputs` (source/scratch), `output`, and
`output_scale`.

## Goals

1. Allow shaders to declare **named auxiliary textures** at registration time.
2. Bundle auxiliary texture source data in the binary via `include_bytes!`, but
   **lazily decode and upload to the GPU** only when a shader that needs the
   texture is first used.
3. Bind auxiliaries into the GPU pipeline alongside existing source/scratch textures.
4. Preserve the existing shader contract — shaders that don't use auxiliaries must
   require zero changes.
5. Keep the architecture general enough for LUTs (1D/2D/3D), noise maps, paper textures,
   and gradient maps without special-casing any single use case.

## Non-goals

- Dynamically user-supplied textures (e.g., "load your own LUT file"). That is a UI/CLI
  feature layered on top of this work; the engine just needs "bind this named texture."
- Sampler configuration per auxiliary (address mode). Start with clamp-to-edge;
  extend later if a shader needs repeat wrapping (the shader can use `fract()` as
  a workaround in the meantime). Filter mode *is* in scope — see `AuxTextureDef`
  below.

## Design

### Auxiliary texture declaration

Extend `PassDef` with an optional list of auxiliary texture requirements:

```rust
/// Describes an auxiliary texture a pass needs bound at render time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuxTextureDef {
    /// Name used to look up the texture in the asset registry and to
    /// reference it in WGSL bindings.
    pub name: &'static str,
    /// Texture dimensionality — controls bind group layout entry type.
    pub dimension: AuxTextureDimension,
    /// Filtering mode for the sampler bound alongside this texture.
    pub filter: AuxSamplerFilter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxTextureDimension {
    D2,   // 2D noise maps, paper textures, gradient maps
    D3,   // 3D color LUTs (e.g., 64×64×64 cube)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxSamplerFilter {
    /// Bilinear interpolation — smooth blending between texels.
    /// Use for LUTs, gradient maps, noise maps.
    Linear,
    /// No interpolation — returns the nearest texel exactly.
    /// Use for halftone patterns, character maps, pixel-art lookups.
    Nearest,
}
```

Add a field to `PassDef`:

```rust
pub struct PassDef {
    pub label: &'static str,
    pub wgsl_source: &'static str,
    pub inputs: &'static [PassInput],
    pub output: PassOutput,
    pub output_scale: PassScale,
    pub aux_textures: &'static [AuxTextureDef],  // NEW — empty slice by default
}
```

Existing shaders set `aux_textures: &[]` and require no other changes.

### Bind group layout

Auxiliary textures live in a **dedicated Group 2**, separate from the existing
per-dispatch textures (Group 0) and uniform params (Group 1). This maps each
group to its GPU update frequency:

```
Group 0 bindings (unchanged — per-dispatch, rebuilt every pass):
  0..N-1      — pass inputs (source / scratch)    [texture_2d<f32>]
  N           — output                            [texture_storage_2d, write]

Group 1 bindings (unchanged — per-transform):
  0           — uniform buffer (params)

Group 2 bindings (NEW — static, built once per aux texture set):
  0           — aux texture 0                     [texture_2d<f32> or texture_3d<f32>]
  1           — sampler for aux texture 0         [sampler]
  2           — aux texture 1 (if present)
  3           — sampler for aux texture 1
  ...etc
```

**Why a separate group instead of appending to Group 0:**

- **Stable WGSL bindings.** Auxiliary bindings are always `@group(2) @binding(0)`,
  `@group(2) @binding(1)`, etc., regardless of how many inputs the pass has.
  No offset arithmetic needed when writing WGSL.
- **Bind group caching.** Group 0 is rebuilt every dispatch because source/scratch
  textures change per-pass. Auxiliary textures are static (uploaded once, never
  change). The Group 2 bind group can be created once when the aux texture is
  first loaded and reused on every subsequent dispatch — zero per-frame cost.
- **No changes to existing shaders.** `build_pass_bind_group_layout()` and
  Group 0 are untouched. Shaders without auxiliaries keep a 2-group pipeline
  layout (`&[group0, group1]`); shaders with auxiliaries get a 3-group layout
  (`&[group0, group1, group2]`). Each shader already has its own compiled
  pipeline in `ShaderPassesCache`, so different layout shapes are not a problem.

Each auxiliary's sampler is created from its `AuxSamplerFilter` setting —
`Linear` produces `FilterMode::Linear`, `Nearest` produces
`FilterMode::Nearest`. Both use `AddressMode::ClampToEdge` (shaders that need
tiling can use `fract()` in WGSL). Samplers are cheap GPU objects and can be
cached by filter mode on the renderer.

A new function builds the Group 2 layout from a pass's `aux_textures` slice:

```rust
fn build_aux_bind_group_layout(
    device: &wgpu::Device,
    aux_textures: &[AuxTextureDef],
    label: &str,
) -> wgpu::BindGroupLayout
```

`build_pass_bind_group_layout()` (Group 0) is **unchanged**.

### Asset registry (two-tier: static manifest + lazy GPU cache)

The registry has two layers, mirroring how `ShaderPassesCache` lazily compiles
pipelines:

**Layer 1 — Static asset manifest (build time).** Each bundled asset is registered
via `inventory` at link time, mapping a name to its raw `&'static [u8]` bytes
(from `include_bytes!`) and decode metadata. No CPU decode or GPU upload happens
here — the bytes are part of the binary's read-only data segment and are only
paged into physical RAM by the OS when accessed.

```rust
/// A bundled auxiliary texture asset. Collected by `inventory` at link time.
pub struct AuxAssetRegistration {
    /// Lookup key — matches the `name` field in `AuxTextureDef`.
    pub name: &'static str,
    /// Raw file bytes embedded via `include_bytes!`.
    pub raw_bytes: &'static [u8],
    /// How to decode `raw_bytes` into pixel data.
    pub format: AuxAssetFormat,
    /// GPU texture dimensionality (2D or 3D).
    pub dimension: AuxTextureDimension,
}

pub enum AuxAssetFormat {
    /// PNG image — decoded via the `image` crate.
    Png,
    /// Pre-baked raw f32 RGB triples for a 3D LUT (width³ × 3 floats).
    CubeRaw { size: u32 },
}

inventory::collect!(AuxAssetRegistration);
```

Each asset module (e.g., `bdip_core/src/gpu/assets/blue_noise.rs`) contains
one `inventory::submit!` call and one `include_bytes!` — nothing else runs
at startup.

**Layer 2 — Lazy GPU texture cache (runtime).** The `AuxTextureCache` lives on
the `Renderer` and is initially empty. When `encode_passes_into()` encounters
a pass that declares an auxiliary texture, it calls `cache.get_or_upload()`.
On first access for a given name, this:

1. Looks up the `AuxAssetRegistration` in the `inventory` collection.
2. Decodes the raw bytes (PNG decode, or reinterpret raw floats).
3. Uploads the decoded pixels to the GPU via `queue.write_texture()`.
4. Caches the resulting `wgpu::Texture` by name.

Subsequent lookups for the same name return the cached GPU texture immediately.
If two shaders reference the same auxiliary name (e.g., both use
`"blue_noise_128"`), only one GPU texture is created.

```rust
pub struct AuxTextureCache {
    gpu_textures: HashMap<&'static str, wgpu::Texture>,
}

impl AuxTextureCache {
    pub fn new() -> Self {
        Self { gpu_textures: HashMap::new() }
    }

    /// Returns the GPU texture for `name`, decoding and uploading on first
    /// access. Returns `Err` if no asset is registered under `name`.
    pub fn get_or_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
    ) -> Result<&wgpu::Texture, BdipError> {
        if self.gpu_textures.contains_key(name) {
            return Ok(&self.gpu_textures[name]);
        }
        let asset = find_asset_by_name(name)
            .ok_or_else(|| BdipError::MissingAuxTexture(name.to_string()))?;
        let texture = decode_and_upload(device, queue, asset);
        Ok(self.gpu_textures.entry(name).or_insert(texture))
    }
}

/// Looks up a registered asset by name from the `inventory` collection.
fn find_asset_by_name(name: &str) -> Option<&'static AuxAssetRegistration> {
    inventory::iter::<AuxAssetRegistration>().find(|a| a.name == name)
}
```

This design means:

- **Startup cost is zero.** No textures are decoded or uploaded until needed.
- **First use of a [TX] shader pays the one-time decode + upload cost** for its
  auxiliary textures (~1–5 ms for a 128×128 PNG, ~10–20 ms for a 64³ LUT).
- **Subsequent uses are a HashMap lookup** (~nanoseconds).
- **Binary size grows** with each `include_bytes!` asset, but the bytes are
  memory-mapped and only paged into RAM when accessed. This is acceptable for
  the first dozen assets; if the count grows large, a sidecar asset directory
  can replace `include_bytes!` without changing any other layer.

### Renderer integration

`Renderer` gains an `AuxTextureCache` field (initially empty). The cache stores
both the GPU texture and its pre-built Group 2 bind group, keyed by a
composite key derived from the pass's `aux_textures` slice (the set of names +
filters). During `encode_passes_into()`:

1. Group 0 is built per-dispatch as today (source/scratch/output — unchanged).
2. Group 1 is built per-transform as today (uniform params — unchanged).
3. If the pass has `aux_textures`, the renderer calls
   `cache.get_or_upload(device, queue, name)` for each auxiliary to ensure the
   GPU texture exists, then creates (or retrieves a cached) Group 2 bind group
   containing all the auxiliary texture views and their samplers. This bind
   group is set via `compute_pass.set_bind_group(2, &aux_bind_group, &[])`.
4. If the pass has no `aux_textures`, Group 2 is not set and the pipeline
   layout has only 2 groups.

If a required auxiliary is not registered in the static manifest, the renderer
returns `BdipError::MissingAuxTexture` (not a panic).

### WGSL binding convention

Shaders that use auxiliaries follow this pattern in their `.wgsl` file:

```wgsl
// Group 0 — per-dispatch textures (unchanged from existing shaders)
@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

// Group 1 — uniform params (unchanged)
@group(1) @binding(0) var<uniform> params: MyParams;

// Group 2 — auxiliary textures (NEW, only for [TX] shaders)
@group(2) @binding(0) var lut_texture: texture_3d<f32>;
@group(2) @binding(1) var lut_sampler: sampler;
```

Auxiliary bindings always start at `@group(2) @binding(0)` regardless of how
many inputs the pass has. Each auxiliary occupies two consecutive bindings:
texture, then sampler. Shaders without auxiliaries omit Group 2 entirely.

### Validation

Extend `validate_pass_list()` to check:

1. No auxiliary name collides with a scratch name.

Add a test-time check (`test_all_aux_textures_have_registered_assets`) that iterates
all registered shaders, collects every `AuxTextureDef.name` across all passes, and
verifies that `find_asset_by_name(name)` returns `Some` for each. This catches
misspellings and missing `inventory::submit!` calls at test time rather than at
runtime.

At runtime, `AuxTextureCache::get_or_upload()` returns `Err(BdipError)` if the
name is not found — this is the safety net if a test is skipped or a new shader
is added without its asset.

## Asset format decisions

### 3D LUTs

Ship as raw `f32` data (`.cube` format parsed at build time or pre-baked to binary).
Standard size: **64×64×64** (786 KB uncompressed as `Rgba16Float`). This is the
industry-standard size used by DaVinci Resolve, Photoshop, and most color grading
tools. Uploaded as `wgpu::TextureDimension::D3` with format `Rgba16Float`.

### 2D noise / texture maps

Ship as PNG images decoded via the `image` crate (already a dependency). Uploaded as
`wgpu::TextureDimension::D2` with format `Rgba16Float`. The `decode_and_upload()`
function expands 8-bit PNG data to 16-bit float during the one-time upload. This
matches the main pipeline's texture format and keeps `decode_and_upload()` to a
single output format.

### Blue noise (for FilmGrainBlue)

Christoph Peters' CC0 blue noise textures (128×128, tileable). Uploaded as a single
`Rgba16Float` 2D texture (expanded from the single-channel PNG during upload).
Sampled with `uv * image_resolution / tile_size` so it tiles across the image. The
`Variation` slider offsets the UV by a pseudo-random shift derived from the seed.

## Changes by file

### New files

| File | Purpose |
|------|---------|
| `bdip_core/src/gpu/assets/mod.rs` | `AuxAssetRegistration`, `AuxAssetFormat`, `AuxTextureCache`, `find_asset_by_name()`, `decode_and_upload()` |

Asset data files (PNGs, raw LUT data) are added per-shader in later PRs, each
with its own submodule under `assets/` containing an `inventory::submit!` and
`include_bytes!`.

### Modified files

| File | Change |
|------|--------|
| `bdip_core/src/gpu/shaders/mod.rs` | Add `AuxTextureDef`, `AuxTextureDimension`, `AuxSamplerFilter`; add `aux_textures` field to `PassDef`; extend `validate_pass_list()` for aux/scratch name collision check |
| `bdip_core/src/error.rs` (or wherever `BdipError` is defined) | Add `MissingAuxTexture(String)` variant |
| `bdip_core/src/gpu/image_pipeline.rs` | Add `build_aux_bind_group_layout()`; extend `compile()` to build 3-group pipeline layouts for passes with auxiliaries; extend `encode_passes_into()` to call `cache.get_or_upload()` and set Group 2 bind group; add `AuxTextureCache` field to `Renderer`; `CompiledPass` gains an optional `aux_bind_group_layout` field |
| Every existing shader `mod.rs` | Add `aux_textures: &[]` to each `PassDef` (mechanical, no behavior change) |
| `specs/adding_a_shader.md` | Document how to declare and use auxiliary textures |

### Unmodified

| File | Why |
|------|-----|
| `bdip_core/src/gpu/engine.rs` | No changes to device/queue setup |
| `bdip_core/src/gpu/texture.rs` | Upload/download paths are unaffected |
| `bdip/src/` (UI binary) | Auxiliary textures are engine-internal; the UI sees no API change |

## PRs

**Prerequisite reading for all PRs:** Before implementing any PR below, read
`specs/adding_a_shader.md` (the existing shader-registration template) and
`AGENTS.md` (workflow constraints: clippy, formatting, test standards). For PRs
that add shaders, also read the Design section of this document (above) for the
bind group layout, WGSL binding convention, and lazy cache API. Use
`bdip_core/src/gpu/shaders/vignette/` (multi-slider, single-pass) and
`bdip_core/src/gpu/shaders/cartoon/` (multi-pass with scratch textures) as
reference implementations for code structure and test patterns.

### PR 1 — Auxiliary texture infrastructure

**Goal:** Add the `AuxTextureDef` type, extend `PassDef`, create the two-tier
asset registry (`AuxAssetRegistration` manifest + `AuxTextureCache` lazy cache),
add `build_aux_bind_group_layout()` for Group 2, extend `compile()` to produce
3-group pipeline layouts for passes with auxiliaries, and wire Group 2 binding
into `encode_passes_into()`. All existing shaders gain `aux_textures: &[]`
with no behavior change — their pipeline layouts remain 2-group.

**Key implementation details:**

- `CompiledPass` gains an `aux_bind_group_layout: Option<wgpu::BindGroupLayout>`
  field. It is `None` for passes with empty `aux_textures` and
  `Some(layout)` otherwise.
- In `compile()`, if a pass has `aux_textures`, build the Group 2 layout via
  `build_aux_bind_group_layout()` and use a 3-element
  `bind_group_layouts: &[group0, group1, group2]` in `PipelineLayoutDescriptor`.
  Otherwise, keep the existing 2-element layout.
- In `encode_passes_into()`, if the compiled pass has an `aux_bind_group_layout`,
  call `cache.get_or_upload()` for each aux name, build the Group 2 bind group
  (texture view + sampler pairs), and call
  `compute_pass.set_bind_group(2, &aux_bg, &[])`.

**Tests:**

- All existing shader tests pass unchanged (regression).
- `test_build_aux_layout_with_one_2d` — Group 2 layout has 2 entries
  (texture + sampler) with correct types.
- `test_build_aux_layout_with_one_3d` — 3D texture binding type is correct.
- `test_build_aux_layout_with_two_aux` — 4 entries (2 texture + sampler pairs).
- `test_compile_no_aux_has_two_group_layout` — a shader with `aux_textures: &[]`
  produces a `CompiledPass` where `aux_bind_group_layout` is `None`.
- `test_compile_with_aux_has_three_group_layout` — a shader with one aux
  texture produces `aux_bind_group_layout: Some(...)`.
- `test_aux_cache_get_or_upload_returns_texture` — register a test asset via
  inventory, call `get_or_upload`, verify a texture is returned.
- `test_aux_cache_second_call_returns_same_texture` — call `get_or_upload`
  twice, verify the same GPU texture is returned (no re-upload).
- `test_aux_cache_missing_name_returns_error` — lookup of unregistered name
  returns `BdipError::MissingAuxTexture`.
- `test_validate_pass_list_aux_name_collides_with_scratch` — validation catches
  collision (compile-time or test-time check).
- `test_all_aux_textures_have_registered_assets` — iterates all registered
  shaders, verifies every declared aux name has a matching
  `AuxAssetRegistration`.

### PR 2 — FilmGrainBlue (first auxiliary-texture shader)

**Goal:** Ship the blue-noise grain shader, proving the auxiliary texture
infrastructure end-to-end. Bundles a 128×128 blue noise PNG.

**Additional reading:** `specs/film_grain_plan.md` — contains the full design
for all film grain variants. The "Deferred transforms → FilmGrainBlue" section
describes the blue noise strategy, texture source (Christoph Peters CC0), and
UV tiling approach. The "Parameters (common to all shipped grain variants)"
table defines the Amount + Variation slider schema this shader must follow.
Use the PR 1 (FilmGrainWhite) section of that file as the structural template
for `mod.rs` and tests.

**Files to add:**

1. `bdip_core/src/gpu/shaders/film_grain_blue/mod.rs`
2. `bdip_core/src/gpu/shaders/film_grain_blue/film_grain_blue.wgsl`
3. `bdip_core/src/gpu/assets/blue_noise.rs` — `inventory::submit!` +
   `include_bytes!("blue_noise_128.png")`
4. `bdip_core/src/gpu/assets/blue_noise_128.png` — 128×128 tileable blue noise
   (Christoph Peters CC0, single-channel)

**Files to modify:**

1. `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod film_grain_blue;`
2. `bdip_core/src/gpu/assets/mod.rs` — add `pub mod blue_noise;`

**Auxiliary texture declaration:**

```rust
const PASSES: &'static [PassDef] = &[PassDef {
    label: "film_grain_blue",
    wgsl_source: include_str!("film_grain_blue.wgsl"),
    inputs: &[PassInput::Source],
    output: PassOutput::Final,
    output_scale: PassScale::Full,
    aux_textures: &[AuxTextureDef {
        name: "blue_noise_128",
        dimension: AuxTextureDimension::D2,
        filter: AuxSamplerFilter::Linear,
    }],
}];
```

**Parameters:**

| Name      | Range      | Default | Purpose                                       |
|-----------|------------|---------|-----------------------------------------------|
| Amount    | [0.0, 0.1] | 0.0    | Grain intensity, scaled by `sqrt(L)` in shader |
| Variation | [0.0, 1.0] | 0.0    | UV offset to reshuffle the tiled pattern      |

**WGSL approach:** The shader samples the blue noise texture at
`fract(pixel_coord / 128.0 + variation_offset)` to tile it across the image.
The `Variation` slider produces a 2D offset derived from the seed (e.g.,
`vec2(fract(variation * 12.9898), fract(variation * 78.233))`). The sampled
noise value is centered to `[-0.5, 0.5]` and applied identically to the
FilmGrainWhite formula: `C_out = C_in + noise * amount * sqrt(luma)`.

**Tests:** Mirror the FilmGrainWhite test suite from `specs/film_grain_plan.md`
(PR 1 section), adapting names from `film_grain_white` → `film_grain_blue`.
The test table in that file lists 9 tests covering registry, identity,
perturbation, variation, determinism, alpha, and black-pixel behavior. Add
these two additional tests:

- `test_film_grain_blue_requires_aux_texture` — verify the shader's `PassDef`
  declares `"blue_noise_128"` in its `aux_textures`.
- `test_film_grain_blue_missing_aux_returns_error` — applying the shader
  without the blue noise asset registered produces
  `BdipError::MissingAuxTexture`, not a panic.

**Performance test** (in `bdip_core/tests/performance.rs`):

- `perf_gpu_roundtrip_24mp_film_grain_blue` — first shader with an auxiliary
  texture. Benchmarks the Group 2 bind group setup cost and the
  `get_or_upload` cache-hit path on the warm run. Single-pass, so any
  regression isolates aux texture overhead from pass count.

### PR 3 — Comic Book (multi-pass + halftone auxiliary texture)

**Goal:** Ship a Comic Book shader that exercises both multi-pass composition and
a nearest-neighbor auxiliary texture. This proves the sampler-per-auxiliary design
and the interaction between auxiliary textures and the existing multi-pass scratch
texture system.

**How it works:** Comic Book uses Sobel edge detection for thick black outlines
and a halftone dot pattern (from an auxiliary texture) for Ben-Day dot shading.
The halftone texture is a tileable threshold map — each cell contains a radial
gradient. The shader compares pixel luminance against the threshold to produce
dots whose size varies with brightness.

**Pass structure:**

The shader reuses the same Sobel edge detection approach as `cartoon/edges.wgsl`
but with different default parameters (lower threshold, higher darkness) to
produce thicker, bolder outlines typical of comic book inking.

| Pass | Inputs | Output | Description |
|------|--------|--------|-------------|
| `edges` | Source | Scratch("edges") | Sobel edge detection (bold outlines) |
| `halftone` | Source + aux `"halftone_dots"` | Scratch("halftone") | Threshold source luminance against tiled halftone pattern |
| `combine` | Source + Scratch("halftone") + Scratch("edges") | Final | Composite: colorize halftone, overlay bold edges |

**Files to add:**

1. `bdip_core/src/gpu/shaders/comic_book/mod.rs`
2. `bdip_core/src/gpu/shaders/comic_book/edges.wgsl`
3. `bdip_core/src/gpu/shaders/comic_book/halftone.wgsl`
4. `bdip_core/src/gpu/shaders/comic_book/combine.wgsl`
5. `bdip_core/src/gpu/assets/halftone_dots.png` — tileable threshold map
   (e.g., 128×128, each 16×16 cell contains a radial gradient)

**Auxiliary texture declaration:**

```rust
const PASSES: &'static [PassDef] = &[
    PassDef {
        label: "edges",
        // ...
        aux_textures: &[],
    },
    PassDef {
        label: "halftone",
        // ...
        aux_textures: &[AuxTextureDef {
            name: "halftone_dots",
            dimension: AuxTextureDimension::D2,
            filter: AuxSamplerFilter::Nearest,  // crisp dot boundaries
        }],
    },
    PassDef {
        label: "combine",
        // ...
        aux_textures: &[],
    },
];
```

**Parameters:**

| Name | Range | Default | Purpose |
|------|-------|---------|---------|
| Strength | [0.0, 1.0] | 0.0 | Blend between original and comic-book output |
| Dot Scale | [4.0, 64.0] | 16.0 | Halftone cell size in pixels (larger = bigger dots) |
| Edge Threshold | [0.0, 1.0] | 0.10 | Sobel magnitude below which no edge is drawn |
| Edge Thickness | [0.01, 0.5] | 0.15 | Softness/width of the edge ramp |

**Halftone pass WGSL sketch:**

```wgsl
@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ComicBookParams;
@group(2) @binding(0) var halftone_tex: texture_2d<f32>;
@group(2) @binding(1) var halftone_sampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<u32>(gid.xy);
    let color = textureLoad(src_texture, coord, 0);
    let luma = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Tile the halftone texture across the image.
    let ht_dims = vec2<f32>(textureDimensions(halftone_tex));
    let uv = fract(vec2<f32>(gid.xy) / params.dot_scale);
    let ht_coord = vec2<i32>(uv * ht_dims);
    let threshold = textureLoad(halftone_tex, ht_coord, 0).r;

    // Ben-Day dots: pixel is "inked" if luminance < threshold.
    let dot = select(1.0, 0.0, luma < threshold);
    textureStore(dst_texture, coord, vec4<f32>(dot, dot, dot, color.a));
}
```

**Why this shader is valuable as an early PR:**

- Exercises both `PassInput::Scratch` and `AuxTextureDef` in the same shader,
  proving they compose correctly.
- The halftone texture requires `AuxSamplerFilter::Nearest` — validates that
  per-auxiliary sampler configuration works (vs. the `Linear` sampler used by
  FilmGrainBlue and LUTs).
- Multi-pass structure mirrors cartoon (edges + combine), confirming auxiliary
  textures work alongside the existing scratch pool.
- Visually distinctive — easy to confirm correctness by eye.

**Tests:**

- `test_comic_book_registry_entry_exists`
- `test_comic_book_registry_metadata` — 4 sliders, 3 passes
- `test_comic_book_zero_strength_is_identity` — strength=0 returns original
- `test_comic_book_full_strength_reduces_unique_values` — halftone quantizes
  the image, reducing distinct pixel values
- `test_comic_book_edges_darken_at_boundaries` — a hard step edge produces
  darkened pixels (same pattern as cartoon's edge test)
- `test_comic_book_halftone_pass_uses_aux_texture` — verify the halftone pass
  declares `"halftone_dots"` in its `aux_textures`
- `test_comic_book_dot_scale_changes_pattern` — two runs with different
  `dot_scale` values produce different output
- `test_comic_book_alpha_preserved`
- `test_comic_book_deterministic`

**Performance test** (in `bdip_core/tests/performance.rs`):

- `perf_gpu_roundtrip_24mp_comic_book` — multi-pass shader with an auxiliary
  texture. Benchmarks the interaction between aux textures and the scratch
  texture pool. The 3-pass structure (edges, halftone with aux, combine) is
  the most complex aux pipeline in the plan.

### PR 4 — 3D LUT color grading shader

**Goal:** Ship a generic "Color LUT" shader that applies a 3D LUT to the image.
This unlocks the `[TX]` filters that use color grading (Kodachrome, 1970s Fade,
Cyberpunk, etc.) — each becomes a matter of bundling a different `.cube` LUT
file and registering it as an `AuxAssetRegistration`.

**Files to add:**

1. `bdip_core/src/gpu/shaders/color_lut/mod.rs`
2. `bdip_core/src/gpu/shaders/color_lut/color_lut.wgsl`
3. `bdip_core/src/gpu/assets/luts/mod.rs` — sub-module for LUT assets
4. `bdip_core/src/gpu/assets/luts/identity.rs` — identity LUT asset
   registration (used for testing; also serves as the default LUT)
5. `bdip_core/src/gpu/assets/luts/identity_64.bin` — pre-baked 64×64×64
   identity LUT (raw `f32` RGB triples, 3 MB). Generate at build time or
   check in a script that produces it: for each `(r, g, b)` in `[0..64]³`,
   emit `(r/63.0, g/63.0, b/63.0)` as three `f32` values.

**Files to modify:**

1. `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod color_lut;`
2. `bdip_core/src/gpu/assets/mod.rs` — add `pub mod luts;`

**Auxiliary texture declaration:**

```rust
const PASSES: &'static [PassDef] = &[PassDef {
    label: "color_lut",
    wgsl_source: include_str!("color_lut.wgsl"),
    inputs: &[PassInput::Source],
    output: PassOutput::Final,
    output_scale: PassScale::Full,
    aux_textures: &[AuxTextureDef {
        name: "identity_lut_64",
        dimension: AuxTextureDimension::D3,
        filter: AuxSamplerFilter::Linear,
    }],
}];
```

Note: the `name` field here is the *default* LUT. When the engine supports
multiple named LUTs (Kodachrome, Cyberpunk, etc.), each gets its own
`AuxAssetRegistration` and the shader's aux name is set per-transform at
runtime. This is a future UI/CLI concern (see open question 2); for this PR,
ship with the identity LUT to prove the 3D texture pipeline works end-to-end.

**Parameters:**

| Name      | Range      | Default | Purpose                               |
|-----------|------------|---------|---------------------------------------|
| Intensity | [0.0, 1.0] | 1.0    | Blend between original and LUT output |

**LUT data format:** The `.bin` file contains `64 * 64 * 64 * 3` packed `f32`
values (no header, no alpha). The `AuxAssetFormat::CubeRaw { size: 64 }` decode
path reinterprets the bytes as `&[f32]` via `bytemuck::cast_slice`, then uploads
as a `wgpu::TextureDimension::D3` texture with format `Rgba16Float` (expanding
RGB→RGBA with alpha=1.0 during upload). The iteration order is R-fastest:
`for b in 0..64 { for g in 0..64 { for r in 0..64 { ... } } }`.

**WGSL sketch:**

```wgsl
struct ColorLutParams {
    intensity: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ColorLutParams;
@group(2) @binding(0) var lut_texture: texture_3d<f32>;
@group(2) @binding(1) var lut_sampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<u32>(global_id.xy);
    let color = textureLoad(src_texture, coord, 0);

    // LUT is authored in sRGB space — convert linear → sRGB, sample, convert back.
    let srgb = pow(clamp(color.rgb, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));

    // Scale and offset to sample from cell centers (half-texel inset).
    let lut_size = f32(textureDimensions(lut_texture).x);
    let scale = (lut_size - 1.0) / lut_size;
    let offset = 0.5 / lut_size;
    let lut_coord = srgb * scale + offset;

    let graded_srgb = textureSampleLevel(
        lut_texture, lut_sampler, lut_coord, 0.0
    ).rgb;
    let graded_linear = pow(graded_srgb, vec3(2.2));

    let out = mix(color.rgb, graded_linear, params.intensity);
    textureStore(dst_texture, coord, vec4(out, color.a));
}
```

**Tests:**

- `test_color_lut_registry_entry_exists`
- `test_color_lut_registry_metadata` — 1 slider, 1 pass, 1 aux texture
- `test_color_lut_identity_lut_is_passthrough` — the identity LUT produces
  output within ±64 u16 of input (tolerance for sRGB↔linear round-trip and
  trilinear interpolation).
- `test_color_lut_intensity_zero_is_identity` — blending at 0.0 returns the
  original image exactly (the LUT result is discarded by `mix(..., 0.0)`).
- `test_color_lut_alpha_preserved` — alpha channel is untouched.
- `test_color_lut_deterministic` — two identical runs produce identical output.
- `test_color_lut_aux_texture_declared` — verify the `PassDef` declares
  an aux texture with `dimension: D3` and `filter: Linear`.

### PR 5 — Thermal Heat Map + Parchment + cross-shader chain test

**Goal:** Ship two more `[TX]`-tagged shaders from `fun_shaders.txt` that
exercise different auxiliary texture patterns (gradient map vs. tileable
overlay), plus a cross-shader chain test proving auxiliaries compose across
stacked transforms.

#### 5a — Thermal Heat Map (#92 in `fun_shaders.txt`)

Remaps pixel luminance to a false-color gradient (black → blue → red → yellow
→ white). The gradient is stored as a thin 2D texture (256×1) and sampled by
luminance. Single-pass, single auxiliary.

**Files to add:**

1. `bdip_core/src/gpu/shaders/thermal/mod.rs`
2. `bdip_core/src/gpu/shaders/thermal/thermal.wgsl`
3. `bdip_core/src/gpu/assets/thermal_gradient.rs` — `inventory::submit!` +
   `include_bytes!("thermal_gradient_256x1.png")`
4. `bdip_core/src/gpu/assets/thermal_gradient_256x1.png` — 256×1 PNG, the
   thermal color ramp (black→blue→red→yellow→white from left to right)

**Files to modify:**

1. `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod thermal;`
2. `bdip_core/src/gpu/assets/mod.rs` — add `pub mod thermal_gradient;`

**Auxiliary texture declaration:**

```rust
aux_textures: &[AuxTextureDef {
    name: "thermal_gradient",
    dimension: AuxTextureDimension::D2,
    filter: AuxSamplerFilter::Linear,  // smooth gradient interpolation
}],
```

**Parameters:**

| Name      | Range      | Default | Purpose                               |
|-----------|------------|---------|---------------------------------------|
| Intensity | [0.0, 1.0] | 0.0    | Blend between original and thermal    |

**WGSL sketch:**

```wgsl
@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ThermalParams;
@group(2) @binding(0) var gradient_tex: texture_2d<f32>;
@group(2) @binding(1) var gradient_sampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<u32>(gid.xy);
    let color = textureLoad(src_texture, coord, 0);
    let luma = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Sample the gradient texture at (luma, 0.5) — horizontal axis is brightness.
    let thermal = textureSampleLevel(
        gradient_tex, gradient_sampler, vec2<f32>(clamp(luma, 0.0, 1.0), 0.5), 0.0
    ).rgb;

    let out = mix(color.rgb, thermal, params.intensity);
    textureStore(dst_texture, coord, vec4<f32>(out, color.a));
}
```

**Tests:**

- `test_thermal_registry_entry_exists`
- `test_thermal_registry_metadata` — 1 slider, 1 pass, 1 aux texture
- `test_thermal_intensity_zero_is_identity` — original image unchanged
- `test_thermal_full_intensity_remaps_luminance` — a white pixel and a black
  pixel produce visibly different colors (not just grayscale)
- `test_thermal_alpha_preserved`
- `test_thermal_deterministic`

#### 5b — Parchment (#98 in `fun_shaders.txt`)

Multiplicative blend of a tileable paper grain texture. Single-pass, single
auxiliary. The paper texture tiles via `fract()` in the shader.

**Files to add:**

1. `bdip_core/src/gpu/shaders/parchment/mod.rs`
2. `bdip_core/src/gpu/shaders/parchment/parchment.wgsl`
3. `bdip_core/src/gpu/assets/paper_grain.rs` — `inventory::submit!` +
   `include_bytes!("paper_grain_256.png")`
4. `bdip_core/src/gpu/assets/paper_grain_256.png` — 256×256 tileable paper
   texture (warm-toned, subtle fiber grain)

**Files to modify:**

1. `bdip_core/src/gpu/shaders/mod.rs` — add `pub mod parchment;`
2. `bdip_core/src/gpu/assets/mod.rs` — add `pub mod paper_grain;`

**Auxiliary texture declaration:**

```rust
aux_textures: &[AuxTextureDef {
    name: "paper_grain_256",
    dimension: AuxTextureDimension::D2,
    filter: AuxSamplerFilter::Linear,
}],
```

**Parameters:**

| Name      | Range      | Default | Purpose                                |
|-----------|------------|---------|----------------------------------------|
| Intensity | [0.0, 1.0] | 0.0    | Blend between original and parchment   |
| Scale     | [0.5, 4.0] | 1.0    | Tile scale (larger = coarser grain)    |

**WGSL sketch:**

```wgsl
@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ParchmentParams;
@group(2) @binding(0) var paper_tex: texture_2d<f32>;
@group(2) @binding(1) var paper_sampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<u32>(gid.xy);
    let color = textureLoad(src_texture, coord, 0);

    // Tile the paper texture across the image.
    let paper_dims = vec2<f32>(textureDimensions(paper_tex));
    let uv = fract(vec2<f32>(gid.xy) / (paper_dims * params.scale));
    let paper = textureSampleLevel(
        paper_tex, paper_sampler, uv, 0.0
    ).rgb;

    // Multiplicative blend: paper darkens the image where it has grain.
    let parchment = color.rgb * paper;
    let out = mix(color.rgb, parchment, params.intensity);
    textureStore(dst_texture, coord, vec4<f32>(out, color.a));
}
```

**Tests:**

- `test_parchment_registry_entry_exists`
- `test_parchment_registry_metadata` — 2 sliders, 1 pass, 1 aux texture
- `test_parchment_intensity_zero_is_identity` — original image unchanged
- `test_parchment_full_intensity_darkens_image` — multiplicative blend with
  a sub-1.0 paper texture must produce darker output than input
- `test_parchment_scale_changes_pattern` — two runs with different `scale`
  produce different output
- `test_parchment_alpha_preserved`
- `test_parchment_deterministic`

#### 5c — Cross-shader chain test

Add to `bdip_core/src/gpu/shaders/cross_shader_tests.rs`:

```rust
#[test]
fn test_film_grain_blue_then_color_lut_composes() {
    // Verifies that two shaders with different auxiliary textures can
    // be stacked in a single transform pipeline without interfering.
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    let img = make_solid_image(16, 16, 32767, 32767, 32767);

    let out = roundtrip(
        &mut renderer, &engine, &img,
        &[
            Transform { shader_id: "film_grain_blue", values: vec![0.05, 0.5] },
            Transform { shader_id: "color_lut", values: vec![1.0] },
        ],
    );

    // With the identity LUT at full intensity, the LUT pass is a near-no-op.
    // Grain from film_grain_blue should still be visible: at least one pixel
    // should differ from the solid mid-gray input by > 128 u16.
    let any_perturbed = out.pixels().any(|p| (p[0] as i32 - 32767).unsigned_abs() > 128);
    assert!(any_perturbed, "grain should survive the LUT pass");
}
```

## Relationship to existing tech debt

The `specs/tech_debt.md` "Boilerplate-Free Shader Registration" item is orthogonal
to this work. The macro sugar, if implemented, would need to accommodate the new
`aux_textures` field but does not block or conflict with auxiliary texture support.

## Open questions

1. **Sampler per auxiliary vs. shared sampler.**
   **Resolved:** The Comic Book halftone texture requires
   `Nearest` filtering while LUTs and noise maps need `Linear`. Each auxiliary
   declares its filter mode via `AuxSamplerFilter` on `AuxTextureDef`, and
   gets its own sampler binding in Group 2. Address mode remains
   `ClampToEdge` for all (shaders use `fract()` for tiling). See the bind
   group layout and `AuxTextureDef` sections above.

2. **User-supplied LUTs.** The lazy cache supports runtime insertion — a UI
   feature to "load custom LUT" would parse a `.cube` file and call a public
   `AuxTextureCache::insert()` method to upload it under a dynamic name.
   The shader's `AuxTextureDef` would reference this name. This is a UI/CLI
   concern, not an engine concern, and can be added later without changing the
   lazy-loading infrastructure.

3. **Binary size budget.** `include_bytes!` grows the binary on disk. A 64³
   Rgba16Float LUT is ~786 KB; a 128×128 blue noise PNG is ~16 KB. The OS
   memory-maps these pages and only faults them in on access, so unused assets
   don't consume RAM. If the bundled asset count grows large (dozens of LUTs),
   consider replacing `include_bytes!` with a sidecar asset directory — the
   `AuxAssetRegistration` trait can be extended with a `File` variant alongside
   `Png`/`CubeRaw` without changing any downstream code. Not a concern for the
   first dozen assets.

4. **`texture_3d` support in WGSL compute shaders.** ~~Verify that
   `textureSampleLevel` on a `texture_3d` works in wgpu's compute shader path on
   Metal.~~ **Resolved:** `texture_3d` is supported in WebGPU compute shaders and
   wgpu's Metal backend. No special handling needed.
