# Performance Optimization Plan: Achieving Sub-20ms Interactive Latency

## Context

`bdip` targets sub-20ms end-to-end latency for image transformation preview on
Apple Silicon (`specs/perf_goal_1.md`). Current measurements from
`test_perf_gpu_roundtrip_24mp` (release mode, warm pipeline) on a 24MP image:

| Stage | Measured | Target | Status |
|-------|----------|--------|--------|
| GPU Execute (ingest + apply + present) | **0.59 ms** | 5–15 ms | Well ahead |
| Buffer Readback (download) | **57.14 ms** | 1–4 ms | ~14–57x too slow |
| **Critical Path Total** | **57.73 ms** | **8–20 ms** | **~3–7x too slow** |
| Upload (not critical path) | 73.91 ms | n/a | Slow, affects initial load |

The earlier analysis in `specs/current_perf.md` (from `temp-perf` branch) remains
directionally correct but the numbers have shifted: execute time improved
dramatically (was 45ms, now 0.59ms due to the warm pipeline), while readback
remains the dominant bottleneck.

**The entire gap to the performance goal is in the readback path.** GPU compute
is already 8–25x faster than the target.

---

## 1. Are the Performance Goals Still Feasible?

**Yes, convincingly so.** The GPU compute phase (0.59ms) proves the hardware is
more than capable. The 57ms readback overhead comes entirely from two
implementation choices that can be fixed without architectural changes:

1. **Per-call staging buffer allocation**: `download_presentation_buffer` allocates
   a 192MB `MAP_READ` buffer on every call (`texture.rs:93`). On Apple Silicon
   (UMA), allocating memory from the OS is the expensive part, not the data
   transfer.
2. **Redundant memcpy via `.to_vec()`**: After mapping, the code copies 192MB from
   the mapped region into a new `Vec<u16>` (`texture.rs:123`). This is a pure CPU
   memcpy of ~192MB.

Both are fixable. Once the staging buffer is pre-allocated and the `.to_vec()` is
eliminated, UMA readback should drop to the 1–4ms range predicted in
`perf_goal_1.md`.

---

## 2. Will This Hurt Code Readability?

**Minimal impact, with one area that needs care.**

The current code is clean because every function is stateless: allocate resources,
use them, let them drop. The performance fix requires *keeping resources alive
across calls* (buffer pooling), which inherently adds state to `Renderer` or a new
companion struct.

However:
- The public API signatures (`present()`, `download_presentation_buffer()`) do not
  need to change for external consumers. The pooling is internal to the
  implementation.
- The readability cost is one or two new fields on `Renderer` (cached buffers) and
  a size-check at the top of each method. This is a well-understood pattern.
- The `apply()` path (which creates a new `dst_texture` per call) is a separate
  concern and does **not** need buffer pooling to hit the performance target. It
  already runs in 0.59ms total for ingest+apply+present. It should be left as-is.

The one area requiring care is `download_presentation_buffer`, which is currently a
free function in `texture.rs`. To reuse a staging buffer across calls, it either
needs to accept an externally managed buffer, or the functionality moves onto
`Renderer`. The plan below proposes the latter, which consolidates GPU resource
lifecycle in one place without fragmenting the module structure.

---

## 3. Root Causes and Fixes

### Root Cause A: Staging Buffer Allocation on Every Readback
- **Location**: `bdip_core/src/gpu/texture.rs:93` — `device.create_buffer()` called
  every time `download_presentation_buffer` is invoked.
- **Cost**: ~15–30ms for a 192MB `MAP_READ` buffer allocation from the OS.
- **Fix**: Pre-allocate a staging buffer when the image dimensions are first known.
  Reuse it on every subsequent readback. Only reallocate if dimensions change.

### Root Cause B: Redundant `.to_vec()` Memcpy
- **Location**: `bdip_core/src/gpu/texture.rs:123` —
  `bytemuck::cast_slice::<u8, u16>(&data).to_vec()`
- **Cost**: ~15–25ms to allocate a 192MB `Vec<u16>` and memcpy into it.
- **Fix**: Maintain a pre-allocated `Vec<u16>` that is reused across calls.
  `copy_from_slice` from the mapped `&[u16]` into the reusable Vec (one memcpy, no
  allocation). Then `Rgba16Image::from_raw` takes ownership and the Vec is
  replenished for the next call via `std::mem::replace`, amortizing allocation cost.

### Root Cause C: Output Buffer Allocation in `present()` on Every Call
- **Location**: `bdip_core/src/gpu/pipeline.rs:422` — `device.create_buffer()` for
  `output_buffer` on every `present()` call. Also tile buffers at line 439.
- **Cost**: Part of the 0.59ms execute time (already fast), but will become the
  next bottleneck once readback is fixed.
- **Fix**: Cache the output buffer and tile buffer(s) on `Renderer`, sized to the
  current image dimensions.

### Root Cause D: CPU Pixel Conversion Loop in Upload
- **Location**: `bdip_core/src/gpu/texture.rs:30-37` — single-threaded loop
  converting 24M pixels from u16 to f16.
- **Cost**: 74ms in release mode. Not on the interactive critical path (images are
  uploaded once and cached as `cached_base_texture`), but affects initial load time.
- **Fix**: Upload raw `u16` data and let the ingest shader handle conversion on GPU.
  Already tracked in `specs/tech_debt.md` under "CPU Upload Pixel Conversion Loop".

---

## 4. Phased Delivery Plan

### Phase 1: Reuse Staging Buffer in Readback (PR 1)

**Goal**: Eliminate per-call staging buffer allocation in
`download_presentation_buffer`.

**Expected impact**: ~15–30ms reduction in readback time (from ~57ms to ~25–35ms).

**Key context for implementer**: Currently `Renderer::ingest` and
`Renderer::present` take `&self`; `Renderer::apply` takes `&mut self` (for the
`PipelineCache`). The new `download` method will also need `&mut self` since it
mutates cached buffer state. In `bdip/src/ui/app.rs`, `render_pipeline` already
obtains the renderer via `self.renderer.as_mut()?`, so the `&mut` requirement is
already satisfied at all call sites.

**Deliverables**:

1. **Add a cached staging buffer field to `Renderer`**
   (`bdip_core/src/gpu/pipeline.rs`):
   ```rust
   // In the Renderer struct, alongside pipeline_cache:
   staging_buffer: Option<(wgpu::Buffer, u64)>,  // (buffer, byte_size)
   ```
   Initialize to `None` in `Renderer::new`.

2. **Add `Renderer::download` method**
   (`bdip_core/src/gpu/pipeline.rs`):
   ```rust
   pub fn download(
       &mut self,
       engine: &GpuEngine,
       src_buffer: &wgpu::Buffer,
       width: u32,
       height: u32,
   ) -> Result<Rgba16Image, BdipError>
   ```
   Implementation logic:
   - Compute `buffer_size = width * height * 8` (4 channels × 2 bytes).
   - Check `self.staging_buffer`: if `Some((buf, sz))` and `sz >= buffer_size`,
     reuse `buf`. Otherwise, allocate a new buffer with
     `usage: MAP_READ | COPY_DST` and store it in `self.staging_buffer`.
   - Encode a `copy_buffer_to_buffer` from `src_buffer` to the staging buffer
     and submit.
   - Map the staging buffer, poll, read the data using
     `bytemuck::cast_slice::<u8, u16>(&data).to_vec()`, unmap, and construct
     `Rgba16Image::from_raw`.
   - The `.to_vec()` copy remains in this phase; Phase 3 eliminates it.

   This method imports `crate::Rgba16Image` and `crate::error::BdipError`.
   The map/poll/read logic is identical to the existing free function in
   `texture.rs` — the only difference is the buffer is reused.

3. **Keep `download_presentation_buffer` unchanged**
   (`bdip_core/src/gpu/texture.rs`): The free function stays as-is. It remains
   the correct choice for one-shot callers (headless CLI, tests) that do not
   hold a `Renderer` across calls.

4. **Update UI call sites** (`bdip/src/ui/canvas.rs`, `bdip/src/ui/app.rs`):

   In `canvas.rs`, change `presentation_to_handle` to accept `&mut Renderer`
   instead of `&GpuEngine`:
   ```rust
   pub fn presentation_to_handle(
       renderer: &mut Renderer,
       engine: &GpuEngine,
       buf: &wgpu::Buffer,
       width: u32,
       height: u32,
   ) -> Option<image::Handle> {
       let img16 = renderer.download(engine, buf, width, height).ok()?;
       let img8 = bdip_core::image::DynamicImage::ImageRgba16(img16).into_rgba8();
       let (w, h) = img8.dimensions();
       Some(image::Handle::from_rgba(w, h, img8.into_raw()))
   }
   ```

   In `app.rs`, update `render_to_handle` (line 332) to pass `renderer`:
   ```rust
   fn render_to_handle(/* ... */) -> Option<iced::widget::image::Handle> {
       let (buf, w, h) = self.render_pipeline(preview)?;
       let engine = self.engine.as_ref()?;
       let renderer = self.renderer.as_mut()?;
       canvas::presentation_to_handle(renderer, engine, &buf, w, h)
   }
   ```
   Note: `render_pipeline` already calls `self.renderer.as_mut()?` internally,
   so after it returns, the mutable borrow is released. The second
   `self.renderer.as_mut()?` in `render_to_handle` will compile because the
   returned `(buf, w, h)` tuple does not borrow `self.renderer`.

   Update `render_to_rgba16` (line 344) similarly — replace the
   `download_presentation_buffer` free function call with
   `renderer.download(engine, &buf, w, h)`.

   Update `Message::ImageLoaded` handler (line 108) — the current code uses
   `presentation_to_handle(engine, &buf, w, h)`. Change to pass `renderer`.
   Since the handler already has `&mut self.renderer` in scope via the
   `if let` destructuring, adjust to avoid conflicting borrows: call
   `renderer.present(...)` first, then call
   `canvas::presentation_to_handle(renderer, engine, &buf, w, h)`.

**Files modified**:
- `bdip_core/src/gpu/pipeline.rs`
- `bdip/src/ui/canvas.rs`
- `bdip/src/ui/app.rs`

**Dependencies**: None.

**Verification**:
- `cargo test -p bdip_core --release` — all existing tests pass (they still use
  the free function).
- `cargo test -p bdip_core --release test_perf_gpu_roundtrip_24mp -- --ignored
  --nocapture` — readback should drop from ~57ms to ~25–35ms. To see the
  improvement, add a second timed iteration in the test (after the first
  `download` call, run the apply+present+download sequence again and time it;
  the second iteration reuses the staging buffer).
- `cargo clippy --workspace`
- `cargo fmt --all`

---

### Phase 2: Reuse Present Tile Buffer (PR 2)

**Goal**: Eliminate per-call tile buffer allocation in `present()`.

**Expected impact**: Minor improvement to execute time (~0.1–0.5ms), but prevents
it from becoming the next bottleneck and completes the "warm pipeline"
architecture.

**Key context for implementer**: `present` currently takes `&self`. Caching
buffers requires `&mut self`. Since `apply` already requires `&mut self`,
changing `present` to `&mut self` has no impact on call sites — every caller
already has a mutable reference to `Renderer`.

**Deliverables**:

1. **Add a cached tile buffer field to `Renderer`**
   (`bdip_core/src/gpu/pipeline.rs`):
   ```rust
   // Alongside the staging_buffer field from Phase 1:
   present_tile_buffer: Option<(wgpu::Buffer, u64)>, // (buffer, byte_size)
   ```
   Initialize to `None` in `Renderer::new`.

   **What is and is not cached**:

   - **`tile_buffer`** (`STORAGE | COPY_SRC`, sized `max_rows * width * 8`
     bytes): **cached**. Tiles are processed sequentially, so the same buffer
     is reused across tiles within a single `present` call. Its contents are
     copied into the fresh output buffer before the next tile overwrites them.
     Safe to cache across calls for the same reason.

   - **`output_buffer`** (`COPY_DST | COPY_SRC`, sized `width * height * 8`
     bytes): **not cached**. `present` returns this buffer to the caller (who
     holds it across the subsequent `download` call). `wgpu::Buffer::clone()`
     is an `Arc` clone — if the buffer were cached on `Renderer`, a subsequent
     `present()` call would write new GPU data into the same underlying buffer
     the caller is still reading. Allocate fresh each call.

   - **`params_buffer`** (`UNIFORM`, 16 bytes, one per tile): **not cached**.
     The natural approach — cache the buffer and update it with
     `queue.write_buffer` — does not work here. `queue.write_buffer` is
     submitted to the GPU queue immediately, while the compute dispatches are
     recorded into a command encoder and only submitted at the end of the loop.
     Updating the same buffer in a loop would result in all compute passes
     seeing the last tile's params. Retain per-tile `create_buffer_init`
     (16 bytes — negligible cost).

2. **Modify `present_with_max_binding` to reuse the tile buffer**
   (`bdip_core/src/gpu/pipeline.rs`):

   Change signature from `&self` to `&mut self`. Also change `present` from
   `&self` to `&mut self`.

   Before the tile loop, check `self.present_tile_buffer`: if `Some((buf, sz))`
   and `sz >= max_tile_size`, reuse `buf`. Otherwise allocate a new buffer
   with `usage: STORAGE | COPY_SRC` and store it. `max_tile_size` is
   `max_rows * width * 8` — the size needed for the worst-case tile.

   Inside the loop body, allocate `output_buffer` and `params_buffer` as
   before (fresh per call and per tile respectively). The loop structure
   otherwise remains the same: create per-tile bind groups, encode the
   compute pass and `copy_buffer_to_buffer` into a single command encoder,
   and submit once after the loop.

3. **Update `present` and `present_with_max_binding` signatures**:
   Change both from `&self` to `&mut self`. Compiles at all existing call
   sites because:
   - `app.rs`: `renderer` is already `&mut Renderer` at both call sites
   - `main.rs` line 95: `renderer` is `mut Renderer`
   - Test bindings that were `let renderer` must become `let mut renderer`
     (five test functions updated)

**Files modified**:
- `bdip_core/src/gpu/pipeline.rs`

**Dependencies**: Phase 1 (establishes the cached-buffer-on-Renderer pattern;
avoids two PRs independently adding fields to the same struct).

**Verification**:
- `cargo test -p bdip_core --release` — all existing tests pass.
- `cargo test -p bdip_core --release test_perf_gpu_roundtrip_24mp -- --ignored
  --nocapture` — execute time should remain well under 5ms. Observed on
  Apple M4 Pro after Phases 1+2: run 2 execute ~0.28ms, run 2 critical
  path ~21.76ms (down from ~35ms after Phase 1 alone, reflecting the tile
  buffer no longer being reallocated on the warm path).
- `cargo clippy --workspace`
- `cargo fmt --all`

---

### Phase 3: Eliminate `.to_vec()` Allocation in Readback (PR 3)

**Goal**: Remove the per-call 192MB `Vec<u16>` allocation on the interactive critical path entirely, avoiding any reliance on allocator free lists or OS page cache behavior.

**Expected impact**: Readback drops from ~25–35ms (after Phase 1) to ~5–15ms.

**Key context for implementer**: The `image` crate's `Rgba16Image` requires *ownership* of a `Vec<u16>`. In Phase 1, `Renderer::download` returns an `Rgba16Image`, which means we have to give away the `Vec` allocation. 

Relying on the OS page cache / `malloc` free list to make repeated allocations fast is an anti-pattern for performance-critical inner loops. To guarantee zero allocations for the 192MB buffer, we must keep the `Vec<u16>` inside `Renderer` at all times and never give ownership to `Rgba16Image` during interactive preview.

Instead, we will add a `download_slice` method that returns a `&[u16]` borrowed directly from `Renderer`'s internal capacity, and update the UI conversion code to work with the slice.

**Deliverables**:

1. **Add a reusable pixel Vec field to `Renderer`**
   (`bdip_core/src/gpu/pipeline.rs`):
   ```rust
   // Alongside the other cached buffer fields:
   pixel_vec: Vec<u16>,
   ```
   Initialize to `Vec::new()` in `Renderer::new`.

2. **Add `download_slice` to `Renderer`**:

   Create a new method analogous to `download`, but returning a slice:
   ```rust
   pub fn download_slice(
       &mut self,
       engine: &GpuEngine,
       src_buffer: &wgpu::Buffer,
       width: u32,
       height: u32,
   ) -> Result<&[u16], BdipError>
   ```
   Inside this method, copy the mapped `u16` data into `self.pixel_vec` without giving up ownership:
   ```rust
   let data = buffer_slice.get_mapped_range();
   let u16_data = bytemuck::cast_slice::<u8, u16>(&data);
   let pixel_count = u16_data.len();

   self.pixel_vec.clear();
   self.pixel_vec.reserve(pixel_count);
   self.pixel_vec.extend_from_slice(u16_data); // Single memcpy

   drop(data);
   staging_buffer.unmap();
   
   Ok(&self.pixel_vec)
   ```

3. **Update UI to use `download_slice` and downscale to 8-bit manually**:

   In `bdip/src/ui/canvas.rs`, the `presentation_to_handle` function currently calls `renderer.download()` (which yields `Rgba16Image`) and then uses `.into_rgba8()` to create the 8-bit preview.

   Change it to use the new `download_slice` and manually map the 16-bit values to 8-bit (by shifting `>> 8`), which avoids the `Vec<u16>` allocation pipeline entirely:
   ```rust
   pub fn presentation_to_handle(
       renderer: &mut Renderer,
       engine: &GpuEngine,
       buf: &wgpu::Buffer,
       width: u32,
       height: u32,
   ) -> Option<image::Handle> {
       let pixels_16 = renderer.download_slice(engine, buf, width, height).ok()?;
       
       // Convert 16-bit to 8-bit inline. This is the same math the `image`
       // crate uses in `into_rgba8()`, but operating on a borrowed slice so
       // we avoid allocating a 192MB Vec<u16> just to hand ownership to
       // Rgba16Image.
       //
       // A `>> 8` shift would be slightly faster (truncating divide-by-256
       // vs. correct divide-by-257), but the difference is negligible
       // relative to the GPU readback memcpy, and using the correct formula
       // avoids off-by-one surprises and maintenance risk.
       // Note: this Vec<u8> (~96MB for 24MP) is allocated every frame
       // because iced's Handle::from_rgba requires ownership. We cannot
       // apply the same Renderer-owned-buffer pattern used for pixel_vec
       // above, since we have no way to reclaim the buffer after iced
       // consumes it. If this allocation shows up in profiles, investigate
       // whether iced offers a zero-copy path or whether the allocator
       // free list is reliable enough at this size.
       let mut u8_pixels = Vec::with_capacity(pixels_16.len());
       u8_pixels.extend(pixels_16.iter().map(|&p| (p as u32 * 255 / 65535) as u8));
       
       Some(image::Handle::from_rgba(width, height, u8_pixels))
   }
   ```

4. **Retain `download` for File Saving**:
   Keep the Phase 1 `download` method (which returns `Rgba16Image`) for instances where we actually *do* need the owned 16-bit image (e.g., saving to disk). It will still allocate, but file-saving is not on the interactive critical path. Update it to call `download_slice` and `.to_vec()` internally to avoid duplicated logic.

**Verification**:
- `cargo test -p bdip_core --release` — all existing tests pass.
- `cargo test -p bdip_core --release test_perf_gpu_roundtrip_24mp -- --ignored
  --nocapture` — readback should drop to ~5–15ms. Critical path total should
  be in or near the 8–20ms target.
- `cargo clippy --workspace`
- `cargo fmt --all`

---

### Phase 4: GPU-Side Upload Conversion (PR 4)

**Goal**: Eliminate the 74ms CPU pixel conversion loop in `upload_texture`.

**Expected impact**: Upload drops from ~74ms to ~5–15ms. Not on the interactive
critical path but improves initial load time significantly.

**Key context for implementer**: Currently `upload_texture`
(`bdip_core/src/gpu/texture.rs`) iterates 24M pixels on the CPU, converting each
`u16` channel to `f16` via `f16::from_f32(pixel[n] as f32 / 65535.0)`, then
writes the resulting `Vec<PixelF16>` to an `Rgba16Float` texture. The ingest
shader (`ingest.wgsl`) then reads this texture (already in `[0,1]` float range)
and applies `srgb_to_linear()`.

The fix is to upload the raw `u16` data as an `Rgba16Unorm` texture. The GPU
hardware automatically normalizes `u16` values to `[0.0, 1.0]` when the shader
reads the texture — i.e., `textureLoad` on an `Rgba16Unorm` texture returns
`f32` values equal to `channel_u16 / 65535.0`. This means the sRGB-to-linear
math in `ingest.wgsl` receives the same input values as today, with no shader
changes needed.

**Deliverables**:

1. **Simplify `upload_texture`** (`bdip_core/src/gpu/texture.rs`):

   Remove the `PixelF16` struct, the `float_data` Vec, and the per-pixel
   conversion loop. Replace with a direct `queue.write_texture` of the raw
   `u16` pixel data:
   ```rust
   pub fn upload_texture(
       device: &Device, queue: &Queue, img: &Rgba16Image
   ) -> wgpu::Texture {
       let (width, height) = img.dimensions();
       let texture_size = Extent3d { width, height, depth_or_array_layers: 1 };

       let texture = device.create_texture(&TextureDescriptor {
           label: Some("upload_texture"),
           size: texture_size,
           mip_level_count: 1,
           sample_count: 1,
           dimension: TextureDimension::D2,
           format: TextureFormat::Rgba16Unorm,  // was Rgba16Float
           usage: TextureUsages::TEXTURE_BINDING
               | TextureUsages::COPY_DST
               | TextureUsages::COPY_SRC
               | TextureUsages::STORAGE_BINDING,
           view_formats: &[],
       });

       // Rgba16Image stores pixels as contiguous [u16; 4] per pixel.
       // Write the raw bytes directly — no CPU conversion needed.
       queue.write_texture(
           TexelCopyTextureInfo { /* same as current */ },
           img.as_raw().as_byte_slice(),  // bytemuck or direct cast
           TexelCopyBufferLayout {
               offset: 0,
               bytes_per_row: Some(width * 8),  // 4 × u16 = 8 bytes
               rows_per_image: Some(height),
           },
           texture_size,
       );

       texture
   }
   ```
   Note: `img.as_raw()` returns `&Vec<u16>`. Use
   `bytemuck::cast_slice::<u16, u8>(img.as_raw())` to get `&[u8]` for
   `write_texture`.

2. **Update ingest bind group layout**
   (`bdip_core/src/gpu/pipeline.rs`, `Renderer::new`, around line 220):

   The `make_texture_only_bind_group_layout` function creates a texture
   binding with `sample_type: Float { filterable: false }`. This is
   compatible with `Rgba16Unorm` — `wgpu` allows `Float` sample type for
   Unorm textures. **No change needed to the bind group layout.**

   However, confirm by testing. If `wgpu` requires `sample_type: Float`
   specifically for `Rgba16Float` and rejects `Rgba16Unorm` at that
   binding, the ingest-specific bind group layout must be split out from
   the shared `make_texture_only_bind_group_layout` helper and use the
   correct sample type.

3. **No changes to `ingest.wgsl`**: The shader reads via `textureLoad`
   which returns `vec4<f32>` regardless of the underlying format.
   `Rgba16Unorm` auto-normalizes `u16` to `[0.0, 1.0]` on read, which is
   the same range the shader currently receives from the `Rgba16Float`
   upload. The `srgb_to_linear` math is unchanged.

4. **Update any downstream format assumptions**: Search the codebase for
   hardcoded `TextureFormat::Rgba16Float` on the *upload* texture. The
   upload texture is only used as input to `ingest`; all textures after
   `ingest` remain `Rgba16Float`. If any test creates a texture and passes
   it directly to `ingest`, that test may need updating to use
   `Rgba16Unorm`.

**Files modified**:
- `bdip_core/src/gpu/texture.rs` — rewrite `upload_texture`, remove
  `PixelF16` struct
- `bdip_core/src/gpu/pipeline.rs` — potentially adjust ingest bind group
  layout if `wgpu` validation requires it

**Dependencies**: None (independent of Phases 1–3), but best ordered after
them since it is not on the interactive critical path.

**Verification**:
- `cargo test -p bdip_core --release` — all roundtrip tests must pass with
  pixel values within existing tolerance (the conversion is mathematically
  identical).
- `cargo test -p bdip_core --release test_perf_gpu_roundtrip_24mp -- --ignored
  --nocapture` — upload should drop from ~74ms to ~5–15ms.
- `cargo clippy --workspace`
- `cargo fmt --all`

---

## 5. Dependency Graph

```
Phase 1 (staging buffer reuse)
    |
    +---> Phase 2 (present buffer reuse)
    |
    +---> Phase 3 (eliminate .to_vec() allocation)

Phase 4 (GPU upload conversion) — independent, can run in parallel
```

Phases 2 and 3 both depend on Phase 1 because they build on the
`Renderer::download` method and the pattern of cached buffers on `Renderer` that
Phase 1 establishes. Phase 4 is fully independent.

---

## 6. Expected Final State

After all four phases, projected warm-pipeline numbers for a 24MP image:

| Stage | Current | After Phase 1 | After Phase 3 | After Phase 4 |
|-------|---------|---------------|---------------|---------------|
| Execute | 0.59 ms | 0.59 ms | 0.59 ms | 0.59 ms |
| Readback | 57.14 ms | ~25–35 ms | ~2–8 ms | ~2–8 ms |
| **Critical path** | **57.73 ms** | **~26–36 ms** | **~3–9 ms** | **~3–9 ms** |
| Upload | 73.91 ms | 73.91 ms | 73.91 ms | ~5–15 ms |

The critical path target of 8–20ms should be achievable after Phase 3.
Phase 4 is an improvement for initial load time.
