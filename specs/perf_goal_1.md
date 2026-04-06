# Performance Goal 1: Beat Commercial Software on Image Transformation Latency

## Goal Statement

`bdip` targets **sub-20ms end-to-end latency** for image transformation preview on Apple Silicon —
faster than any current commercial image editing software for equivalent operations on the same
hardware.

The primary benchmark target is the time from a user interaction (e.g. slider release or drag) to
an updated image preview being visible on screen. This must be measurably faster than Lightroom
Classic, Capture One, and Affinity Photo on the same machine.

---

## Why This Is Achievable

### The Competition's Weaknesses

Commercial software carries significant architectural baggage that constrains GPU performance:

- **Lightroom Classic**: Notoriously CPU-bound for most operations. GPU acceleration is partial and
  inconsistent across adjustment types. Preview update latency on a 24MP image is commonly
  100–500ms. The processing pipeline was designed in an era before modern GPU compute was viable.
- **Capture One**: Better GPU utilisation than Lightroom Classic, known for snappier previews, but
  still typically 50–200ms for complex adjustments on large files. Also carries cross-platform
  abstraction overhead.
- **Affinity Photo**: Good GPU acceleration and the closest competitor in responsiveness. Still
  targets a wide range of hardware and OS versions, which constrains Metal-specific optimisation.

None of these applications are built around a **pure compute shader pipeline** as the primary
processing path. They layer GPU work on top of legacy CPU pipelines. `bdip` is designed GPU-first
from the ground up.

### The `bdip` Advantage

`bdip_core` processes images entirely via WGSL compute shaders dispatched through `wgpu`, which
targets **Metal** natively on macOS. This means:

- No intermediate CPU processing steps between transformations
- Transformations are chained entirely in GPU memory — the output `wgpu::Texture` of one shader is
  fed directly as input to the next without any readback between stages
- On Apple Silicon (Unified Memory Architecture), even the final display readback is near-free
  since the CPU and GPU share the same physical RAM

---

## Target Hardware: Apple Silicon (Primary)

Apple Silicon's Unified Memory Architecture (UMA) is the key enabler. There is no discrete VRAM
and no PCIe bus. The CPU and GPU share the same physical memory pool at 100–200 GB/s bandwidth.

### Expected End-to-End Pipeline Latency (Apple Silicon)

For a 24MP image (Rgba16Float, 192 MB) with a single brightness adjustment:

| Stage | Cost |
|-------|------|
| Compute shader dispatch (Metal via wgpu) | 5–15 ms |
| `download_texture()` readback fence (UMA — no real copy) | 1–4 ms |
| Display re-upload (UMA — no real copy) | ~1 ms |
| **Total** | **~8–20 ms** |

This is **5–25× faster** than Lightroom Classic's equivalent operation on the same hardware.

For smaller images that fit typical editing workflows:

| Image size | Expected total latency |
|------------|----------------------|
| 1080p / web (2MP) | < 5 ms |
| 12MP (iPhone) | ~5–10 ms |
| 24MP (DSLR) | ~8–20 ms |
| 50MP (medium format) | ~20–40 ms |

---

## What Determines Actual Performance

The display path is not the bottleneck. The variables that govern competitive performance are:

### 1. Compute Shader Quality (Highest Impact)

WGSL shader workgroup sizes, memory access patterns, and arithmetic efficiency directly determine
GPU throughput. A well-tuned compute shader on Apple Silicon will exhaust the memory bandwidth long
before the Metal command overhead becomes relevant.

Key tuning targets:
- **Workgroup size**: Currently `16×16` per dispatch. May need tuning per shader type and GPU tier
  (Apple M-series has specific SIMD group widths).
- **Memory access patterns**: Coalesced reads/writes are critical. Spatial filters (blur, sharpen)
  in particular require careful tiling to avoid cache thrashing.
- **Arithmetic precision**: `Rgba16Float` gives headroom above 1.0 without clamping, but half-
  precision arithmetic is faster than full `f32` on most GPU hardware. Shaders should use `f16`
  where precision is sufficient.

### 2. Transformation Chaining in VRAM (High Impact)

The current pipeline already achieves this correctly: the output `wgpu::Texture` of one
`apply_*` call is passed directly as the input `src_texture` of the next, with no CPU readback
between transformations. This is verified by the `test_shader_chaining` test in `pipeline.rs`.

Commercial software frequently does not achieve clean in-VRAM chaining across all adjustment types,
falling back to CPU for certain operations. This is a structural advantage of bdip's design.

### 3. Proxy Resolution During Live Preview (Medium Impact)

Applying adjustments at full resolution (24MP) during a live slider drag is unnecessary — the
display is typically 2–5MP at most. Processing a downscaled proxy during interaction and applying
the full-resolution pipeline only on slider release can reduce live-preview latency by 4–10×.

This is a Phase 4+ optimisation but should be designed for from the start. The transformation
pipeline should accept a `src_texture` at any resolution.

### 4. Display Path (Low Impact on Apple Silicon)

The `download_texture()` → `RgbaImage` → UI widget path, while conceptually a "CPU readback", is
near-free on Apple Silicon due to UMA. It does not require zero-copy GPU context sharing between
`bdip_core` and the UI framework to achieve competitive performance on the primary target hardware.

This means `bdip_core` can maintain its architectural independence (owning its own `wgpu::Device`,
pinned to whatever `wgpu` version it uses) without sacrificing the performance goal.

---

## Discrete GPU Considerations (Secondary)

On Windows and Linux machines with discrete NVIDIA/AMD GPUs, the readback crosses a PCIe bus:

| Bus | Real-world bandwidth | 24MP readback cost |
|-----|---------------------|-------------------|
| PCIe 4.0 x16 | ~20–25 GB/s | ~8–10 ms |
| PCIe 3.0 x16 | ~10–13 GB/s | ~15–20 ms |

Total latency on discrete GPU: roughly **20–40 ms** for a 24MP image. This still beats Lightroom
Classic (100–500 ms) substantially, though the margin is narrower.

The key mitigation is keeping the readback off the UI thread (see `tech_debt.md` — Synchronous
Readback Blocks UI Thread). With the compute + readback on a background thread, the window remains
fully responsive while the preview updates asynchronously. A 20–40 ms background update is
imperceptible as lag for a photo editor.

---

## Benchmark Plan

Performance claims should be validated with concrete measurements before declaring a competitive
advantage.

### Internal Benchmarks (Automated)
Use `criterion` to measure `bdip_core` pipeline throughput in isolation:
- `apply_brightness` on 2MP, 12MP, 24MP images
- N-deep transformation chains (2, 5, 10 transformations)
- Upload (`upload_texture`) and readback (`download_texture`) separately

These benchmarks run headlessly and are independent of any UI framework.

### Competitive Benchmarks (Manual)
On the same Apple Silicon machine, measure:
- `bdip`: time from calling `apply_brightness` to `RgbaImage` returned by `download_texture()`
- Lightroom Classic: time from slider release to preview update (measured with screen capture +
  frame analysis)
- Capture One: same
- Affinity Photo: same

Image sizes to test: 12MP, 24MP, 50MP.

> [!NOTE]
> This should be measured in Phase 4 or 5 once the full pipeline is wired to UI interactions.
> The Phase 2 compute pipeline is the primary driver — validating its raw throughput early is
> sufficient to project the end-to-end result.

---

## Summary

| Factor | `bdip` position |
|--------|----------------|
| Processing architecture | Pure GPU compute shaders, first-class |
| Transformation chaining | In-VRAM, zero CPU readback between stages |
| Primary target hardware | Apple Silicon (UMA eliminates display overhead) |
| Display path cost (Apple Silicon) | ~2–5 ms — not the bottleneck |
| Expected latency vs Lightroom Classic | 5–25× faster on equivalent hardware |
| Key optimisation focus | Shader quality, proxy-resolution live preview |
| Discrete GPU support | Competitive but narrower margin; background thread keeps UI responsive |
