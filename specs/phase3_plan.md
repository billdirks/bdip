# Phase 3: UI Prototype Spike

This implementation plan details Phase 3 of the `bdip` architecture roadmap. The fundamental goal of this phase is to evaluate and select a UI framework, validate that it can cleanly display a GPU-processed image, and confirm the integration pattern for Phase 4.

## Core Architecture Concept: The CPU Bridge

To preserve strict architectural boundaries, `bdip` utilizes a **CPU Bridge** pattern to connect the GPU-accelerated core library to the UI binary. 

Attempting to share a single GPU context (a `wgpu::Device`) between `bdip_core` and a UI framework forces both crates to resolve to the exact same major version of the `wgpu` crate. Because UI frameworks dictate their own dependency trees (e.g., `iced` 0.14 requires `wgpu 27`), this approach would force the core library to downgrade or upgrade entirely at the whim of the UI layer. This violates the specification's mandate that `bdip_core` must remain an independent, headless library capable of being consumed by any Rust project.

Therefore, the integration boundary is in host memory (CPU RAM):

```
[User action]
     │
     ▼
[bdip_core: GpuEngine::new() — headless, self-owned wgpu device]
     │
     ▼
[bdip_core: Renderer::apply_brightness() — compute shader on GPU]
     │
     ▼
[bdip_core: download_texture() — readback Rgba16Float → RgbaImage]
     │
     ▼  ← CPU boundary (RgbaImage in host memory)
     │
     ▼
[UI framework: standard image widget — re-uploads to its own GPU surface]
     │
     ▼
[Display]
```

**Performance:** On Apple Silicon (Unified Memory Architecture), the CPU and GPU share the same physical RAM. The `download_texture()` readback operation does not move data across a bus; it merely acts as a synchronisation fence. This takes approximately **1–4 ms** for typical high-resolution photos, meaning the CPU Bridge imposes no perceivable performance penalty to the user.

---

## Framework Selection

Because the CPU Bridge decouples `bdip_core`'s GPU logic from the binary's `wgpu` requirements, framework selection is based purely on developer ergonomics, styling capability, and architectural fit for complex state management (undo/redo).

**Preference order:**
1. **`iced`** — The primary target. Its declarative Elm-architecture guarantees excellent state management, while its native styling model supports the "modern, beautiful, and extremely flexible" aesthetics required by the spec.
2. **`slint`** — Excellent declarative DSL (`.slint` files) capable of highly polished, consumer-grade UIs. Requires learning a new syntax but outputs pristine visuals natively.
3. **`egui` / `eframe`** — Immediate mode. Provides extremely fast scaffolding, but its layout architecture natively leans toward technical/developer tooling. Achieving highly custom, fluid visual layouts requires significant manual widget painting.

### Step 0: Prototype Preparation
Before writing windowing code, confirm the target framework's current version and locate the exact API used to display an in-memory image buffer (e.g., `iced::widget::image::Handle::from_pixels`). 

---

## Proposed Approach & Implementation

### Minimal UI Scope
The spike UI is intentionally minimal to rapidly validate the integration. No interactive controls are in scope for this phase. The spike must implement exactly:
- A **test image path** loaded at startup. The application must accept an optional image path via a command-line argument, falling back to a hardcoded default path if none is provided. *Note: Using the CLI argument, the user should test high-resolution, large photos (e.g., 24MP+) to accurately benchmark the readback payload against smaller web-resolution files.*
- A **hardcoded brightness value** applied as a static transformation (no slider)
- A **single application window** displaying the resulting image, sized to fit the image or a fixed resolution
- No toolbars, sidebars, menus, or any other chrome

### bdip_core (No Changes Required)
**`bdip_core` must not be modified in this phase.** The library initializes itself headlessly via `GpuEngine::new()` exactly as it does today. All existing headless tests must continue to pass unchanged. The `download_texture()` function in `bdip_core/src/gpu/texture.rs` will serve as the integration bridge.

### bdip (The UI Binary)

#### [MODIFY] `bdip/Cargo.toml`
- Add the chosen framework (e.g., `iced`) as a dependency.
- Do not add a direct `wgpu` dependency to the binary crate unless explicitly required by the framework's image handling API.

#### [NEW] `bdip/src/ui_spike.rs`
- Implement a barebones application window.
- On startup: load the test image (from CLI arg or fallback macro) → call `bdip_core`'s GPU pipeline to apply brightness.
- **Profiling Step:** Wrap the `download_texture()` call with `std::time::Instant::now()`. Log the exact duration of the GPU-to-CPU readback to stdout.
- Convert the resulting `RgbaImage` to the framework's native image format → display it.

---

## Verification Plan

Phase 3 is considered complete when the `bdip` binary successfully opens a window displaying a GPU-processed image. This is a functional milestone.

### Automated Tests
- N/A for UI display. Headless pipeline correctness is already verified by Phase 2 tests.

### Manual Verification
Boot the spike binary with the hardcoded test image. Confirm:
1. **A window appears** and displays the loaded image with a visible brightness adjustment.
2. **CPU Bridge runtime is profiled and logged:** Confirm that the stdout log proves the readback takes ~1-4ms on Apple Silicon for the large test image, replacing our architectural estimations with hard data.
3. **No CPU readback panic or pipeline error** is logged.
4. **All existing Phase 1 & 2 tests still pass** (`cargo test -p bdip_core`).

### Qualitative Assessment
While the spike is running, subjectively evaluate the framework:
- Does the API feel ergonomic and maintainable for Phase 4?
- Is the development iteration loop acceptable?
- Does the rendered result look clear and native?

*If the framework proves overly cumbersome or restrictive, it will be discarded (remove from Cargo.toml, delete `ui_spike.rs`), and the spike will be repeated with the next framework on the preference list.*

Once this spike is successful, Phase 4 will begin to build the full interactive editor shell around this proven display bridge.
