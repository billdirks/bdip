# Phase 5: Full UI Integration

This document is the authoritative implementation plan for Phase 5. It covers the
full interactive UI, the three remaining V1 shaders (Contrast, Grayscale, Invert),
async I/O bootstrapping, error handling, and the CLI parity updates needed to
complete V1.

## 1. UI Layout Overview

The interface features a dark-themed layout (`iced::Theme::Dark`) partitioned into
three main zones:

1. **Top Menu Bar**
   - Action buttons for **Load Image** and **Save Image**.
   - Occupies minimal vertical space to maximize canvas area.
   - Both buttons are always visible. "Save Image" is disabled (grayed out) when
     no image is loaded.

2. **Central Canvas**
   - The primary viewing area displaying the active image.
   - When no image is loaded, displays a centered placeholder message:
     "No image loaded — click Load Image to begin."
   - Displays the 8-bit downsampled (`Rgba8`) representation of the core engine's
     16-bit linear buffer via `iced::widget::image`.
   - The image should be displayed with `ContentFit::Contain` to preserve aspect
     ratio and fit within the available space.

3. **Left Tools Tray** (fixed-width sidebar, ~250px)
   A fixed-width sidebar spanning the left side of the window, partitioned into two
   sections:

   **a. Transform Component (Top)**
   - A `pick_list` dropdown to select a transformation type. Initially
     populated with `Brightness` and `Saturation` (the two existing shaders).
     `Contrast`, `Grayscale`, and `Invert` are added in Phase 6 (PRs 6–8).
   - Based on the selected variant, one of two secondary widgets appears:
     - **Parameterized transforms** (Brightness, Contrast, Saturation): An
       `iced::widget::slider` with range `-1.0..=1.0` and default `0.0`. A text
       label beside or below the slider shows the current value formatted to two
       decimal places (e.g., "0.35").
     - **Parameterless transforms** (Grayscale, Invert): An "Apply" button.
   - **Slider interaction model:**
     - `on_change`: Updates `preview_value` and triggers a GPU pipeline replay
       showing the tentative result. The canvas updates immediately (live preview).
       Nothing is pushed to `HistoryManager`.
     - `on_release`: Pushes an absolute value entry to `HistoryManager` and
       re-renders from the collapsed committed stack. The slider stays at its
       released position — it does not reset to `0.0`.
     - See Section 2.1 for the full slider/history model with examples.
   - The "Apply" button for parameterless transforms is equivalent to an immediate
     commit — it pushes to `HistoryManager` and re-renders.

   **b. History Component (Bottom)**
   - A header row containing **Undo** and **Redo** text buttons with keyboard
     shortcut hints: "Undo (⌘Z)" and "Redo (⌘⇧Z)".
   - A scrollable list of transformations drawn from the **raw** history (not the
     collapsed render list). The raw history is shown because each entry represents
     a distinct user action and can be individually undone. Displayed in
     reverse-chronological order (most recent on top).
   - Each entry displays the transformation name and, for parameterized variants,
     the value (e.g., "Brightness: 0.35").
   - **Styling:**
     - Active entries (`history.applied_transforms()`, newest first) use normal
       text color.
     - A subtle visual divider separates active from undone entries.
     - Undone entries (`history.redo_transforms()`, in redo order) appear below
       the divider with dimmed/grayed text. They are shown so the user can see
       what would be restored by Redo.
   - **Size constraint:** The history list area has a fixed max height designed to
     show approximately 5 items. If the list exceeds this, an `iced::widget::scrollable`
     enables traversal, preserving screen real estate for the Transform component.
   - **Undo** is disabled when the applied stack is empty. **Redo** is disabled
     when the redo stack is empty.

## 2. Application State Model

The `iced` application struct holds the following state:

```
struct BdipApp {
    // --- Image state ---
    base_image: Option<Rgba16Image>,          // Pristine loaded image (CPU)
    image_handle: Option<iced::widget::image::Handle>,  // 8-bit display handle

    // --- GPU state ---
    engine: Option<GpuEngine>,                   // None if GPU init failed
    renderer: Option<Renderer>,                  // None if GPU init failed
    cached_base_texture: Option<wgpu::Texture>,  // Ingested linear texture

    // --- Transform state ---
    history: HistoryManager,
    selected_transform: TransformOption,      // Currently selected pick_list item
    preview_value: f32,                       // Current slider display value (see §2.1)
    is_previewing: bool,                      // Whether a slider drag is in progress

    // --- UI state ---
    error_message: Option<String>,            // Displayed in a modal/banner
    is_loading: bool,                         // Shows a loading indicator
    is_saving: bool,                          // Prevents double-save
}
```

**`TransformOption`** is a UI-only enum used for the `pick_list`. It maps to
`bdip_core::Transformation` variants but does not carry parameter values:
```
enum TransformOption {
    Brightness,
    Contrast,
    Saturation,
    Grayscale,
    Invert,
}
```

### GPU Texture Caching Strategy

When an image is loaded, the full-resolution base image is uploaded to the GPU and
ingested (sRGB → linear) exactly once. The resulting `wgpu::Texture` is cached in
`cached_base_texture`. On every render (commit or preview), the pipeline replays
the *collapsed* transform list starting from this cached texture — avoiding the
expensive upload and ingest on every edit. See Section 2.1 for how the collapsed
list is derived from the raw history.

The render helper (`render_to_handle`) signature:
```
fn render_to_handle(&mut self, preview: Option<&Transformation>)
    -> Option<iced::widget::image::Handle>
```

The replay loop:
```
let committed = history.applied_transforms().to_vec();
let collapsed = collapse_adjacent(&committed);

let render_list = match preview {
    Some(p) => {
        // Replace the trailing entry of the same type, or append.
        if collapsed.last() has same type as p {
            collapsed.pop(); collapsed.push(p.clone())
        } else {
            collapsed.push(p.clone())
        }
        collapsed
    }
    None => collapsed,
};

let mut tex = &cached_base_texture;
for t in &render_list {
    tex = renderer.apply(&engine, &tex, t);
}
let buf = renderer.present(&engine, &tex);
let img = download_presentation_buffer(..., &buf, w, h)?;
// Convert to 8-bit, update image_handle
```

**Note on texture cloning:** `wgpu::Texture` cannot be cloned. On each render pass
the pipeline starts from `cached_base_texture` (a reference) and produces new
intermediate textures via `renderer.apply()`. The cached texture is never consumed
or moved.

## 2.1. Slider/History Model

This section is the authoritative specification of how parameterized slider
transforms interact with history, preview, and rendering.

### Core Principles

**History stores absolute slider values.** Each slider release pushes the
current slider position as an absolute value to `HistoryManager`. Releasing at
0.7 records `Brightness(0.7)`, not a delta from the previous release.

**Render collapses adjacent runs.** Before applying transforms to the GPU, the
raw history is collapsed: consecutive entries of the same type are reduced to
the last entry in each run. The raw history is the source of truth for undo/redo;
the collapsed list is only used for rendering.

```
Raw:      [B(0.3), B(0.7), S(0.5), S(0.3), B(0.1)]
Collapsed: [B(0.7), S(0.3), B(0.1)]
```

**Slider stays where released.** After `SliderReleased`, `preview_value` is not
reset to `0.0`. The slider shows the committed position. Moving back to `0.0`
means "no brightness adjustment at this point in the stack."

**Slider resets when a different transform interrupts.** When switching back to a
transform type, the slider shows the last committed value only if the trailing
entry in history is of that type. If a different type was applied after it, the
slider resets to `0.0`, because the subsequent transform has changed the image
relative to which the original adjustment was made.

### Slider Position on Transform Switch

`slider_value_for_type(opt, history)` returns:
- The value of `history.last()` if its type matches `opt`.
- `0.0` otherwise (history is empty, or the last entry is a different type).

```
history = [B(0.7), S(0.3)]
switch to Saturation  → last is S(0.3), matches → slider shows 0.3
switch to Brightness  → last is S(0.3), no match → slider shows 0.0

history = [B(0.3), B(0.7)]
switch to Brightness  → last is B(0.7), matches → slider shows 0.7
switch to Saturation  → last is B(0.7), no match → slider shows 0.0
```

### Live Preview Render List

During a drag, the render uses the collapsed history but replaces the trailing
entry of the same type as the active slider (or appends if the type differs):

```
history = [B(0.7), S(0.2)], dragging S to 0.4
  collapsed = [B(0.7), S(0.2)]
  last is S, matches preview type → replace → render [B(0.7), S(0.4)]

history = [B(0.7), S(0.2)], switched to B, dragging to 0.3
  collapsed = [B(0.7), S(0.2)]
  last is S, does not match B → append → render [B(0.7), S(0.2), B(0.3)]
```

### Worked Examples

**Example 1 — Fine-tuning brightness:**
```
User drags B slider to 0.5, releases → history: [B(0.5)], slider: 0.5
User drags B slider to 0.7, releases → history: [B(0.5), B(0.7)], slider: 0.7
  render (collapsed): [B(0.7)]        ← only latest B in the trailing run
Undo → history: [B(0.5)], slider: 0.5
  render (collapsed): [B(0.5)]
Redo → history: [B(0.5), B(0.7)], slider: 0.7
```

**Example 2 — Interleaved transforms:**
```
Drag B to 0.7, release → history: [B(0.7)]
Drag S to 0.4, release → history: [B(0.7), S(0.4)]
  render (collapsed): [B(0.7), S(0.4)]
Switch to Brightness → last entry is S(0.4), not B → slider resets to 0.0
Drag B to 0.3, release → history: [B(0.7), S(0.4), B(0.3)]
  render (collapsed): [B(0.7), S(0.4), B(0.3)]
Undo → history: [B(0.7), S(0.4)]
  render (collapsed): [B(0.7), S(0.4)]
  switch to Brightness now → last is S(0.4), slider resets to 0.0
```

**Example 3 — Preview during drag on trailing-same-type:**
```
history = [B(0.7), S(0.2)], currently viewing S slider at 0.2
User drags S to 0.4 (not released yet):
  preview render list: [B(0.7), S(0.4)]   ← S(0.2) replaced by preview
User releases at 0.4:
  history: [B(0.7), S(0.2), S(0.4)]
  render (collapsed): [B(0.7), S(0.4)]    ← S(0.2) collapsed away
  slider stays at 0.4
```

**Example 4 — Why non-additive shaders require absolute values:**
Saturation uses `mix(gray, color, 1.0 + offset)`, which is multiplicative.
Applying `S(0.5)` (scale 1.5×) then `S(0.3)` (scale 1.3×) gives 1.95×, not
the 1.8× implied by a slider at 0.8. By collapsing adjacent entries and using
only the latest value (`S(0.8)`), the render applies exactly one pass at 1.8×,
which matches what the slider shows.

## 3. Message Enum

```
enum Message {
    // --- I/O ---
    LoadImagePressed,
    ImageLoaded(Result<(PathBuf, Rgba16Image), String>),
    SaveImagePressed,
    ImageSaved(Result<PathBuf, String>),

    // --- Transform controls ---
    TransformSelected(TransformOption),
    SliderChanged(f32),
    SliderReleased,
    ApplyParameterless,

    // --- History ---
    Undo,
    Redo,

    // --- Error handling ---
    DismissError,

    // --- Misc ---
    Noop,  // No-op for cancelled file pickers, etc.
}
```

## 4. Async I/O & Error Handling

These address the two Phase 5 tech debt requirements from `specs/tech_debt.md`.

### 4.1. Non-blocking File I/O

All file I/O is performed asynchronously via `iced::Task` to prevent blocking the
UI thread:

- **Load:** `LoadImagePressed` spawns a `Task` that:
  1. Opens `rfd::AsyncFileDialog` with filters for supported formats
     (png, jpg/jpeg, gif, tiff/tif).
  2. On file selection, calls `bdip_core::io::load_image()` on a background
     thread (via `Task::perform` with a blocking closure).
  3. Returns `Message::ImageLoaded(Result<(PathBuf, Rgba16Image), String>)`.
- **Save:** `SaveImagePressed` spawns a `Task` that:
  1. Opens `rfd::AsyncFileDialog` in save mode with the same format filters.
  2. Runs the full GPU pipeline (present + download) and then
     `bdip_core::io::save_image()` on a background thread.
  3. Returns `Message::ImageSaved(Result<PathBuf, String>)`.

### 4.2. No Panics in Constructor

The application constructor (`BdipApp::new`) returns immediately with an empty
state (`base_image: None`, placeholder canvas). All fallible initialization is
deferred:

- **GPU initialization:** `GpuEngine::new()` is called in the constructor. If
  this fails (no GPU), the app should display an error message in the canvas area
  and disable all transform/load functionality. GPU init failure is the only case
  where a startup error is surfaced inline rather than via a modal.
- **CLI preload path:** If a file path is provided via CLI args (`bdip foo.jpg`),
  the constructor dispatches a `Task` that loads the image asynchronously, exactly
  like `LoadImagePressed`. The user sees the empty canvas briefly, then the image
  appears once loaded.

### 4.3. Error Display

Errors from I/O or GPU operations are stored in `error_message: Option<String>`
and displayed as a dismissible banner or overlay at the top of the canvas area.
The `DismissError` message clears the banner. Errors are never panics.

## 5. Keyboard Shortcuts

- **⌘Z** → `Message::Undo`
- **⌘⇧Z** → `Message::Redo`

Implemented via `iced::keyboard::on_key_press` subscription. The subscription
maps the key combination to the appropriate `Message`.

## 6. Module Structure

The current `ui_spike.rs` is replaced by a `ui/` module directory inside `bdip/src/`:

```
bdip/src/
├── main.rs          # CLI router (unchanged structure)
├── cli.rs           # clap definitions (updated for new transforms)
└── ui/
    ├── mod.rs       # Re-exports, `run()` entry point
    ├── app.rs       # BdipApp struct, new(), update(), view(), subscription()
    ├── sidebar.rs   # Transform picker + slider/button, history list
    ├── canvas.rs    # Central image display widget
    ├── menu_bar.rs  # Top bar with Load/Save buttons
    ├── style.rs     # Custom styling (button colors, history item styles)
    └── message.rs   # Message enum and TransformOption enum
```

`ui_spike.rs` is deleted once the new UI module is functional. During development
it can coexist — `main.rs` can be switched to call `ui::run()` instead of
`ui_spike::run()` as the final integration step.

## 7. V1 Shader Completion — Phase 6 (Contrast, Grayscale, Invert)

Three V1 transformations remain unimplemented as GPU shaders: Contrast,
Grayscale, and Invert. These are Phase 6 work (PRs 6–8), sequenced after
Phase 5 (UI integration) is complete. Each shader is an independent unit of
work. The `Transformation` enum variants already exist in `bdip_core`; only
the GPU shaders, pipeline wiring, CLI parsing, and UI `pick_list` entries
need to be added.

### 7.1. Contrast Shader
- **File:** `bdip_core/src/gpu/contrast.wgsl`
- **Params:** `ContrastParams { contrast_offset: f32, _padding: [f32; 3] }`
- **Algorithm:** For each channel: `(pixel - 0.5) * (1.0 + contrast) + 0.5`,
  clamped to [0.0, 1.0]. Operates in linear space.
- **Pipeline touch points:** Add `TransformKind::Contrast`, `From` impl,
  `compile()` match arm, `apply()` match arm (per `specs/adding_a_shader.md`).

### 7.2. Grayscale Shader
- **File:** `bdip_core/src/gpu/grayscale.wgsl`
- **Params:** No user params. Use a dummy 16-byte uniform
  `GrayscaleParams { _unused: [f32; 4] }` to satisfy the bind group layout, OR
  use a separate bind group layout with no params group. The dummy uniform
  approach is simpler and consistent with the existing architecture.
- **Algorithm:** Luminance = `0.2126 * R + 0.7152 * G + 0.0722 * B` (ITU-R
  BT.709 coefficients, correct for linear-light values). Set R=G=B=luminance.
- **Pipeline touch points:** Same 4-point checklist.

### 7.3. Invert Shader
- **File:** `bdip_core/src/gpu/invert.wgsl`
- **Params:** No user params. Same dummy uniform approach as Grayscale.
- **Algorithm:** `1.0 - channel` for R, G, B. Alpha unchanged.
- **Pipeline touch points:** Same 4-point checklist.

### 7.4. CLI Parity

`parse_transform()` in `bdip/src/main.rs` currently only handles `brightness` and
`saturation`. Each Phase 6 shader PR adds its own CLI parsing:
- PR 6: `contrast:<f32>` → `Transformation::Contrast(val)`
- PR 7: `grayscale` → `Transformation::Grayscale`
- PR 8: `invert` → `Transformation::Invert`

## 8. Rendering Pipeline Integration Detail

This section specifies exactly how the UI wires into the `bdip_core` GPU pipeline,
addressing the "CPU Bridge" architecture from the main spec.

### 8.1. On Image Load
1. Receive `Rgba16Image` from async I/O task.
2. Store as `base_image`.
3. Call `upload_texture()` → `renderer.ingest()` → store result as
   `cached_base_texture`.
4. Call `renderer.present()` on the ingested texture (no transforms yet).
5. Call `download_presentation_buffer()` → convert to `Rgba8` →
   create `image::Handle` → store as `image_handle`.
6. Clear `HistoryManager`.

### 8.2. On Transform Commit (slider release)
1. Push the absolute slider value as a `Transformation` onto `HistoryManager`.
   The slider position does NOT reset — `preview_value` stays at its released
   value.
2. Clear `is_previewing`.
3. Build the collapsed render list from the now-updated history (see Section 2.1).
4. Replay `render_to_handle(None)` from `cached_base_texture` → update
   `image_handle`.

For parameterless transforms (Apply button), the flow is identical except the
pushed value is always the fixed parameterless transform (e.g., `Grayscale`).

### 8.3. On Slider Drag (live preview)
1. Set `is_previewing = true`, `preview_value = slider_value`.
2. Do NOT push to `HistoryManager` — this is ephemeral.
3. Build the render list via `render_to_handle(Some(&preview_transform))`:
   - Collapse the committed history.
   - If the collapsed list's last entry is the same type as the active slider,
     replace it with the preview value. Otherwise, append.
4. `renderer.present()` → `download_presentation_buffer()` → update
   `image_handle`.

The net effect: the canvas reflects exactly what the final committed result would
look like if the user released at the current drag position.

### 8.4. On Undo / Redo
1. Call `history.undo()` or `history.redo()`.
2. Update `preview_value` by calling `slider_value_for_type` on the updated
   history so the slider reflects the new last committed value for the selected
   transform type (or 0.0 if that type is no longer the trailing entry).
3. Build the collapsed render list from the updated history.
4. `renderer.present()` → `download_presentation_buffer()` → update
   `image_handle`.

### 8.5. Performance Consideration

Every slider movement triggers a full-stack replay. Measured on Apple Silicon
(release build, large TIFF):

| Stage | Typical latency |
|---|---|
| Shader dispatch per transform (`renderer.apply`) | ~1–3ms |
| `renderer.present` | ~1–3ms |
| `download_presentation_buffer` (GPU readback) | ~40–55ms |
| `into_rgba8` conversion | ~10–15ms |
| **Total per drag event** | **~65–75ms** |

At ~70ms per event the UI is responsive enough for interactive dragging in
release builds. Debug builds are substantially slower (~700ms) due to
unoptimized `into_rgba8` — always test slider performance in release mode.

The `download_presentation_buffer` cost is the known tech debt item from
`specs/tech_debt.md` (synchronous GPU readback blocks the UI thread). If
latency becomes perceptible with very large images or future shaders with
higher dispatch costs, moving the readback into an `iced::Task` is the
remediation path — see that document for details.

## 9. Implementation PRs

The work is organized into independently reviewable PRs. Each PR should be
self-contained: it compiles, passes `cargo clippy`, and existing tests continue
to pass. PRs are ordered by dependency — later PRs may depend on earlier ones
being merged.

PRs 1–5 are **Phase 5** (UI integration) and use only the two existing shaders
(Brightness, Saturation). PRs 6–8 are **Phase 6** (V1 shader completion) and are
independent of each other — they can be implemented and reviewed in parallel.

### Dependency Graph

```
PR 1 (UI Scaffold) ──┬──→ PR 2 (Transform Controls) ──→ PR 3 (History & Undo/Redo)
                      │
                      └──→ PR 4 (Save & Error Handling)
                                                            ↓
                                                   PR 5 (Polish & Integration Testing)
                                                            ↓
                                          PR 6, PR 7, PR 8 (independent, parallel)
```

---

### PR 1: UI Module Scaffold & Application Shell

**Goal:** Replace the `ui_spike.rs` with the new `ui/` module structure. Wire up
the `BdipApp` struct with async initialization, dark theme, and the three-zone
layout (menu bar, sidebar, canvas) — but with placeholder/stub content in each
zone. No GPU integration yet beyond what's needed to display a loaded image.

**Scope:**
- Create `bdip/src/ui/` module directory with `mod.rs`, `app.rs`, `message.rs`,
  `sidebar.rs`, `canvas.rs`, `menu_bar.rs`, `style.rs`.
- Implement `BdipApp` struct with the state model from Section 2.
- Implement `Message` enum (Section 3).
- Implement `new()` that returns immediately with empty state (no panics). If a
  CLI path is provided, dispatch an async load task.
  **[Addresses tech_debt.md: Panic-on-Failure in Application Constructor]**
- Implement async image loading via `rfd::AsyncFileDialog` and `iced::Task`.
  **[Addresses tech_debt.md: Synchronous Disk I/O Blocks UI Initialization]**
- Wire `LoadImagePressed` / `ImageLoaded` flow: file picker → background load →
  GPU upload + ingest → present → display on canvas.
- Implement the three-zone layout in `view()`: menu bar on top, sidebar on left,
  canvas in center.
- Sidebar contains the `pick_list` and a placeholder "History" label.
- Canvas displays loaded image or placeholder text.
- Add `rfd` dependency to `bdip/Cargo.toml`.
- Update `main.rs` to call `ui::run()` instead of `ui_spike::run()`.
- Delete `ui_spike.rs`.

#### Application Entry Point (`ui/mod.rs`)

The `run()` function uses the function-based `iced::application` API established
in the spike. It must set the dark theme:

```rust
pub fn run(input: Option<PathBuf>) -> anyhow::Result<()> {
    iced::application(
        move || BdipApp::new(input.clone()),
        BdipApp::update,
        BdipApp::view,
    )
    .title("bdip")
    .theme(|_| iced::Theme::Dark)
    .subscription(BdipApp::subscription)
    .run()
    .context("Failed to run iced application")?;
    Ok(())
}
```

The `subscription` method is wired here for keyboard shortcuts (used in PR 3).
It can return `Subscription::none()` in this PR.

#### Constructor (`BdipApp::new`)

`GpuEngine::new()` uses `pollster::block_on` internally and completes in under
1ms — it is safe to call synchronously in the constructor. `Renderer::new()`
is also synchronous and cheap. Both are initialized here, not deferred.

If GPU init fails, store the error in `error_message` and set `engine` /
`renderer` to `None` (change the struct field types to `Option<GpuEngine>` and
`Option<Renderer>` accordingly). All transform and load operations check these
are `Some` before proceeding.

If a CLI path is provided, the constructor dispatches a load task immediately
(same code path as `LoadImagePressed`, see below).

```rust
fn new(input_path: Option<PathBuf>) -> (Self, Task<Message>) {
    let (engine, renderer, error) = match GpuEngine::new() {
        Ok(e) => {
            let r = Renderer::new(&e);
            (Some(e), Some(r), None)
        }
        Err(e) => (None, None, Some(format!("GPU init failed: {e}"))),
    };

    let task = match input_path {
        Some(path) => load_image_task(path),  // reusable helper
        None => Task::none(),
    };

    (BdipApp {
        base_image: None,
        image_handle: None,
        engine,
        renderer,
        cached_base_texture: None,
        history: HistoryManager::new(),
        selected_transform: TransformOption::Brightness,
        preview_value: 0.0,
        is_previewing: false,
        error_message: error,
        is_loading: input_path.is_some(),
        is_saving: false,
    }, task)
}
```

#### Async Image Loading

The load flow has two trigger points that share the same task helper:

1. **"Load Image" button** → `LoadImagePressed` opens the file picker first,
   then loads.
2. **CLI preload path** → skips the file picker, loads directly.

The shared helper for the actual I/O:
```rust
fn load_image_task(path: PathBuf) -> Task<Message> {
    Task::perform(
        async move {
            // This closure runs on a background thread.
            // bdip_core::io::load_image is synchronous/blocking — that's fine
            // inside Task::perform.
            let img = bdip_core::io::load_image(&path)
                .map_err(|e| e.to_string())?;
            Ok((path, img))
        },
        Message::ImageLoaded,
    )
}
```

For the file-picker path (`LoadImagePressed`):
```rust
Message::LoadImagePressed => {
    Task::perform(
        async {
            let handle = rfd::AsyncFileDialog::new()
                .add_filter("Images", &["png", "jpg", "jpeg", "gif", "tif", "tiff"])
                .pick_file()
                .await;
            match handle {
                Some(h) => {
                    let path = h.path().to_path_buf();
                    let img = bdip_core::io::load_image(&path)
                        .map_err(|e| e.to_string())?;
                    Ok((path, img))
                }
                None => Err("cancelled".into()),
            }
        },
        |result: Result<_, String>| match result {
            Ok(pair) => Message::ImageLoaded(Ok(pair)),
            Err(e) if e == "cancelled" => Message::Noop,
            Err(e) => Message::ImageLoaded(Err(e)),
        },
    )
}
```

**Note on `Message::Noop`:** Add a `Noop` variant to `Message` for cases like
a cancelled file picker where no action is needed. The update handler simply
returns `Task::none()`.

#### `ImageLoaded` Handler

When the async load completes successfully, the handler performs GPU upload and
ingest synchronously (this is fast — see Section 8.5):

```rust
Message::ImageLoaded(Ok((path, img))) => {
    let (w, h) = img.dimensions();
    // GPU upload + ingest (synchronous, ~1-4ms on Apple Silicon)
    if let (Some(engine), Some(renderer)) = (&self.engine, &mut self.renderer) {
        let uploaded = upload_texture(&engine.device, &engine.queue, &img);
        let linear = renderer.ingest(engine, &uploaded);
        // Present the unmodified image for initial display
        let buf = renderer.present(engine, &linear);
        let handle = presentation_to_handle(engine, &buf, w, h);
        self.cached_base_texture = Some(linear);
        self.image_handle = handle;
    }
    self.base_image = Some(img);
    self.history.clear();
    self.is_loading = false;
    self.error_message = None;
    Task::none()
}
```

#### 16-bit → 8-bit Conversion Helper

The spike (lines 75–81 of `ui_spike.rs`) contains the pattern for converting a
presentation buffer to an iced image handle. This will be called on every render
(load, commit, preview, undo/redo), so extract it as a reusable function in
`canvas.rs` or `app.rs`:

```rust
fn presentation_to_handle(
    engine: &GpuEngine,
    buf: &wgpu::Buffer,
    width: u32,
    height: u32,
) -> Option<iced::widget::image::Handle> {
    let img16 = download_presentation_buffer(
        &engine.device, &engine.queue, buf, width, height,
    ).ok()?;
    let img8 = bdip_core::image::DynamicImage::ImageRgba16(img16).into_rgba8();
    let (w, h) = img8.dimensions();
    let pixels = img8.into_raw();
    Some(iced::widget::image::Handle::from_rgba(w, h, pixels))
}
```

This helper is reused extensively in PRs 2–4 whenever the canvas needs updating.

#### `view()` Layout Composition

The top-level `view()` builds the three-zone layout:

```rust
fn view(&self) -> Element<'_, Message> {
    let menu = menu_bar::view(self);   // Row: [Load] [Save]
    let sidebar = sidebar::view(self); // Column: pick_list, "History"
    let canvas = canvas::view(self);   // Image or placeholder

    let content = row![
        container(sidebar).width(Length::Fixed(250.0)),
        container(canvas).width(Length::Fill).height(Length::Fill),
    ];

    column![menu, content]
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}
```

#### Module Responsibilities

Each file has a focused role:

| File | Exports | Responsibility |
|------|---------|----------------|
| `mod.rs` | `pub fn run(...)` | Application entry point, iced wiring |
| `app.rs` | `BdipApp`, `new()`, `update()`, `view()`, `subscription()` | State struct, message dispatch, layout composition |
| `message.rs` | `Message`, `TransformOption` | Enum definitions. `Message` must derive `Debug, Clone`. |
| `sidebar.rs` | `pub fn view(app: &BdipApp) -> Element<Message>` | Transform picker + placeholder history |
| `canvas.rs` | `pub fn view(app: &BdipApp) -> Element<Message>`, `presentation_to_handle()` | Image display, placeholder text, error overlay (PR 4) |
| `menu_bar.rs` | `pub fn view(app: &BdipApp) -> Element<Message>` | Load/Save buttons |
| `style.rs` | (empty or minimal in PR 1) | Custom styling, used more heavily in PRs 3–4 |

**Key files:**
- `bdip/src/ui/mod.rs` (new)
- `bdip/src/ui/app.rs` (new)
- `bdip/src/ui/message.rs` (new)
- `bdip/src/ui/sidebar.rs` (new)
- `bdip/src/ui/canvas.rs` (new)
- `bdip/src/ui/menu_bar.rs` (new)
- `bdip/src/ui/style.rs` (new)
- `bdip/src/main.rs` (modify)
- `bdip/src/ui_spike.rs` (delete)
- `bdip/Cargo.toml` (add `rfd` dependency)

**References:**
- Section 1 (layout), Section 2 (state model), Section 3 (messages),
  Section 4 (async I/O), Section 6 (module structure) of this document.
- `specs/tech_debt.md` — the two Phase 5 required items.
- `bdip/src/ui_spike.rs` — working iced application pattern and 16-bit → 8-bit
  conversion code to reference (but not copy verbatim — the spike has the
  problems this PR is fixing).

**Verification:**
- App launches in dark mode with empty canvas and menu bar.
- "Load Image" opens a file picker; selected image appears on canvas.
- `bdip foo.jpg` (CLI preload) launches the app and displays the image after
  a brief empty-canvas flash.
- `bdip` (no args) launches with placeholder text, no errors.
- Invalid CLI path (e.g., `bdip nonexistent.jpg`) shows an error message in
  the UI, does not panic.
- `cargo clippy --workspace` passes.

---

### PR 2: Transform Controls & Live Preview

**Goal:** Wire the transform picker and slider to the GPU pipeline for live
preview and committed transforms. This is the core interactive editing flow.
This PR only needs to support Brightness and Saturation (the two existing
shaders). Contrast, Grayscale, and Invert are added in PRs 6–8 and will
automatically appear in the `pick_list` once their shaders exist.

**Depends on:** PR 1 (UI scaffold).

**Scope:**
- Implement `TransformOption` enum and `pick_list` integration in sidebar.
  Initially populated with `Brightness` and `Saturation` only. Design the
  enum so that adding new variants later is trivial (Contrast, Grayscale,
  Invert will be added in PRs 6–8).
- Implement dynamic widget switching: slider for parameterized transforms,
  "Apply" button for parameterless transforms. The "Apply" button path can
  be stubbed or built proactively — there are no parameterless transforms
  yet, but the code path should exist so PRs 6–8 only need to add enum
  variants.
- Wire `SliderChanged` → live preview pipeline (Section 8.3).
- Wire `SliderReleased` → commit to `HistoryManager` (Section 8.2).
- Wire `ApplyParameterless` → commit flow (Section 8.2). May be untestable
  until Grayscale/Invert are added, but the handler should exist.
- Implement the GPU texture caching strategy (Section 2): upload + ingest
  once on load, replay collapsed history from cached texture on every edit.
- Implement `collapse_adjacent` and `slider_value_for_type` per Section 2.1.
  Add `TransformOption::from_transformation()` to `message.rs`.
- Display current slider value as formatted text.
- Slider step size must be set to `0.01` — iced's default step for `f32` is
  `1.0`, which produces only -1, 0, and 1 as values.
- Slider stays at its released position — does NOT reset to `0.0`.
- `TransformSelected` loads the slider position from history via
  `slider_value_for_type` (Section 2.1).

**Key files:**
- `bdip/src/ui/sidebar.rs` (modify — transform picker + slider)
- `bdip/src/ui/app.rs` (modify — update handler for transform messages,
  `collapse_adjacent`, `slider_value_for_type`, `render_to_handle`)
- `bdip/src/ui/message.rs` (modify — add `TransformOption::from_transformation`)

**References:**
- Section 1.3a (transform component), Section 2.1 (slider/history model),
  Section 8.2–8.3 (pipeline integration) of this document.

**Verification:**
- Select "Brightness" → slider appears → dragging updates canvas in real time.
- Release slider at 0.35 → transform committed, slider stays at 0.35.
- Drag slider from 0.35 to 0.70, release → history has two entries; canvas
  shows result of brightness at 0.70 (adjacent collapse).
- Switch to "Saturation", drag and release → slider shows saturation value.
- Switch back to "Brightness" → slider shows 0.0 (interrupted by Saturation).
- Switch back to "Saturation" → slider shows last Saturation value.
- Select "Saturation" with no prior saturation commits → slider at 0.0.
- `cargo clippy --workspace` passes.

---

### PR 3: History UI & Undo/Redo

**Goal:** Implement the history list visualization and undo/redo controls,
including keyboard shortcuts.

**Depends on:** PR 2 (transforms must be committable to test history).

**Scope:**
- Implement history list widget in sidebar (Section 1.3b): scrollable,
  reverse-chronological, max ~5 visible items.
- Display transform name + value for each history entry. This requires
  adding a `std::fmt::Display` impl for `Transformation` in
  `bdip_core/src/transformation.rs`. Format: `"Brightness: 0.35"` for
  parameterized variants, `"Grayscale"` for parameterless variants.
- Style active entries with normal text, undone entries with dimmed text.
- Wire Undo/Redo buttons to `HistoryManager::undo()` / `redo()` + pipeline
  replay (Section 8.4).
- After undo/redo, update `preview_value` via `slider_value_for_type` so the
  active slider reflects the new trailing committed value (or 0.0 if the type
  is no longer the trailing entry). This keeps the slider in sync with history
  without requiring the user to switch transforms.
- Disable Undo when applied stack is empty; disable Redo when redo stack is
  empty.
- Implement `iced::keyboard::on_key_press` subscription for ⌘Z / ⌘⇧Z.
- Expose redo and undo state from `HistoryManager`. Currently only
  `applied_transforms()` is public. Add:
  - `can_undo() -> bool` — true when the applied stack is non-empty.
  - `can_redo() -> bool` — true when the redo stack is non-empty.
  - `redo_transforms() -> &[Transformation]` — the redo stack in redo order
    (index 0 is the next item to be re-applied by Redo). Used by the sidebar
    to render dimmed undone entries below the divider.

**Key files:**
- `bdip/src/ui/sidebar.rs` (modify — history list)
- `bdip/src/ui/app.rs` (modify — undo/redo handlers, subscription)
- `bdip/src/ui/style.rs` (modify — dimmed text style for undone items)
- `bdip_core/src/transformation.rs` (modify — add `Display` impl)
- `bdip_core/src/history.rs` (modify — add `can_undo()`, `can_redo()`,
  and `redo_transforms()` accessors)

**Already implemented in PR 2 — do not re-implement:**
- `render_to_handle(preview: Option<&Transformation>)` in `app.rs` — call
  this with `None` after undo/redo to re-render from the updated history.
- `slider_value_for_type(opt, history)` in `app.rs` — call this after
  undo/redo to sync `preview_value` with the new history state.
- `collapse_adjacent` in `app.rs` — used internally by `render_to_handle`.
- `TransformOption::from_transformation` in `message.rs`.

**References:**
- Section 1.3b (history component), Section 2.1 (slider/history model),
  Section 5 (keyboard shortcuts), Section 8.4 (undo/redo pipeline).

**Verification:**
- Apply 3 transforms → history list shows all 3 in reverse order.
- Apply B(0.3) then B(0.7) → list shows both; canvas shows B(0.7) result.
- Undo → top item grays out, canvas reverts to B(0.3), slider moves to 0.3.
- Redo → item re-activates, canvas shows B(0.7), slider moves to 0.7.
- ⌘Z and ⌘⇧Z work.
- Apply 6+ transforms → scrollbar appears in history.
- Undo all → Undo button disabled. Redo all → Redo button disabled.
- Undo past a type boundary → slider for active transform resets to 0.0.
- `cargo clippy --workspace` passes.

---

### PR 4: Save Workflow & Error Handling

**Goal:** Implement the save file dialog and comprehensive error handling
throughout the UI.

**Depends on:** PR 1 (UI scaffold with load working).

**Scope:**
- Wire `SaveImagePressed` → `rfd::AsyncFileDialog` (save mode) → background
  pipeline execution → `bdip_core::io::save_image()`.
- `ImageSaved` handler: show success feedback or error.
- Disable "Save Image" button when no image is loaded.
- Implement error banner/overlay in the canvas area (Section 4.3).
- `DismissError` message to clear the banner.
- Handle all error paths: load failure (bad file, unsupported format), save
  failure (permissions, invalid path), GPU errors.
- Handle the "user cancelled file picker" case (no file selected — not an
  error, just a no-op).

**Key files:**
- `bdip/src/ui/app.rs` (modify — save flow, error handling)
- `bdip/src/ui/canvas.rs` (modify — error banner overlay)
- `bdip/src/ui/menu_bar.rs` (modify — disable save button)
- `bdip/src/ui/message.rs` (modify — save messages)

**References:**
- Section 4 (async I/O & error handling) of this document.

**Verification:**
- "Save Image" is disabled when no image is loaded.
- Load → apply transform → Save → file picker → select path → file saved.
- Attempt to load a non-image file → error banner appears → dismiss works.
- Cancel file picker → no error, app remains functional.
- `cargo clippy --workspace` passes.

---

### PR 5: UI Polish & Integration Testing

**Goal:** Final cleanup, visual polish, and end-to-end validation of the UI
and CLI workflows using the two existing shaders (Brightness, Saturation).
This PR closes out Phase 5.

**Depends on:** PRs 1–4.

**Scope:**
- Visual polish: consistent spacing, padding, font sizes in sidebar/menu bar.
- Verify dark theme consistency across all widgets.
- Add/update integration tests for CLI with Brightness and Saturation.
- Manual end-to-end test checklist (documented in PR description):
  - Cold launch → empty canvas.
  - Load PNG, JPG, TIFF, GIF → each displays correctly.
  - Apply Brightness and Saturation → visual result is correct.
  - Stack multiple transforms → result is correct.
  - Undo/Redo full stack → canvas matches expectations.
  - Save as PNG, JPG, TIFF → output file is valid.
  - CLI headless mode with existing transforms → output is correct.
- Clean up any remaining references to `ui_spike` in comments or docs.
- Run `cargo clippy --workspace` and `rustfmt` on all files.

**Key files:**
- Various UI files (minor adjustments)
- `bdip/tests/e2e_cli_pipeline.rs` (modify — add/verify transform tests)

**Verification:**
- All automated tests pass: `cargo test --workspace`.
- `cargo clippy --workspace` clean.
- Manual checklist completed and documented in PR.

---

### PR 6: Contrast Shader (Phase 6)

**Goal:** Implement the Contrast GPU shader and wire it into both CLI and UI.

**Depends on:** PR 5 (Phase 5 complete). Independent of PRs 7 and 8.

**Scope:**
- Add `contrast.wgsl` in `bdip_core/src/gpu/`.
- Add `ContrastParams` uniform struct in `pipeline.rs`.
- Note: `Transformation::Contrast(f32)` already exists in
  `bdip_core/src/transformation.rs` — no changes to that file are needed.
- Add `TransformKind::Contrast` variant, `From` impl, `compile()` match arm,
  `apply()` match arm (per `specs/adding_a_shader.md`).
- Add `contrast:<f32>` parsing to `parse_transform()` in `bdip/src/main.rs`.
- Add `TransformOption::Contrast` to the UI `pick_list` in
  `bdip/src/ui/sidebar.rs` (or `message.rs`).
- Unit tests: identity case (`contrast:0.0` unchanged), extreme values.
- Integration test: multi-transform pipeline including contrast chained with
  an existing shader (e.g., brightness then contrast).

**Key files:**
- `bdip_core/src/gpu/contrast.wgsl` (new)
- `bdip_core/src/gpu/pipeline.rs` (modify)
- `bdip/src/main.rs` (modify `parse_transform`)
- `bdip/src/ui/sidebar.rs` or `bdip/src/ui/message.rs` (modify — add variant)

**References:**
- `specs/adding_a_shader.md` — the 4-point checklist.
- `specs/transformations_reference.md` — algorithm details.
- Section 7.1 of this document for contrast shader spec.

**Verification:**
```
$ bdip --headless test.jpg --output out.png --apply contrast:0.5
$ bdip --headless test.jpg --output out.png --apply contrast:0.0
  (output matches input)
$ cargo test -p bdip_core
$ cargo clippy --workspace
```
- UI: Select "Contrast" → slider appears → dragging previews, release commits.

---

### PR 7: Grayscale Shader (Phase 6)

**Goal:** Implement the Grayscale GPU shader and wire it into both CLI and UI.

**Depends on:** PR 5 (Phase 5 complete). Independent of PRs 6 and 8.

**Scope:**
- Add `grayscale.wgsl` in `bdip_core/src/gpu/`.
- Note: `Transformation::Grayscale` already exists in
  `bdip_core/src/transformation.rs` — no changes to that file are needed.
- Add `GrayscaleParams` dummy uniform struct in `pipeline.rs` (16-byte aligned,
  no user-facing parameters — exists only to satisfy the bind group layout).
- Add `TransformKind::Grayscale` variant, `From` impl, `compile()` match arm,
  `apply()` match arm. Because `Transformation::Grayscale` carries no payload,
  the `apply()` arm must create the params buffer from a zeroed
  `GrayscaleParams` directly: `GrayscaleParams { _unused: [0.0; 4] }`.
- Add `grayscale` parsing to `parse_transform()` in `bdip/src/main.rs`.
- Add `TransformOption::Grayscale` to the UI `pick_list`. This is a
  parameterless transform — verify the "Apply" button path built in PR 2
  works correctly.
- Unit tests: grayscale produces equal R=G=B channels, preserves alpha,
  extreme input values (all-white, all-black) produce correct luminance.
- Integration test: grayscale chained with an existing shader (e.g.,
  brightness then grayscale).

**Key files:**
- `bdip_core/src/gpu/grayscale.wgsl` (new)
- `bdip_core/src/gpu/pipeline.rs` (modify)
- `bdip/src/main.rs` (modify `parse_transform`)
- `bdip/src/ui/sidebar.rs` or `bdip/src/ui/message.rs` (modify — add variant)

**References:**
- `specs/adding_a_shader.md` — the 4-point checklist.
- `specs/transformations_reference.md` — algorithm details.
- Section 7.2 of this document for grayscale shader spec.

**Verification:**
```
$ bdip --headless test.jpg --output out.png --apply grayscale
  (output is visually grayscale)
$ bdip --headless test.jpg --output out.png --apply saturation:-1.0
  vs --apply grayscale  (results should be very similar)
$ cargo test -p bdip_core
$ cargo clippy --workspace
```
- UI: Select "Grayscale" → "Apply" button appears → clicking converts to gray.

---

### PR 8: Invert Shader (Phase 6)

**Goal:** Implement the Invert GPU shader and wire it into both CLI and UI.

**Depends on:** PR 5 (Phase 5 complete). Independent of PRs 6 and 7.

**Scope:**
- Add `invert.wgsl` in `bdip_core/src/gpu/`.
- Note: `Transformation::Invert` already exists in
  `bdip_core/src/transformation.rs` — no changes to that file are needed.
- Add `InvertParams` dummy uniform struct in `pipeline.rs` (same approach as
  Grayscale — exists only to satisfy the bind group layout).
- Add `TransformKind::Invert` variant, `From` impl, `compile()` match arm,
  `apply()` match arm. Because `Transformation::Invert` carries no payload,
  the `apply()` arm must create the params buffer from a zeroed
  `InvertParams` directly: `InvertParams { _unused: [0.0; 4] }`.
- Add `invert` parsing to `parse_transform()` in `bdip/src/main.rs`.
- Add `TransformOption::Invert` to the UI `pick_list`. Parameterless — uses
  the "Apply" button path.
- Unit tests: double invert restores original, preserves alpha, single invert
  produces `1.0 - x` per channel.
- Integration test: invert chained with an existing shader (e.g.,
  brightness then invert).

**Key files:**
- `bdip_core/src/gpu/invert.wgsl` (new)
- `bdip_core/src/gpu/pipeline.rs` (modify)
- `bdip/src/main.rs` (modify `parse_transform`)
- `bdip/src/ui/sidebar.rs` or `bdip/src/ui/message.rs` (modify — add variant)

**References:**
- `specs/adding_a_shader.md` — the 4-point checklist.
- `specs/transformations_reference.md` — algorithm details.
- Section 7.3 of this document for invert shader spec.

**Verification:**
```
$ bdip --headless test.jpg --output out.png --apply invert
$ bdip --headless test.jpg --output out.png --apply invert --apply invert
  (double invert output matches input)
$ cargo test -p bdip_core
$ cargo clippy --workspace
```
- UI: Select "Invert" → "Apply" button appears → clicking inverts colors.
