# bdip — Image Processing Application Specification (Rust)

## 1. Overview
The Image Processing Application is a high-performance, GPU-accelerated tool built completely in Rust. Engineered for macOS (leveraging WebGPU/Metal), it provides a snappy, modern desktop UI that can evolve rapidly. Crucially, the application is designed with strict separation of concerns, operating seamlessly as either an interactive desktop app or a headless command-line interface (CLI). The underlying image processing engine will be isolated as a standalone, consumable Rust library, empowering other external Rust programs to leverage its powerful transformation pipelines completely free from UI overhead.

## 2. Architecture & Design Principles (Library vs. Binary)
To guarantee the user interface is completely decoupled from the image processing operations, the project will be structured fundamentally as a Core Library (`lib.rs`) and an Application Binary (`main.rs`).

The project is structured as a **Cargo workspace** with two independent crates (`bdip_core` and `bdip`). This ensures the core library has zero dependency on UI crates (`iced`, `clap`, `rfd`) and can be consumed by external Rust projects purely as an image processing engine.

1. **Core Processing Library (`bdip_core` crate)**:
   * **GPU Transformation Engine**: Encapsulates all `wgpu` (WebGPU) compute/render logic. It consumes a base image buffer and a list of declarative `Transformation` instructions, processes them on the GPU, and outputs a formatted texture. It has strictly zero knowledge of `iced` or windowing systems.
   * **State Data Models**: Owns the standardized data structures dictating transformation parameters and `Undo`/`Redo` histories.
   * **I/O Manager**: Focuses comprehensively on serialization/deserialization using the `image` crate. Initially supported formats: `PNG`, `JPG`, `GIF`, and `TIFF`. This list may be expanded in future iterations as needs arise.
   * **Error Types**: Defines a structured error enum using `thiserror` with variants for I/O failures, GPU errors, unsupported formats, and invalid transformation parameters. All public API functions return `Result<T, BdipError>`.
   * *Extensibility*: Other Rust projects can import this crate and use its pipeline freely.
   
2. **Application Binary (`bdip` crate)**:
   * **CLI Controller**: Uses the `clap` crate to parse startup arguments. It determines whether to boot the `iced` windowing context or execute purely headless logic.
   * **UI Layer**: Built using `iced`. It translates interactive UI states (slider drags, clicks) into commands passed directly to the Core Processing Library.
   * **Error Presentation**: Uses `anyhow` to wrap library errors with user-facing context. The CLI reports errors to stderr with non-zero exit codes. The UI surfaces errors via modal dialogs.

## 3. Technology Stack & Evaluation

### Core Processing Stack
* **Graphics API**: `wgpu`. The Rust implementation of the WebGPU standard. Natively targets Metal on macOS. Provides unparalleled performance for headless compute tasks and interactive rendering alike.
* **Media Handling**: `image` crate. A highly optimized pure-Rust library for seamless decoding and encoding of all supported formats.
* **CLI Parser**: `clap`. The robust standard for parsing terminal arguments securely.
* **Error Handling**: `thiserror` in `bdip_core` for structured, matchable error types. `anyhow` in `bdip` for ergonomic error wrapping with user-facing context.

### Internal Pixel Format & Color Space
The engine operates internally on **`Rgba16Float`** textures in **linear color space**. Input images are converted from sRGB gamma space on upload to the GPU; output is converted back to sRGB for display and file export. This 16-bit floating-point format preserves precision across chained transformations (eliminating banding artifacts) and provides headroom above 1.0 for intermediate calculations without clamping. The sRGB↔linear conversion is handled automatically by GPU hardware at zero performance cost.

### UI Framework Evaluation
A top-priority requirement is that the UI must be modern, beautiful, and extremely flexible for dramatic layout redesigns. Declarative frameworks like `iced` (Elm Architecture) or `slint` (Custom DSL) are the primary candidates over immediate mode (`egui`) or webviews (`Tauri`).

To ensure `bdip_core` remains an entirely uncoupled library, the UI binary integrates with the core GPU engine via a **CPU Bridge**. This means that instead of attempting to perfectly align `wgpu` versions to share a single GPU context, the core library processes textures on its own isolated `wgpu` device and downloads the final processed frame into standard CPU host-memory. It then returns a standard `RgbaImage` struct to the UI binary, which the UI framework seamlessly re-uploads to its own rendering surface. On Apple Silicon, this host-memory handoff takes ~1-4ms, rendering the performance cost negligible. This grants complete freedom to choose a UI framework based on state-management and styling capabilities, rather than being forced to match a specific `wgpu` version dependency.

## 4. Key Workflows & Features

### 4.1. Command-Line Interface (CLI) & Headless Execution
By offloading rendering and routing to the Core Library, the CLI supports dual run-modes:
* **Interactive UI with Asset Preloading**: Running `$ bdip /path/to/img.jpg` launches the UI immediately, bypassing file pickers to load the respective file smoothly into the visual canvas.
* **Headless Batch Mode**: To ensure that the command line can safely handle order-dependent arrays of image transformations seamlessly, the CLI runs a hybrid parser approach:
   * **Ordered Multivalue Arguments**: Operating from the terminal directly uses repeatable flags. Example: `$ bdip --headless in.jpg --output out.jpg --apply brightness:0.5 --apply blur:5.0`. The `clap` parser evaluates these left-to-right safely. 
   * **Pipeline Manifest Files**: Executing `$ bdip --headless in.jpg --output out.jpg --pipeline script.txt` references an external file for repeatable pipelines. This file cleanly holds a newline-separated list of the same underlying `--apply` arguments. The CLI binary iterates these lines directly as isolated parameters handed down to the transformation library.

### 4.2. File I/O
* **Open**: UI file pickers (`rfd` crate) or CLI arguments pass local file strings to the Core Library. The `image` crate deserializes raw buffers into a unified `wgpu` layout map.
* **Save**: The rendering engine extracts the final processed texture buffer from the GPU, routing it out in any supported format.

### 4.3. History Buffer System (Undo/Redo)
Transformations are stored as lightweight standardized structs (e.g., `Brightness(0.5)`). The base image is kept pristine.
* **Undo/Redo Operation**: Safely pops actions between an `undo_buffer` and `redo_buffer` and requests the Core Library to repaint the GPU buffer based on the active array list. 
* **New Changes**: Applying a new transformation dynamically purges the `redo_buffer` entirely, and pushes the newly requested action onto the `undo_buffer`.
* **Performance Note**: V1 replays the full transformation stack from the pristine base image on every undo/redo. Intermediate texture caching (checkpointing every N operations) may be introduced as a performance optimization in future iterations if replay latency becomes perceptible with deep stacks of spatial filters.

## 5. Iterative Implementation Plan

1. **Core Library Foundation**:
   * Format Cargo workspace with `bdip_core` (library) and `bdip` (binary).
   * Define the `Transformation` enum, `BdipError` types, and history data structures in `bdip_core`.
   * Implement image decoding/encoding via the `image` crate in `bdip_core`.
   * *Verified when*: Unit tests confirm load → encode round-trip for all supported formats.
2. **GPU Pipeline & Headless CLI**:
   * Initialize headless `wgpu` device/queue in `bdip_core` (no window context).
   * Implement the first WGSL compute shader (Brightness) and the GPU texture pipeline (upload → bind → dispatch → readback).
   * Build the transformation stack and history (undo/redo) model alongside the pipeline, as they are tightly coupled.
   * Wire `clap` argument parsing in `bdip` (positional input path, `--output`, `--apply`, `--pipeline`).
   * *Verified when*: `$ bdip --headless test.jpg --output out.png --apply brightness:0.5` produces a visibly brighter output image.
3. **UI Prototype Spike**:
   * Build a minimal application window in `bdip` using the chosen UI framework (e.g., `iced` or `slint`).
   * Validate the **CPU Bridge** integration strategy: reading an `RgbaImage` from `bdip_core`'s headless pipeline and displaying it in a standard image widget.
   * *Verified when*: A window successfully displays a loaded image with a brightness adjustment applied, validating the framework's ergonomics and rendering pipeline before committing to full UI controls.
4. **Full UI Integration**:
   * Implement file open/save dialogs (`rfd` crate).
   * Construct the sidebar layout with slider controls for each V1 transformation.
   * Wire slider state changes to the transformation pipeline in `bdip_core`.
   * Bind keyboard shortcuts for undo/redo (Cmd+Z / Cmd+Shift+Z).
   * *Verified when*: A user can open an image, adjust brightness/contrast/saturation via sliders, undo/redo changes, and save the result.
5. **V1 Completion & Polish**:
   * Implement remaining V1 shaders: Contrast, Saturation, Grayscale, Invert.
   * End-to-end testing of both CLI and UI workflows.
   * *Verified when*: All V1 transformations work in both headless CLI and interactive UI modes.

**Future Considerations (Selection and Extensibility)**:
When dramatic UI layouts are requested, only the UI layer in `bdip` requires modification, while the transformation logic inside `bdip_core` remains perfectly sealed and safe. Building Region-Selection tools natively fits into this structure; a user selection merely provides an alpha-channel Mask texture to the Core Library. Because of the headless/binary separation constraints, this capability to apply transformations strictly against a mask will inherently be supported visually in the UI and pragmatically across the Command Line (e.g. passing a bounding box logic payload to the batch processor).

## 6. Addendum: Image Transformations Reference
The Core Library's transformation pipeline is driven by a `Transformation` enum. Each variant represents a single filter operation with typed parameters. All parameter values use normalized floating-point ranges (typically `-1.0` to `1.0` or `0.0` to `1.0`) unless otherwise specified.

### V1 Transformations
The following transformations are in scope for the initial release:
* **Basic Color Adjustments**: Brightness (`f32`), Contrast (`f32`), Saturation (`f32`)
* **Conversions**: Black & White (Grayscale) (no parameters), Invert (no parameters)

### V2+ Transformations
The following are planned for future iterations:
* **Basic Color Adjustments**: Vibrance, Exposure, Shadows/Highlights, Temperature/Tint
* **Conversions**: Sepia
* **Spatial Filters**: Gaussian Blur (`radius: u32`, `sigma: f32`), Unsharp Mask (`radius: u32`, `amount: f32`, `threshold: f32`), Edge Detection

For a detailed breakdown of the internal mathematics and operational mechanics driving these effects, please refer to the adjoining document: [transformations_reference.md](./transformations_reference.md).

## 7. Addendum: Visual Architecture & Component Flow
To explicitly guide implementation, the absolute dependency boundaries separating the Application Binary and the Core Library, along with their nested internal module workflows, have been functionally diagrammed. 

For the complete visualization of how user interactions, I/O states, and GPU data execution strictly flow, please refer to the graphic documentation: [architecture_diagram.md](./architecture_diagram.md).
