# Image Processing Application Specification (Rust)

## 1. Overview
The Image Processing Application is a high-performance, GPU-accelerated tool built completely in Rust. Engineered for macOS (leveraging WebGPU/Metal), it provides a snappy, modern desktop UI that can evolve rapidly. Crucially, the application is designed with strict separation of concerns, operating seamlessly as either an interactive desktop app or a headless command-line interface (CLI). The underlying image processing engine will be isolated as a standalone, consumable Rust library, empowering other external Rust programs to leverage its powerful transformation pipelines completely free from UI overhead.

## 2. Architecture & Design Principles (Library vs. Binary)
To guarantee the user interface is completely decoupled from the image processing operations, the project will be structured fundamentally as a Core Library (`lib.rs`) and an Application Binary (`main.rs`).

1. **Core Processing Library (`lib`)**:
   * **GPU Transformation Engine**: Encapsulates all `wgpu` (WebGPU) compute/render logic. It consumes a base image buffer and a list of declarative `Transformation` instructions, processes them on the GPU, and outputs a formatted texture. It has strictly zero knowledge of `iced` or windowing systems.
   * **State Data Models**: Owns the standardized data structures dictating transformation parameters and `Undo`/`Redo` histories.
   * **I/O Manager**: Focuses comprehensively on serialization/deserialization for standard formats (`JPG`, `GIF`, `TIFF`) using the `image` crate.
   * *Extensibility*: Other Rust projects can import this crate and use its pipeline freely.
   
2. **Application Binary (`bin`)**:
   * **CLI Controller**: Uses the `clap` crate to parse startup arguments. It determines whether to boot the `iced` windowing context or execute purely headless logic.
   * **UI Layer**: Built using `iced`. It translates interactive UI states (slider drags, clicks) into commands passed directly to the Core Processing Library.

## 3. Technology Stack & Evaluation

### Core Processing Stack
* **Graphics API**: `wgpu`. The Rust implementation of the WebGPU standard. Natively targets Metal on macOS. Provides unparalleled performance for headless compute tasks and interactive rendering alike.
* **Media Handling**: `image` crate. A highly optimized pure-Rust library for seamless decoding and encoding.
* **CLI Parser**: `clap`. The robust standard for parsing terminal arguments securely.

### UI Framework Evaluation (The Decision: `iced`)
A top-priority requirement is that the UI must be modern, beautiful, and extremely flexible for dramatic layout redesigns. `iced` (Declarative / Elm Architecture) is chosen over `egui` (Immediate Mode) or Tauri (Webview). 

Because it is declarative, `iced` accommodates sweeping visual changes and advanced styling cleanly. Most importantly, `iced` renders using `wgpu` internally. This allows the Core Library's `wgpu` pipeline and the UI window device to natively share memory, processing everything in a unified, instantly "snappy" environment with absolutely no Swift/C++ shims or IPC barriers.

## 4. Key Workflows & Features

### 4.1. Command-Line Interface (CLI) & Headless Execution
By offloading rendering and routing to the Core Library, the CLI supports dual run-modes:
* **Interactive UI with Asset Preloading**: Running `$ app_name /path/to/img.jpg` launches the UI immediately, bypassing file pickers to load the respective file smoothly into the visual canvas.
* **Headless Batch Mode**: To ensure that the command line can safely handle order-dependent arrays of image transformations seamlessly, the CLI runs a hybrid parser approach:
   * **Ordered Multivalue Arguments**: Operating from the terminal directly uses repeatable flags. Example: `$ app_name --headless in.jpg --output out.jpg --apply brightness:0.5 --apply blur:5.0`. The `clap` parser evaluates these left-to-right safely. 
   * **Pipeline Manifest Files**: Executing `$ app_name --headless --input in.jpg --output out.jpg --pipeline script.txt` references an external file for repeatable pipelines. This file cleanly holds a newline-separated list of the same underlying `--apply` arguments. The CLI binary iterates these lines directly as isolated parameters handed down to the transformation library.

### 4.2. File I/O
* **Open**: UI file pickers (`rfd` crate) or CLI arguments pass local file strings to the Core Library. The `image` crate deserializes raw buffers into a unified `wgpu` layout map.
* **Save**: The rendering engine extracts the final processed texture buffer from the GPU, routing it safely out as `JPG`/`TIFF`.

### 4.3. History Buffer System (Undo/Redo)
Transformations are stored as lightweight standardized structs (e.g., `Brightness(0.5)`). The base image is kept pristine.
* **Undo/Redo Operation**: Safely pops actions between an `undo_buffer` and `redo_buffer` and requests the Core Library to repaint the GPU buffer based on the active array list. 
* **New Changes**: Applying a new transformation dynamically purges the `redo_buffer` entirely, and pushes the newly requested action onto the `undo_buffer`.

## 5. Iterative Implementation Plan

1. **Library Scaffolding & I/O Phase**:
   * Format Cargo workspace with `core_lib` and `app_bin`.
   * Apply `clap` argument parsing resolving `--input`, multiple `--apply` vectors, and `--pipeline` iterators.
   * Implement decoding images into base buffers via `image` running specifically inside `core_lib`.
2. **Headless Engine & Processing Phase**:
   * Build the standalone `wgpu` execution logic in the `core_lib` (fully detached from window contexts).
   * Achieve proof-of-concept for the Headless CLI: Read a file, apply pipeline transformations based on the list, and write via `wgpu`. 
3. **UI Integration Phase**:
   * Boot the `iced` event lifecycle in `app_bin`.
   * Share the initialized `wgpu` adapter context between `core_lib` and `iced`. Display the output texture into a primary application viewport seamlessly.
4. **Toolbar & Interactive Enhancements**:
   * Construct `iced` right-hand layout sidebar modules. 
   * Formally introduce the Brightness shader in `core_lib` (mapped similarly to Headless capability).
   * Map `iced` slider feedback messages directly to pipeline alterations.
5. **History & Buffer Execution**:
   * Refine History Structs tracking transformations independently.
   * Bind `iced` keystrokes into queue stack array mutators correctly.

**Future Considerations (Selection and Extensibility)**:
When dramatic UI layouts are requested, only the UI layer in `app_bin` requires modification, while the transformation logic inside `core_lib` remains perfectly sealed and safe. Building Region-Selection tools natively fits into this structure; a user selection merely provides an alpha-channel Mask texture to the Core Library. Because of the headless/binary separation constraints, this capability to apply transformations strictly against a mask will inherently be supported visually in the UI and pragmatically across the Command Line (e.g. passing a bounding box logic payload to the batch processor).

## 6. Addendum: Image Transformations Reference
The Core Library's stateless transformation pipeline will natively support various algorithms mapped directly to WGSL compute shaders on the GPU. Example transformations include:
* **Basic Color Adjustments**: Brightness, Contrast, Saturation, Vibrance
* **Conversions**: Black & White (Grayscale), Sepia, Invert
* **Spatial Filters**: Gaussian Blur, Unsharp Mask (Sharpening), Edge Detection

For a detailed breakdown of the internal mathematics and operational mechanics driving these effects, please refer to the adjoining document: [transformations_reference.md](./transformations_reference.md).

## 7. Addendum: Visual Architecture & Component Flow
To explicitly guide implementation, the absolute dependency boundaries separating the Application Binary and the Core Library, along with their nested internal module workflows, have been functionally diagrammed. 

For the complete visualization of how user interactions, I/O states, and GPU data execution strictly flow, please refer to the graphic documentation: [architecture_diagram.md](./architecture_diagram.md).
