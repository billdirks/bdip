# bdip — High-Performance Image Processor

A GPU-accelerated image processing application built in Rust, targeting macOS (Metal/WebGPU). The
project's goal is to achieve sub-20ms rendering latency for 24MP images on Apple Silicon by
utilizing floating-point precision on the GPU.

## Architecture

The project is structured as a Cargo workspace:
- **`bdip_core`**: The headless core library. Contains the `wgpu` transformation engine, error
  handling, history logic, and file I/O operations.
- **`bdip`**: The application binary. Serves as both a headless CLI tool for batch processing
  and an `iced`-backed UI window (the declarative frontend).

## Quick Start & Usage

This project uses custom Cargo aliases to streamline development workflows. You can view the
underlying command definitions in `.cargo/config.toml`.

### Running the Application

**Run the UI Prototype (Spike)**
Boots the `iced` window to validate the readback bridge. By default, it generates a test image:
```bash
cargo ui
```
You can also pass a specific image path to test the UI with real data:
```bash
cargo ui -- path/to/your/image.jpg
```

**Run the UI (Performance Testing)**
To test pipeline performance without the latency penalty of Rust's debug mode, use the release
alias. This applies optimizations such as SIMD vectorization to the CPU Bridge:
```bash
cargo ui-release -- path/to/your/image.jpg
```

**Run the CLI (Headless processing)**
Processes an image through the core pipeline without starting the UI. Requires an input path,
`--output`, and at least one `--apply` transformation:
```bash
cargo headless path/to/your/image.jpg --output out.png --apply brightness:0.5
```
You can also chain multiple transformations or use a pipeline file:
```bash
cargo headless input.jpg --output out.png --apply brightness:0.3 --apply brightness:0.2
cargo headless input.jpg --output out.png --pipeline transforms.txt
```

### Development Commands

Run the following commands before finalizing any edits to ensure code quality:

**1. Code Formatting**
Ensure all Rust code is neatly formatted:
```bash
cargo format
```

**2. Static Analysis **
Check for warnings, clippy issues, and unoptimized patterns across the entire workspace:
```bash
cargo lint
```

**3. Testing**
Run the core library unit tests and the end-to-end CLI flow tests:
```bash
cargo test --workspace
```

## Documentation & Performance
`bdip` has strict, predefined requirements targeting the commercial photo editing space. For detailed
latency calculations, architecture diagrams, and system specifications, see the documentation in the
`specs/` directory.
