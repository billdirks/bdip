# bdip - High-performance image transformations

A performance minded library and application for applying gpu-based image shaders. There are 2 major goals:

1. This project aims to be as fast as possible and competitive with commericial implementations. I'm specificially targeting 24MP images on Apple silicon.
2. It should be easy to add shaders. This should be doable without any real understanding of Rust. For more information [see below](#adding-a-shader). 

## Architecture

This project is broken up into 2 crates:
- **`bdip_core`**: The core library that is responsible for all image transformation logic and file I/O. This should be importable by any Rust application that wants to apply image transformations.
- **`bdip`**: An example application. This can be run in 2 ways:
* As a desktop GUI application that allows interactively editing images.
* As a headless CLI tool that can be used in a pipeline for batch processing.

## Quickstart

Cargo aliases can be helpful and are found in `.cargo/config.toml`.

### Run the UI 

```bash
cargo ui-release
```

If you decide on a set of transformations you like in the UI you can export them (`Export pipeline` in the `File` pulldown) and pass that file to the CLI to apply it to images in a pipeline.

The command also takes an image path to be loaded:

```bash
cargo ui-release -- path/to/your/image.png
```

You can replace `ui-release` by `ui` for the debug build. It will be noticeable slower.

### Run the CLI

You can run the app in headless mode. You can string together `--apply` flags to apply multiple transforms (these happen from left to right) or pass in a file. Here are some examples:

```bash
cargo headless input.tif --output out.png --apply 'abstract_geometry:0.37:18.5:0.2:0.51' --apply 'fisheye:-0.19'
```

```
cargo headless input.tif --output out.png --pipeline pipeline.txt
```

where `pipeline.txt` is:

```
abstract_geometry:0.37:18.5:0.20:0.51
fisheye:-0.19
```

**BDIRKS - add --help example**

### Development Commands

**1. Code Formatting**
Ensure all Rust code is neatly formatted:
```bash
cargo format
```

**2. Static Analysis**
Check for warnings, clippy issues, and unoptimized patterns across the entire workspace:
```bash
cargo lint
```

**3. Testing**
Run the core library unit tests and the end-to-end CLI flow tests:
```bash
cargo test --workspace
```

**4. Performance testing**
```bash
cargo perf-test
```

**5. Headless performance diagnostics**
```bash
cargo headless input.tif --output out.png --pipeline pipeline.txt --timings
```

## Adding a Shader

All shader implementations are found in [this directory](./bdip_core/src/gpu/shaders). Auxilary image assets are stored [here](./bdip_core/src/gpu/assets/) and can be shared between shaders. A how to write a shader doc, which can also be used by AI, can be found [here](./specs/adding_a_shader.md).

## Specification

The [original specification](./specs/specification.md) and supporting [architectural diagram](./specs/architecture_diagram.md) along with a description of the [execution model](./specs/execution_model.md) are found in the [specs directory](./specs/).
