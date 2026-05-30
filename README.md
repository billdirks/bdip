# bdip - High-performance image transformations

A performance minded library and application for applying gpu-based image shaders. There are 2
major goals:

1. This project aims to be as fast as possible and competitive with commercial implementations.
   I'm specifically targeting 24MP images on Apple silicon.
2. It should be easy to add shaders. This should be doable without any real understanding of
   Rust. For more information [see below](#adding-a-shader).

## Architecture

This project is broken up into 3 crates: a core library and 2 applications which consume it.

- **`bdip-core`**: The core library responsible for all image transformation logic and file I/O.
  It can be imported by any Rust application that wants to apply image transformations.
- **`bdip`**: A desktop GUI application for interactively editing images.
- **`bdip-cli`**: A CLI applicaiton that can be used for batch processing. It depends only on 
`bdip-core` and does not pull in any GUI dependencies (`iced`, `rfd`).

## Quickstart

The cargo aliases are helpful and can be found in `.cargo/config.toml`. I highlight some here.

### Run the UI

```bash
cargo ui-release
```

If you decide on a set of transformations you like in the UI you can export them (`Export pipeline` in the `File` pulldown) and pass that file to `bdip-cli` to apply it to images in a pipeline.

The command also takes an image path to be loaded:

```bash
cargo ui-release path/to/your/image.png
```

You can replace `ui-release` with `ui` for a debug build. It will be noticeably slower.

### Run the CLI

Use `cargo cli` to run the headless app. You can chain `--apply` flags to apply
multiple transforms in order (left to right), or pass a pipeline file. Examples:

```bash
cargo cli input.tif --output out.png \
    --apply 'abstract_geometry:0.37:18.5:0.2:0.51' \
    --apply 'fisheye:-0.19'
```

```bash
cargo cli input.tif --output out.png --pipeline pipeline.txt
```

where `pipeline.txt` is:

```
abstract_geometry:0.37:18.5:0.20:0.51
fisheye:-0.19
```

```bash
cargo cli --help
```

#### Shader discovery

```bash
# List all available shaders
cargo cli --list-shaders

# Get detailed help for a specific shader
cargo cli --describe-shader <shader_id>
```

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

**5. CLI performance diagnostics**
```bash
cargo cli input.tif --output out.png --pipeline pipeline.txt --timings
```

## Adding a Shader

All shader implementations are found in
[this directory](./bdip_core/src/gpu/shaders). Auxiliary image assets are stored
[here](./bdip_core/src/gpu/assets/) and can be shared between shaders. A how-to-write-a-shader
doc, which can also be used by AI, can be found [here](./specs/adding_a_shader.md).

## Specification

The [original specification](./specs/specification.md) and supporting
[architectural diagram](./specs/architecture_diagram.md) along with a description of the
[execution model](./specs/execution_model.md) are found in the [specs directory](./specs/).
