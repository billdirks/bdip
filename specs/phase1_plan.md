# Phase 1: Core Library Foundation — Implementation Plan

This plan covers the first phase of the bdip implementation as defined in [specification.md](./specification.md) §5.1. The goal is to establish the Cargo workspace, define all core data models, and implement image I/O — with no GPU or UI code.

## Scope

**In scope:** Workspace structure, `Transformation` enum, `BdipError` type, `HistoryManager`, image load/save via the `image` crate.

**Out of scope:** `wgpu` GPU pipeline, WGSL shaders, `iced` UI, `clap` CLI parsing. These belong to Phase 2+.

---

## 1. Workspace Scaffolding

### [NEW] Cargo.toml (workspace root)

Create a root workspace manifest at `/Users/bdirks/Projects/ImageProcessingApp/Cargo.toml`:

```toml
[workspace]
members = ["bdip_core", "bdip"]
resolver = "2"
```

### [NEW] bdip_core/Cargo.toml

```toml
[package]
name = "bdip_core"
version = "0.1.0"
edition = "2024"

[dependencies]
image = "0.25"
thiserror = "2"
```

> [!NOTE]
> `wgpu` is intentionally excluded from Phase 1. It will be added in Phase 2 when the GPU pipeline is built. This keeps Phase 1 compilable and testable without GPU hardware concerns.

### [NEW] bdip/Cargo.toml

```toml
[package]
name = "bdip"
version = "0.1.0"
edition = "2024"

[dependencies]
bdip_core = { path = "../bdip_core" }
anyhow = "1"
```

> [!NOTE]
> `clap`, `iced`, and `rfd` are deferred to later phases. Phase 1's binary is a minimal skeleton that imports `bdip_core` and compiles.

---

## 2. Error Types

### [NEW] bdip_core/src/error.rs

Define `BdipError` using `thiserror`. Phase 1 variants cover I/O and format errors. GPU variants will be added in Phase 2.

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BdipError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image decoding/encoding error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Invalid transformation parameter: {0}")]
    InvalidParameter(String),
}
```

---

## 3. Transformation Enum

### [NEW] bdip_core/src/transformation.rs

Defines the V1 transformation variants per [specification.md](./specification.md) §6. All `f32` parameters use normalized ranges (`-1.0` to `1.0`), where `0.0` means no change.

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Transformation {
    Brightness(f32),    // -1.0 (full dark) to 1.0 (full bright)
    Contrast(f32),      // -1.0 (flat gray) to 1.0 (max contrast)
    Saturation(f32),    // -1.0 (grayscale) to 1.0 (max saturation)
    Grayscale,          // No parameters — converts to luminance
    Invert,             // No parameters — inverts all channels
}
```

> [!IMPORTANT]
> **Design decision: `0.0` = identity.** For all parameterized V1 transforms, a value of `0.0` produces no change to the image. This is critical for slider UX (slider centered = no effect) and for the history system (avoids storing no-ops).

---

## 4. History Manager

### [NEW] bdip_core/src/history.rs

Implements the undo/redo buffer system per [specification.md](./specification.md) §4.3.

```rust
use crate::transformation::Transformation;

pub struct HistoryManager {
    undo_stack: Vec<Transformation>,
    redo_stack: Vec<Transformation>,
}

impl Default for HistoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl HistoryManager {
    pub fn new() -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }
    // ... other methods ...
}
```

**Public API:**

| Method | Behavior |
|---|---|
| `apply(t: Transformation)` | Pushes `t` onto `undo_stack`, clears `redo_stack` |
| `undo() -> Option<()>` | Pops from `undo_stack`, pushes onto `redo_stack`. Returns `None` if nothing to undo. |
| `redo() -> Option<()>` | Pops from `redo_stack`, pushes onto `undo_stack`. Returns `None` if nothing to redo. |
| `active_transforms() -> &[Transformation]` | Returns the current `undo_stack` — the list of transforms to replay from the base image. |
| `clear()` | Resets both stacks. Used when loading a new image. |

---

## 5. Image I/O

### [NEW] bdip_core/src/io.rs

Handles loading images from disk into raw pixel buffers and saving buffers back to disk.

**Loading:**
- Accept a file path, use the `image` crate to decode
- Format detection is automatic via the `image` crate's `ImageReader`
- Convert to and return `image::RgbaImage` (which is an alias for `ImageBuffer<Rgba<u8>, Vec<u8>>`). This explicit type makes it easy to extract width, height, and raw bytes later for GPU upload.

**Saving:**
- Accept an `&image::RgbaImage` and an output path
- Determine output format from the file extension
- Encode and write using the `image` crate
- Return `BdipError::UnsupportedFormat` for unrecognized extensions

> [!NOTE]
> **Why `Rgba8` and not `Rgba16Float`?** The spec calls for `Rgba16Float` as the *internal GPU texture format*. At this phase (no GPU), we load into the `image` crate's native `Rgba8` buffer. In Phase 2, this buffer will be uploaded to the GPU and converted to `Rgba16Float` during texture creation. The I/O layer always works with on-disk formats (8-bit sRGB), and the GPU layer works with `Rgba16Float` linear. This is the correct boundary.

---

## 6. Library Entry Point

### [NEW] bdip_core/src/lib.rs

Module declarations and public re-exports:

```rust
pub mod error;
pub mod history;
pub mod io;
pub mod transformation;

pub use error::BdipError;
pub use history::HistoryManager;
pub use transformation::Transformation;
```

---

## 7. Binary Skeleton

### [NEW] bdip/src/main.rs

Minimal binary that proves the workspace compiles and the library is importable:

```rust
use bdip_core::{BdipError, HistoryManager, Transformation};

fn main() -> anyhow::Result<()> {
    println!("bdip v0.1.0");
    Ok(())
}
```

---

## Verification Plan

### Automated Tests

Run via `cargo test -p bdip_core`:

1. **I/O round-trip** (`io.rs`):
   - Load a PNG test image → verify dimensions and pixel count
   - Save to a new PNG → reload → verify pixels match
   - Load a JPG → save as JPG → reload → verify dimensions match (lossy, so pixel-exact match not expected)
   - Attempt to load a nonexistent file → verify `BdipError::Io` is returned
   - Attempt to save with an unsupported extension (e.g., `.bmp`) → verify `BdipError::UnsupportedFormat`

2. **History** (`history.rs`):
   - `apply` 3 transforms → verify `active_transforms()` returns all 3 in order
   - `undo` → verify stack has 2 transforms, redo stack has 1
   - `redo` → verify stack restored to 3
   - `undo` then `apply` new transform → verify redo stack is cleared
   - `undo` on empty stack → returns `None`

3. **Transformation** (`transformation.rs`):
   - Verify `Clone` and `PartialEq` work correctly
   - Verify enum variants construct with expected values

### Build Check

- `cargo check --workspace`: Full workspace compiles with no errors or warnings.
- `cargo build -p bdip`: Binary crate builds and the `bdip_core` dependency resolves.

> [!NOTE]
> Ensure your local `rustc` toolchain is fully updated to support the Rust 2024 edition specified in the `Cargo.toml` files.

### Test Assets

Test images are generated programmatically in test setup code using the `image` crate (e.g., a 64×64 gradient). No committed fixture files needed.
