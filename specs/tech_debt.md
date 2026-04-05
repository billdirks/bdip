# Technical Debt & Refactoring Tracker

This document tracks known architectural shortcuts, generic naming, and structural issues that were accepted during early iterations (Phase 1 & 2) to maintain velocity. 

> [!CAUTION]
> **To Future AI Contexts / Agents:** The remediations proposed below are *suggestions* based on the state of the codebase at the time they were documented. Do NOT execute them blindly. Always cross-reference these suggestions with the *current* state of the code and evaluate if a better, more modern pattern is appropriate before executing refactors.

## Generalization Blockers (GPU Pipeline)

### Generic Parameter Structs
- **Location:** `bdip_core/src/gpu/pipeline.rs`
- **Current Pattern:** The GPU parameter mapping struct is generically named `ParamsUniform`.
- **Refactor Goal:** Prevent a "Monster Struct" pattern where a single giant struct holds unused parameters for *all* possible transformations, which wastes GPU memory alignment space and causes developer confusion.
- **Suggested Remediation:** Rename `ParamsUniform` to `BrightnessUniform` (or `BrightnessParams`). Moving forward, create dedicated, tightly packed, and specifically named parameter structs for *each* discrete transformation (e.g., `ContrastUniform`, `SaturationUniform`).

### Generic Shader Naming
- **Location:** `bdip_core/src/gpu/shader.wgsl`
- **Current Pattern:** The core compute shader file is generically named `shader.wgsl` despite only handling the Brightness algorithm.
- **Refactor Goal:** Avoid confusion and filename collision when managing multiple shaders handling different calculations on the GPU.
- **Suggested Remediation:** Rename `shader.wgsl` to something explicitly descriptive like `brightness.wgsl`. Future shaders should follow this specific isolated naming convention (`contrast.wgsl`, `saturation.wgsl`) to keep the pipeline renderer modular.

### Monolithic Pipeline Initialization
- **Location:** `bdip_core/src/gpu/pipeline.rs` (`Renderer::new`)
- **Current Pattern:** The `Renderer::new` constructor eagerly compiles the shader module and statically generates the `ComputePipeline` directly inside the startup sequence.
- **Refactor Goal:** Prevent a "Monster Initialization" bottleneck. As the application grows to support dozens of transformations, globally compiling all shaders and building all pipelines at startup will aggressively bottleneck application launch times and waste GPU resources for transformations the user may never actually click.
- **Suggested Remediation:** Refactor the `Renderer` (or create a dedicated `PipelineCache`) to implement a **Lazy-Loading** pattern. Shaders and their corresponding pipelines should safely JIT (Just-In-Time) compile upon their first invocation during an `apply_*` call, and then be stored in a HashMap/cache for rapid re-use.

## Future Considerations
*(Add new items here as they are discovered during development)*
