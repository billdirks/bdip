---
name: audit-transform
description: >
  Audit an existing GPU shader transform for correctness. Researches what the transform should do,
  evaluates the implementation against that research, and writes up findings with a fix plan.
  Use when you want to verify a shader is implemented correctly or discover issues.
  Example: "/audit-transform halftone_dots"
user-invocable: true
---

# /audit-transform — Audit a GPU Shader Transform for Correctness

Audit an existing shader transform by researching domain requirements and evaluating the
implementation. Produce a fix plan if issues are found.

## Transform to Audit

$ARGUMENTS

The transform is located at: `bdip_core/src/gpu/shaders/$ARGUMENTS/`

---

## Pipeline Context

Before auditing, understand the image pipeline architecture:

### Color Space
All transform shaders operate in **linear light**. The pipeline handles color space conversion:
- **Ingest** (`ingest.wgsl`): Converts uploaded sRGB → linear on GPU
- **Transform passes**: All math runs in linear space — shaders should NOT apply gamma correction
- **Present** (`presentation.wgsl`): Converts linear → sRGB for output

If a shader appears to be "missing gamma correction," it's likely correct — the pipeline handles it.

### Texture Format & Value Range
Intermediate textures use `Rgba16Float` format. Expected value range is [0.0, 1.0] for standard
images, though HDR values >1.0 are preserved through the pipeline.

### Multi-Pass Infrastructure
Transforms can define multiple passes with scratch textures:
- `PassInput::Source` — reads the original input texture
- `PassInput::Scratch("name")` — reads a named scratch texture from a prior pass
- `PassOutput::Final` — writes to the final output texture
- `PassOutput::Scratch("name")` — writes to a named scratch texture
- `PassScale::Full` — output at full resolution
- `PassScale::Down(n)` — output at 1/n resolution (e.g., `Down(4)` for quarter-size blurs)

Scratch textures are pooled and reused across passes and across shader invocations.

### Bind Group Layout
Shaders use a fixed bind group structure:
- **Group 0**: Texture bindings — inputs at bindings 0..n-1, output storage texture at binding n
- **Group 1**: Parameters uniform buffer at binding 0
- **Group 2** (if aux textures): Paired texture+sampler bindings (texture at 2i, sampler at 2i+1)

---

## Step 1 — Read the Transform Implementation

Read all files in the transform directory:
- `mod.rs` — Rust module with parameters, slider definitions, pass configuration
- `*.wgsl` — WGSL shader source(s)

Note the following:
- Parameter names, ranges, defaults, and descriptions
- Number of passes and their order
- Algorithm used in each shader
- Any auxiliary textures referenced

---

## Step 2 — Research What This Transform Should Do

Use web search to research the domain:
- What is this effect and how does it work technically?
- What are the standard algorithms for implementing it?
- What parameters typically control this effect?
- What are the correct parameter ranges?
- Are there industry-standard approaches or academic references?

Search for authoritative sources: Wikipedia, academic papers, shader tutorials, image
processing textbooks. Aim for 3-5 quality sources.

---

## Step 3 — Evaluate the Implementation

Compare the implementation against your research. Evaluate:

### Algorithm Correctness
- Is the mathematical approach correct for this effect?
- Are the passes in the correct order?
- Is each shader implementing the right algorithm?

### Parameter Correctness
- Are the right terms parameterized?
- Are parameter ranges appropriate?
- Are defaults sensible?
- Are any important parameters missing?

### Asset Quality (if applicable)
- Check `bdip_core/src/gpu/assets/` for any auxiliary textures
- Are textures production quality (resolution, accuracy)?
- Note if registered assets are not used by this shader (they may be used elsewhere)
- **NEVER recommend deleting assets** — assets may be shared across shaders

### Code Quality
- Is the shader efficient?
- Is there unnecessary complexity?
- Are there any bugs or edge cases?

---

## Step 4 — Write Findings to Spec File

First, read `bdip_core/shaders-wip/specs/fix-halftone-dots-plan.md` as an example of the expected
format and level of detail. Then create the file 
`bdip_core/shaders-wip/specs/fix-$ARGUMENTS-plan.md` with:

### 1. Problem Summary
- List all issues found (critical, moderate, minor)
- Reference specific lines and explain what's wrong
- Cite research sources that inform your assessment

### 2. Implementation Plan
Break fixes into PRs that can each be implemented by a fresh Sonnet instance with the prompt:
"Implement PR X from @bdip_core/shaders-wip/specs/fix-$ARGUMENTS-plan.md"

Each PR should include:
- **Goal**: One sentence describing the objective
- **Scope**: Specific files and changes required
- **Implementation details**: Pseudocode or algorithm description where helpful
- **Tests to add**: Specific test cases with names and assertions

### 3. Test Specifications
For each new test, provide:
- Test function name
- What behavior it verifies
- Setup required
- Assertions to make
- Example code skeleton

### 4. Validation Checklist
- List of checks to run after all PRs are merged
- Commands to verify correctness

---

## Step 5 — Report

If **no issues** were found:
- Report that the transform passes audit
- Note any minor suggestions for improvement (optional)
- Do NOT create a fix plan file

If **issues were found**:
- Summarize the critical issues
- Report the path to the fix plan file
- List the PRs in the plan
