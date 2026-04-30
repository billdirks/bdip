---
name: create-transform
description: >
  This skill should be used when the user asks to "create a transform", "add a shader",
  "implement a new effect", "add a new filter", or provides a description of a GPU image
  processing effect to implement in this project. It spawns a fresh Sonnet subagent to
  create a new shader in the bdip_core crate following the project conventions.
  Examples: "create a transform for film grain", "add a hue rotation shader",
  "implement a cross-process film effect".
user-invocable: true
allowed-tools:
  - Agent
---

# /create-transform — Spawn Subagent to Implement a New GPU Shader Transform

Spawn a fresh Sonnet subagent with no prior context to implement the requested shader.

Use the Agent tool with the following parameters:

```
Agent({
  description: "Implement shader transform",
  model: "sonnet",
  prompt: <see below>
})
```

---

**Prompt to pass to the Agent:**

You are implementing a new GPU shader transform for an image processing application.

## User Request

$ARGUMENTS

## Step 1 — LUT requirement check (BLOCKING)

Before any implementation work, determine if this transform requires a LUT (Look-Up Table).

**Does this transform require a LUT?** Examples that typically need LUTs:
- Film stock emulation (Kodak Portra, Fuji Velvia, etc.)
- Color grading presets
- Specific "look" recreations
- Tone mapping with non-mathematical curves

**If a LUT is required, can it be generated programmatically?**
- **YES, proceed** if: the LUT represents a mathematically-defined transformation
  (identity LUT, gamma curves, cross-process formulas, HSL rotations, etc.)
- **NO, STOP** if: the LUT requires proprietary/measured data that cannot be derived
  from first principles (specific film stock characteristics, calibrated color profiles,
  artistic presets from external sources)

**If a user-provided LUT is required, STOP IMMEDIATELY:**

```
BLOCKED: This transform requires a production-grade LUT that cannot be generated.

The "<transform name>" effect requires a LUT file because: <reason>

To proceed, please rerun this command and provide a path to a production-grade LUT:
  /create-transform <transform name> --lut /path/to/lut.cube

Alternatively, if you want a placeholder implementation using an identity LUT
(no visual effect until a real LUT is provided), say so explicitly.
```

Only proceed if no LUT is needed OR the LUT can be generated programmatically.

---

## Step 2 — Determine names and create branch

From the user request, extract:

- **Display name**: The human-readable name (e.g., "Film Grain")
- **Shader ID**: The snake_case identifier (e.g., `film_grain`)
- **Module name**: Same as the shader ID
- **Branch name**: Display name lowercased, spaces replaced with `.` (e.g., `film.grain`)

Create the branch: `git checkout -b <branch-name>`

If uncommitted changes exist, stop and report this to the user.

---

## Step 3 — Read project rules and implementation guide

1. Read `AGENTS.md` for workflow constraints (clippy, formatting, goal specs, testing standards)
2. Read `specs/adding_a_shader.md` — this is the canonical implementation guide
3. Check for goal specs: `find specs/ -name '*goal*'` and read any that exist
4. Check for shader-specific specs matching the shader name/ID

**Follow `specs/adding_a_shader.md` exactly for all implementation details.**

---

## Step 4 — Implement, test, lint, format

Follow the steps in `specs/adding_a_shader.md`:
- Create the module directory and files
- Register in `shaders/mod.rs`
- Write granular tests (one behavior per test per `AGENTS.md`)
- Run `cargo clippy --all-targets -- -D warnings` and fix all warnings
- Run `cargo fmt --all`
- Run `cargo test <module_name>` to verify

---

## Step 5 — Commit

Stage all created and modified files and create a commit with the message:

```
Created <display-name> transform
```

where `<display-name>` is the human-readable display name determined in Step 2
(e.g., `Created Film Grain transform`).

Use `git add <file1> <file2> ...` to stage only the relevant files (do not use
`git add -A` or `git add .`). Then commit with the exact message format above.

---

## Step 6 — Report

Report:
- Branch name and shader ID
- Files created/modified
- Test results summary
- Commit hash and message
- Any non-obvious design decisions (parameter ranges, identity values, multi-pass rationale)

---

After spawning the agent, wait for it to complete and relay its final report to the user.
