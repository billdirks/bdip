# Fix Invert Transform

## Problem Summary

The shader algorithm is correct. `1.0 - color.rgb` in linear light is the industry standard for
color inversion — confirmed by [Ponies & Light](https://poniesandlight.co.uk/reflect/gamma/) and
[LearnOpenGL](https://learnopengl.com/Advanced-Lighting/Gamma-Correction) — and is consistent with
the pipeline's linear-light architecture. No algorithm or parameter issues exist.

The only issues are in the test suite, which violates the project's single-behavior unit test rule
(AGENTS.md).

### Moderate Issues

1. **`test_invert_shader` combines two behaviors** (line 59–87 in `mod.rs`): it checks RGB channel
   inversion AND alpha preservation in a single test. Per AGENTS.md, each test must cover one
   isolated behavior.

2. **Blue channel output is never asserted** in any direct inversion test: the input image has
   `B=32767`, expected output is `~32768`, but no assertion is made on `pixel[2]`. This leaves a
   gap — a bug affecting only the B channel would not be caught.

### Minor Issues

3. **`test_invert_registry_metadata`** (line 44–48) checks two attributes (`display_name` and
   `param`) in one test. These are logically distinct registry properties and should be split to
   match the single-behavior standard.

---

## Implementation Plan

### PR 1: Split and Complete Invert Tests

**Goal**: Replace `test_invert_shader` and `test_invert_registry_metadata` with granular
single-behavior tests, and add missing coverage for the B channel and alpha isolation.

**Scope**:
- `bdip_core/src/gpu/shaders/invert/mod.rs` — test section only

**Changes**:

Remove `test_invert_shader` and `test_invert_registry_metadata`. Replace with:

1. `test_invert_display_name` — checks `reg.meta.display_name == "Invert"`
2. `test_invert_param_kind` — checks `reg.meta.param == ParamKind::Toggle`
3. `test_invert_black_channel_becomes_white` — input `R=0` → output `R≈65535`
4. `test_invert_white_channel_becomes_black` — input `G=65535` → output `G≈0`
5. `test_invert_midtone_channel_inverts` — input `B=32767` → output `B≈32768`
6. `test_invert_alpha_preserved` — input `A=65535` → output `A=65535`

Existing tests to keep unchanged:
- `test_invert_registry_entry_exists`
- `test_invert_make_uniform_known_value`
- `test_double_invert_restores_original`

**Implementation Notes**:

Each GPU-backed test needs its own `GpuEngine` and `Renderer` instance. Use a helper to reduce
boilerplate:

```rust
fn run_invert(engine: &GpuEngine, renderer: &mut Renderer, r: u16, g: u16, b: u16)
    -> image::Rgba16Image
{
    let img = make_solid_image(2, 2, r, g, b);
    roundtrip(renderer, engine, &img, &[Transform { shader_id: "invert", values: vec![] }])
}
```

---

## Test Specifications

### `test_invert_display_name`

```rust
/// Registry display name is "Invert".
#[test]
fn test_invert_display_name() {
    let reg = registry_by_id("invert").unwrap();
    assert_eq!(reg.meta.display_name, "Invert");
}
```

### `test_invert_param_kind`

```rust
/// Registry param kind is Toggle (no sliders).
#[test]
fn test_invert_param_kind() {
    let reg = registry_by_id("invert").unwrap();
    assert_eq!(reg.meta.param, ParamKind::Toggle);
}
```

### `test_invert_black_channel_becomes_white`

```rust
/// A fully black input channel (0) inverts to fully white (65535).
#[test]
fn test_invert_black_channel_becomes_white() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    let out = run_invert(&engine, &mut renderer, 0, 32767, 32767);
    let r = out.get_pixel(0, 0)[0];
    assert!((r as i32 - 65535).abs() <= 100, "R: expected ~65535, got {r}");
}
```

### `test_invert_white_channel_becomes_black`

```rust
/// A fully white input channel (65535) inverts to fully black (0).
#[test]
fn test_invert_white_channel_becomes_black() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    let out = run_invert(&engine, &mut renderer, 32767, 65535, 32767);
    let g = out.get_pixel(0, 0)[1];
    assert!(g <= 100, "G: expected ~0, got {g}");
}
```

### `test_invert_midtone_channel_inverts`

```rust
/// A midtone sRGB input (32767 ≈ 0.5 sRGB ≈ 0.214 linear) inverts in linear space
/// to 0.786 linear ≈ 0.899 sRGB ≈ 58922 u16 after the sRGB round-trip.
#[test]
fn test_invert_midtone_channel_inverts() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    let out = run_invert(&engine, &mut renderer, 32767, 32767, 32767);
    let b = out.get_pixel(0, 0)[2];
    // 32767 ≈ 0.5 sRGB ≈ 0.214 linear; inverted → 0.786 linear ≈ 0.899 sRGB ≈ 58922 u16.
    assert!((b as i32 - 58922).abs() <= 300, "B: expected ~58922, got {b}");
}
```

### `test_invert_alpha_preserved`

```rust
/// Alpha channel is not inverted — it passes through unchanged.
#[test]
fn test_invert_alpha_preserved() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);
    let out = run_invert(&engine, &mut renderer, 0, 0, 0);
    assert_eq!(out.get_pixel(0, 0)[3], 65535, "alpha must be preserved");
}
```

---

## Validation Checklist

After PR 1 is merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test -p bdip_core invert` passes (all 9 tests)
- [ ] `cargo fmt --all` reports no changes needed

---

## References

- [Ponies & Light — Notes on Gamma](https://poniesandlight.co.uk/reflect/gamma/)
- [LearnOpenGL — Gamma Correction](https://learnopengl.com/Advanced-Lighting/Gamma-Correction)
- [Prolost — Linear Light, Gamma, and ACES](https://prolost.com/blog/aces)
