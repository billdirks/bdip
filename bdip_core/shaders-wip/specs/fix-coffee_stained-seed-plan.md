# Coffee Stained: Procedural Layout (Seed + Count)

## Overview

The current `coffee_stained` shader has 7 stain rings at hardcoded UV positions with
hardcoded ring radii. This plan replaces those fixed constants with a hash-based
procedural generator controlled by two new parameters:

- **Seed** — a float value that deterministically seeds the RNG; any change produces a
  different stain layout, and the same value always reproduces the same layout.
- **Count** — the number of stain rings to render, from 1 to a maximum of 12.

The hardcoded `CENTRE_*` and `RING_RADIUS_*` constants are eliminated entirely. The
`stain_mask` function is replaced by a loop that generates positions and radii on the
fly from a hash of `(seed, ring_index)`.

This is a **visual breaking change** for users who relied on the default layout. The
default seed (0.0) will produce a different 7-ring layout than the previous hardcoded
positions. This is expected and acceptable.

---

## New Parameters

| Parameter | Type   | Range    | Default | Description                                      |
|-----------|--------|----------|---------|--------------------------------------------------|
| Count     | f32→u32| 1–12     | 7       | Number of stain rings                            |
| Seed      | f32    | 0.0–1.0  | 0.0     | RNG seed; any change produces a new layout       |

---

## Struct Changes

### Rust (`mod.rs`)

Current (16 bytes / 4 f32s):

```rust
pub struct CoffeeStainedParams {
    pub strength:      f32,
    pub ring_width:    f32,
    pub inner_clarity: f32,
    pub _padding:      f32,
}
```

New (32 bytes / 8 f32s):

```rust
pub struct CoffeeStainedParams {
    pub strength:      f32,   // offset 0
    pub ring_width:    f32,   // offset 4
    pub inner_clarity: f32,   // offset 8
    pub count:         f32,   // offset 12  → cast to u32 in shader
    pub seed:          f32,   // offset 16  → scaled to u32 in shader
    pub _padding:      [f32; 3], // offsets 20, 24, 28
}
```

The size increases from 16 to 32 bytes. 32 is a valid multiple of the 16-byte
WGSL uniform alignment requirement.

### WGSL (`coffee_stained.wgsl`)

```wgsl
struct CoffeeStainedParams {
    strength:      f32,
    ring_width:    f32,
    inner_clarity: f32,
    count:         f32,
    seed:          f32,
    _padding0:     f32,
    _padding1:     f32,
    _padding2:     f32,
}
```

---

## WGSL Algorithm Changes

### Remove

All 14 hardcoded constants:

```wgsl
// DELETE all of these:
const CENTRE_0 … CENTRE_6: vec2<f32>
const RING_RADIUS_0 … RING_RADIUS_6: f32
```

The explicit 7-call sum in `stain_mask` is also removed.

### Add: Hash Function

A fast, high-quality 32-bit integer hash. Each call to `hash(x)` returns a
uniform-looking u32 given any u32 input. `hash_f` normalizes to [0, 1):

```wgsl
const MAX_STAINS: u32 = 12u;

fn hash(x: u32) -> u32 {
    var v = x;
    v ^= v >> 17u;
    v  = v * 0xbf324c81u;
    v ^= v >> 11u;
    v  = v * 0x68bcae27u;
    v ^= v >> 13u;
    return v;
}

fn hash_f(x: u32) -> f32 {
    return f32(hash(x)) / 4294967295.0;
}
```

This hash is a two-round variant of the Murmur/Wang family. It produces good
avalanche behavior — a 1-bit change in `x` flips ~50% of output bits — so
adjacent seed values produce visually distinct layouts.

### Add: Position and Radius Generators

Each stain ring `i` draws its centre and radius from three consecutive hash
inputs. The stride of 3 per stain means stain 0 uses slots {0, 1, 2}, stain 1
uses {3, 4, 5}, etc., with no collisions up to `MAX_STAINS = 12`.

```wgsl
fn stain_centre(seed: u32, i: u32) -> vec2<f32> {
    let base = seed + i * 3u;
    // Constrain to [0.05, 0.95] so rings are not clipped at frame edges.
    return vec2<f32>(
        0.05 + hash_f(base + 0u) * 0.90,
        0.05 + hash_f(base + 1u) * 0.90,
    );
}

fn stain_radius(seed: u32, i: u32) -> f32 {
    // Ring radii in [0.08, 0.20] UV — small intimate rings to large sweeping ones.
    return 0.08 + hash_f(seed + i * 3u + 2u) * 0.12;
}
```

### Replace: `stain_mask`

```wgsl
fn stain_mask(uv: vec2<f32>) -> f32 {
    // Convert float seed to u32 — 65536 distinct seed values.
    let seed = u32(params.seed * 65535.0);
    let n    = min(u32(params.count), MAX_STAINS);
    var raw  = 0.0;
    for (var i = 0u; i < n; i++) {
        raw += ring_blob(uv, stain_centre(seed, i), stain_radius(seed, i));
    }
    let clamped = min(raw, 1.0);
    return pow(clamped, 0.6);
}
```

The `ring_blob` function itself is unchanged — it still uses `params.ring_width`
and `params.inner_clarity` for all rings.

---

## Slider Definitions

```rust
const PARAM: ParamKind = ParamKind::Sliders(&[
    SliderDef {
        name: "Strength",
        min: 0.0, max: 1.0, default: 0.0,
        description: "Intensity of the stain effect; 0 is unchanged, 1 is the full \
             coffee-stain look.",
    },
    SliderDef {
        name: "Ring Width",
        min: 0.0, max: 1.0, default: 0.3,
        description: "Thickness of the dark ring edge. Lower values produce thin, \
             defined edges; higher values spread the darkening wider.",
    },
    SliderDef {
        name: "Inner Clarity",
        min: 0.0, max: 1.0, default: 0.7,
        description: "How clear the center of each stain is. 1.0 = center nearly \
             unchanged (realistic ring); 0.0 = center also darkened (filled stain).",
    },
    SliderDef {
        name: "Count",
        min: 1.0, max: 12.0, default: 7.0,
        description: "Number of stain rings. 1 produces a single isolated ring; \
             12 produces a heavily stained surface.",
    },
    SliderDef {
        name: "Seed",
        min: 0.0, max: 1.0, default: 0.0,
        description: "Controls the random placement and size of each stain ring. \
             Any change to this value produces a different layout; the same value \
             always reproduces the same layout.",
    },
]);
```

`from_values`:

```rust
fn from_values(values: &[f32]) -> Self {
    Self {
        strength:      values[0],
        ring_width:    values[1],
        inner_clarity: values[2],
        count:         values[3],
        seed:          values[4],
        _padding:      [0.0; 3],
    }
}
```

---

## Test Strategy

### The Core Problem

Two existing tests pin specific pixel coordinates that were computed from the old
hardcoded stain layout:

- `test_coffee_stained_ring_effect_darker_at_edge` — samples pixel (23, 28) which
  was CENTRE_0 = (0.18, 0.22) in a 128×128 image.
- `test_coffee_stained_inner_clarity_affects_center` — also samples (23, 28).

With procedural generation, the stain at seed=0, index=0 lands at a
hash-determined position that is not (0.18, 0.22). Both tests will fail if left
unchanged.

### Solution: Mirror the Hash in Rust for Tests

Add a `#[cfg(test)]` helper module that implements the same hash and position
derivation as the WGSL shader. This lets tests compute expected stain positions
without hard-coding layout assumptions:

```rust
#[cfg(test)]
mod stain_layout {
    fn hash(mut v: u32) -> u32 {
        v ^= v >> 17;
        v  = v.wrapping_mul(0xbf324c81);
        v ^= v >> 11;
        v  = v.wrapping_mul(0x68bcae27);
        v ^= v >> 13;
        v
    }

    fn hash_f(v: u32) -> f32 {
        hash(v) as f32 / u32::MAX as f32
    }

    pub fn centre(seed_f: f32, i: u32) -> (f32, f32) {
        let seed = (seed_f * 65535.0) as u32;
        let base = seed + i * 3;
        (
            0.05 + hash_f(base)     * 0.90,
            0.05 + hash_f(base + 1) * 0.90,
        )
    }

    pub fn radius(seed_f: f32, i: u32) -> f32 {
        let seed = (seed_f * 65535.0) as u32;
        0.08 + hash_f(seed + i * 3 + 2) * 0.12
    }
}
```

The helper is kept in-file (`mod.rs`) inside `#[cfg(test)]`. Keeping the
WGSL hash and the Rust mirror identical is a maintenance requirement — if the
WGSL hash changes, the Rust mirror must change too. A comment in both locations
should call this out.

### Rewriting Position-Sensitive Tests

With the mirror in hand, the two failing tests become:

**`test_coffee_stained_ring_effect_darker_at_edge`** — compute the position of
stain 0 at seed=0.0 from `stain_layout::centre`, convert to pixel coords in
128×128, then sample the center pixel and the ring-edge pixel at
`stain_layout::radius` distance. The assertion `edge_R < center_R` stays the same.

**`test_coffee_stained_inner_clarity_affects_center`** — same: compute stain 0
center at seed=0.0, sample that pixel at inner_clarity=1.0 vs inner_clarity=0.0,
assert the high-clarity center is lighter.

---

## Existing Tests — Required Updates

All GPU roundtrip tests currently pass 3 values; they need to pass 5:

| Test | Old values | New values |
|------|-----------|------------|
| `test_coffee_stained_zero_strength_is_identity` | `[0.0, 0.3, 0.7]` | `[0.0, 0.3, 0.7, 7.0, 0.0]` |
| `test_coffee_stained_full_strength_warms_image` | `[1.0, 0.3, 0.7]` | `[1.0, 0.3, 0.7, 7.0, 0.0]` |
| `test_coffee_stained_full_strength_darkens_image` | `[1.0, 0.3, 0.7]` | `[1.0, 0.3, 0.7, 7.0, 0.0]` |
| `test_coffee_stained_alpha_preserved` | `[1.0, 0.3, 0.7]` | `[1.0, 0.3, 0.7, 7.0, 0.0]` |
| `test_coffee_stained_deterministic` | `[0.8, 0.3, 0.7]` | `[0.8, 0.3, 0.7, 7.0, 0.0]` |
| `test_coffee_stained_chaining_with_brightness` | `[0.5, 0.3, 0.7]` | `[0.5, 0.3, 0.7, 7.0, 0.0]` |
| `test_coffee_stained_ring_width_affects_edge_thickness` | thin `[1.0, 0.1, 0.8]` wide `[1.0, 0.5, 0.8]` | add `7.0, 0.0` to both |
| `test_coffee_stained_ring_effect_darker_at_edge` | hardcoded pixel (23, 28) | rewrite using `stain_layout` mirror |
| `test_coffee_stained_inner_clarity_affects_center` | hardcoded pixel (23, 28) | rewrite using `stain_layout` mirror |

Additionally:

- `test_coffee_stained_registry_metadata` — extend the expected `ParamKind::Sliders`
  array from 3 to 5 sliders.
- `test_coffee_stained_make_uniform_known_value` — update to pass 5 values and check
  the new 32-byte struct layout.

---

## New Tests

### `test_coffee_stained_different_seeds_produce_different_output`

```rust
/// Different seed values must produce visually distinct stain layouts.
#[test]
fn test_coffee_stained_different_seeds_produce_different_output() { … }
```

Render the same white image twice with `seed=0.0` and `seed=0.5`, same
strength/ring_width/inner_clarity/count. Assert at least one pixel differs between
the two outputs.

---

### `test_coffee_stained_same_seed_reproduces_layout`

```rust
/// The same seed must always produce the identical layout (determinism per seed).
#[test]
fn test_coffee_stained_same_seed_reproduces_layout() { … }
```

Render with `seed=0.42` twice; assert every pixel is identical. (Extends the
existing `test_coffee_stained_deterministic` concept to a non-zero seed.)

---

### `test_coffee_stained_count_one_fewer_stains_than_default`

```rust
/// Reducing count from the default should produce fewer darkened pixels.
#[test]
fn test_coffee_stained_count_one_fewer_stains_than_default() { … }
```

Render the same white 64×64 image at count=3 and count=7, same seed.
Count pixels below a darkening threshold. Assert count=7 darkens more pixels
than count=3.

---

### `test_coffee_stained_count_one_produces_single_ring`

```rust
/// Count=1 must produce at least one darkened pixel on a white image.
/// Verifies that the minimum-count edge case renders something visible.
#[test]
fn test_coffee_stained_count_one_produces_single_ring() { … }
```

Render a 128×128 white image at count=1, seed=0.0, strength=1.0. Assert at
least one pixel has R below 90% of white (i.e., is visibly darkened).

---

## Validation Checklist

After the PR is merged:

- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes (all 15 coffee_stained tests)
- [ ] `cargo fmt --all` reports no changes needed
- [ ] Dragging the Seed slider visibly changes stain positions
- [ ] The same Seed value always produces the same layout (manual verification)
- [ ] Setting Count=1 shows a single ring
- [ ] Setting Count=12 shows a heavily covered image
- [ ] strength=0 remains identity (no stains visible)
- [ ] Alpha channel is preserved
- [ ] The WGSL hash implementation and the Rust mirror in `#[cfg(test)]` are kept
  in sync (comment in both locations makes this explicit)

---

## Design Decisions and Trade-offs

### Why f32 for seed instead of u32?

The slider system uses f32 throughout. Converting to u32 in the shader
(`u32(seed * 65535.0)`) gives 65,536 distinct seed values — enough variety that
users will never exhaust them by dragging, while keeping the struct uniform.
Using `u32` in the Rust struct would require a bytemuck workaround (mixing
`f32` and `u32` in a `Pod` struct) and a custom slider-to-uniform conversion.

### Why max count = 12?

12 gives a "very stained" look without excessive GPU work (12 distance
computations per pixel). Raising this later is a one-line constant change in the
WGSL. Setting it much higher (e.g., 50) would likely produce muddy, fully-brown
output with little artistic control.

### Why position range [0.05, 0.95]?

Stain rings at `centre ± ring_radius` should not exit the frame. A centre at UV
0.02 with radius 0.18 would be mostly off-screen. Constraining centres to
[0.05, 0.95] means a maximum-radius ring (0.20) can still be 85% visible even
in the worst case, which looks intentional rather than clipped.

### Why maintain a Rust hash mirror for tests?

The alternative — testing the property probabilistically across the full image
without knowing stain positions — is possible but fragile (thresholds become
magic numbers tied to the hash distribution). Mirroring the hash in Rust is ~15
lines of pure arithmetic with no dependencies, and it lets tests be precise about
which pixels to sample. The maintenance burden (keeping mirror in sync with WGSL)
is documented with a comment in both files.
