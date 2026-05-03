// Daguerreotype — Pass 1: vignette, grain, and final blend
//
// Reads both the original source (binding 0) and the toned scratch from Pass 0
// (binding 1), applies a strong elliptical vignette and procedural fine grain to
// the toned image, then blends the result with the original based on `strength`.
//
// Grain approach:
//   A single-pass hash-based noise generator avoids external texture dependencies
//   (as required for this [MS] effect).  The hash function combines pixel coordinates
//   with a fixed salt to produce a pseudo-random scalar in [0, 1].  The grain
//   amplitude is modulated by luminance — brighter areas receive more grain, matching
//   the behaviour of silver-halide emulsions where dense, bright regions trap more
//   silver particles.
//
// Bind-group layout (2-input pass; N=2, so output is binding 2):
//   group(0) binding(0): source texture  (original)
//   group(0) binding(1): toned scratch   (Pass 0 output)
//   group(0) binding(2): output storage
//   group(1) binding(0): uniform params

struct DaguerreotypeParams {
    strength: f32,
    _pad0:    f32,
    _pad1:    f32,
    _pad2:    f32,
}

@group(0) @binding(0) var src_texture:   texture_2d<f32>;
@group(0) @binding(1) var toned_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform>       params: DaguerreotypeParams;

// Rec. 709 luminance — reused for grain weighting.
const LUM_R: f32 = 0.2126;
const LUM_G: f32 = 0.7152;
const LUM_B: f32 = 0.0722;

// Maximum grain amplitude in linear [0,1] space.  0.015 corresponds to roughly
// ±980 u16 at mid-tone, giving a fine but perceptible silver-salt texture.
const GRAIN_AMP: f32 = 0.015;

// Hash function: combines two u32 coordinates into a pseudo-random f32 in [0, 1].
// Based on the "pcg-like" integer hash; chosen for its even bit distribution at
// low cost relative to alternative noise approaches (Perlin, Simplex, etc.).
fn hash2(x: u32, y: u32) -> f32 {
    var v = x * 1664525u + y * 1013904223u;
    v = v ^ (v >> 16u);
    v = v * 2246822519u;
    v = v ^ (v >> 13u);
    v = v * 3266489917u;
    v = v ^ (v >> 16u);
    return f32(v) / 4294967295.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let src_dims = textureDimensions(src_texture);
    if global_id.x >= src_dims.x || global_id.y >= src_dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let src   = textureLoad(src_texture,   coord, 0);
    let toned = textureLoad(toned_texture, coord, 0);

    // --- Vignette ---
    // Normalised UV: (0,0) top-left, (1,1) bottom-right.
    let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / vec2<f32>(src_dims);
    let centered = uv - vec2<f32>(0.5);

    // Use an elliptical distance that accounts for non-square images, then
    // apply a strong falloff starting at ~0.4 of the half-diagonal.
    let aspect     = f32(src_dims.x) / f32(src_dims.y);
    let d_sq       = centered.x * centered.x + (centered.y * aspect) * (centered.y * aspect);
    let d          = sqrt(d_sq);
    let vig_start  = 0.38;
    let vig_end    = 0.80;
    let vig_factor = 1.0 - smoothstep(vig_start, vig_end, d);

    // --- Grain ---
    // Luminance of toned pixel drives grain weight: denser silver = more grain.
    let luma        = toned.r * LUM_R + toned.g * LUM_G + toned.b * LUM_B;
    let grain_weight = sqrt(clamp(luma, 0.0, 1.0));
    let noise        = hash2(global_id.x, global_id.y);
    // Centre noise around 0: [0,1] → [-1, +1], then scale.
    let grain        = (noise * 2.0 - 1.0) * GRAIN_AMP * grain_weight;

    // Apply vignette and grain to the toned image.
    let processed = vec3<f32>(
        toned.r * vig_factor + grain,
        toned.g * vig_factor + grain,
        toned.b * vig_factor + grain,
    );

    // Final blend: mix original source with processed result based on strength.
    // At strength=0 the output equals the source (identity).
    let out_rgb = mix(src.rgb, processed, params.strength);
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
