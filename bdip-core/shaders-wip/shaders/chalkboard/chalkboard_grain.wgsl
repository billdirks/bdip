// Chalkboard — Pass 2: procedural chalk grain and final composite.
//
// Reads the chalk-line scratch texture from pass 1 (.r = chalk line intensity)
// and the original source. Produces the chalkboard output:
//   1. Starts from the dark chalkboard background colour.
//   2. Adds the bright chalk lines from pass 1.
//   3. Overlays procedural chalk-grain noise (hash-based) to simulate the
//      rough texture of chalk strokes on a textured board surface.
//   4. Blends the chalkboard result with the original source via `strength`.
//
// The grain is generated entirely in the shader using a 2D integer hash — no
// external texture is required.
//
// Identity: when strength = 0.0, the output equals the source image exactly,
// regardless of chalk_boost.
//
// All ChalkboardParams fields must be declared in every pass to satisfy
// WebGPU's uniform binding-size validation requirement.

struct ChalkboardParams {
    // Blend factor: 0.0 = source unchanged (identity), 1.0 = full effect.
    strength:    f32,
    // Multiplier on raw Sobel magnitude. Unused in this pass but declared for
    // WebGPU uniform-size parity with pass 1.
    chalk_boost: f32,
    _padding:    vec2<f32>,
}

// Bindings — 2 inputs: source at binding 0, edges scratch at binding 1,
// output at binding 2.
@group(0) @binding(0) var src_texture:   texture_2d<f32>;
@group(0) @binding(1) var edge_texture:  texture_2d<f32>;
@group(0) @binding(2) var dst_texture:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: ChalkboardParams;

// Dark chalkboard background: a muted dark green-black (linear light).
// Chosen to approximate the classic green school chalkboard colour.
const BOARD_COLOR: vec3<f32> = vec3<f32>(0.007, 0.020, 0.007);

// Grain amplitude: maximum ±deviation added to each channel.
// 0.04 in linear light is perceptible but subtle — enough to simulate chalk
// texture without overwhelming the edge lines.
const GRAIN_SCALE: f32 = 0.04;

// A fast, integer-based 2D hash producing a value in [0, 1].
// Uses Wang hash mixing to distribute the input coordinates into a
// pseudo-random scalar. No state is needed — output is purely a function
// of the pixel coordinate, so the grain pattern is stable (deterministic).
fn hash2(coord: vec2<u32>) -> f32 {
    var h: u32 = coord.x * 1664525u + coord.y * 1013904223u + 2891336453u;
    h ^= (h >> 16u);
    h *= 0x45d9f3bu;
    h ^= (h >> 16u);
    return f32(h) / f32(0xffffffffu);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(src_texture,  coord, 0);
    let edge  = textureLoad(edge_texture, coord, 0);

    // chalk_line is the bright chalk-line intensity from pass 1, in [0, 1].
    let chalk_line = edge.r;

    // Start from the dark chalkboard background, then add the chalk lines.
    // chalk_line = 0 → background; chalk_line = 1 → white chalk (1.0, 1.0, 1.0).
    let chalk_rgb = BOARD_COLOR + (vec3<f32>(1.0) - BOARD_COLOR) * chalk_line;

    // Procedural chalk grain: a fine noise pattern with different seeds per channel
    // to break up the uniform appearance of the hash grid.
    let g_r = hash2(gid.xy + vec2<u32>(0u,   0u));
    let g_g = hash2(gid.xy + vec2<u32>(317u, 0u));
    let g_b = hash2(gid.xy + vec2<u32>(0u,   591u));
    // Remap hash output from [0,1] to [-1,+1] and scale by GRAIN_SCALE.
    let grain = vec3<f32>(
        (g_r - 0.5) * (2.0 * GRAIN_SCALE),
        (g_g - 0.5) * (2.0 * GRAIN_SCALE),
        (g_b - 0.5) * (2.0 * GRAIN_SCALE),
    );

    // Add grain to the chalk layer. Clamping is intentionally omitted to
    // preserve headroom for downstream shaders in the pipeline.
    let chalk_with_grain = chalk_rgb + grain;

    // Final blend: at strength=0 the output equals the source (identity).
    let out_rgb = mix(src.rgb, chalk_with_grain, params.strength);

    // Alpha is preserved from the source image.
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
