// Charcoal Sketch — Pass 2: procedural charcoal grain and final composite.
//
// Reads the inverted-edge scratch texture from pass 1 (.r = paper/stroke intensity,
// where 1.0 = paper background and 0.0 = dark charcoal stroke) and the original
// source. Produces the charcoal sketch output:
//   1. Starts from the paper-background intensity stored in pass 1.
//   2. Applies a multi-frequency hash-based grain to simulate the rough,
//      granular texture of charcoal on paper — distinct from Pencil Sketch's
//      directional stroke blur and Chalkboard's dark-background chalk grain.
//      Charcoal grain is coarser and denser, applied as a dark smear rather than
//      a fine sparkle, so the grain is additive-dark (subtracted from lightness)
//      and scaled to the stroke regions.
//   3. Tints the result toward a warm cream paper tone (rather than pure white),
//      giving the characteristic off-white charcoal paper look.
//   4. Blends the charcoal result with the original source via `strength`.
//
// The grain is generated entirely in the shader using a 2D integer hash — no
// external texture is required.
//
// Identity: when strength = 0.0, the output equals the source image exactly,
// regardless of edge_strength or grain_amount.
//
// All CharcoalSketchParams fields must be declared in every pass to satisfy
// WebGPU's uniform binding-size validation requirement.

struct CharcoalSketchParams {
    // Blend factor: 0.0 = source unchanged (identity), 1.0 = full effect.
    strength:      f32,
    // Multiplier on raw Sobel magnitude. Unused in this pass but declared for
    // WebGPU uniform-size parity with pass 1.
    edge_strength: f32,
    // Amplitude of the procedural grain noise, in [0.0, 1.0].
    grain_amount:  f32,
    _padding:      f32,
}

// Bindings — 2 inputs: source at binding 0, edges scratch at binding 1,
// output at binding 2.
@group(0) @binding(0) var src_texture:   texture_2d<f32>;
@group(0) @binding(1) var edge_texture:  texture_2d<f32>;
@group(0) @binding(2) var dst_texture:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CharcoalSketchParams;

// Warm cream paper background colour (linear light).
// Chosen to approximate the off-white tone of drawing paper used for charcoal work.
// Pure white (1.0, 1.0, 1.0) is too stark for charcoal; cream warms the highlights.
const PAPER_COLOR: vec3<f32> = vec3<f32>(0.96, 0.93, 0.88);

// Maximum grain amplitude at grain_amount = 1.0 (linear light).
// Charcoal grain is darker than Chalkboard's chalk grain — 0.12 linear is noticeable
// without overwhelming the stroke detail. The grain is applied as a dark subtraction.
const MAX_GRAIN: f32 = 0.12;

// A fast, integer-based 2D hash producing a pseudo-random value in [0, 1].
// Uses Wang hash mixing. Output is purely a deterministic function of the
// pixel coordinate, so the grain pattern is stable across frames.
fn hash2(coord: vec2<u32>) -> f32 {
    var h: u32 = coord.x * 1664525u + coord.y * 1013904223u + 2891336453u;
    h ^= (h >> 16u);
    h *= 0x45d9f3bu;
    h ^= (h >> 16u);
    return f32(h) / f32(0xffffffffu);
}

// A coarser-frequency hash at half-resolution coordinates, blended with the
// pixel-frequency hash to simulate the multi-scale clumping of charcoal grain.
fn hash2_coarse(coord: vec2<u32>) -> f32 {
    // Shift seed constants to produce an independent sequence from hash2.
    var h: u32 = coord.x * 22695477u + coord.y * 1664525u + 1013904223u;
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

    // paper_value is the inverted Sobel result from pass 1:
    //   1.0 = flat paper region, 0.0 = dark charcoal stroke.
    let paper_value = edge.r;

    // Map paper_value to the cream paper colour range:
    //   paper_value = 1.0 → PAPER_COLOR (light cream background)
    //   paper_value = 0.0 → near-black charcoal stroke (very dark grey)
    // The dark endpoint is not pure black to avoid fully clipping fine stroke detail.
    let STROKE_COLOR: vec3<f32> = vec3<f32>(0.04, 0.035, 0.03);
    let charcoal_rgb = mix(STROKE_COLOR, PAPER_COLOR, paper_value);

    // Procedural charcoal grain: multi-frequency noise to simulate the granular
    // texture of real charcoal medium. Two octaves are blended:
    //   - Fine grain (pixel frequency): sharp, fine-grained texture.
    //   - Coarse grain (half-pixel frequency): broader clumps of charcoal pigment.
    // Different seed offsets per channel break up hash-grid regularity.
    let fine_r   = hash2(gid.xy + vec2<u32>(0u,    0u));
    let fine_g   = hash2(gid.xy + vec2<u32>(317u,  0u));
    let fine_b   = hash2(gid.xy + vec2<u32>(0u,  591u));

    let half_coord = gid.xy / 2u;
    let coarse_r   = hash2_coarse(half_coord + vec2<u32>(0u,   0u));
    let coarse_g   = hash2_coarse(half_coord + vec2<u32>(173u, 0u));
    let coarse_b   = hash2_coarse(half_coord + vec2<u32>(0u, 251u));

    // Blend fine and coarse at 60/40 to give charcoal's characteristic lumpy texture.
    let blended_r = fine_r * 0.6 + coarse_r * 0.4;
    let blended_g = fine_g * 0.6 + coarse_g * 0.4;
    let blended_b = fine_b * 0.6 + coarse_b * 0.4;

    // Charcoal grain is applied as a dark subtraction: grain values in [0,1] are
    // remapped to [0, MAX_GRAIN] and subtracted, darkening the paper in a smeared
    // pattern. This is different from Chalkboard grain (bilateral ±, light sparkle)
    // and more closely resembles the dark smear of charcoal pigment on paper.
    let grain_scale = params.grain_amount * MAX_GRAIN;
    let grain = vec3<f32>(
        blended_r * grain_scale,
        blended_g * grain_scale,
        blended_b * grain_scale,
    );

    // Subtract grain from the charcoal layer (dark smear, not bilateral ± noise).
    // Clamping is intentionally omitted to preserve headroom for downstream shaders.
    let charcoal_with_grain = charcoal_rgb - grain;

    // Final blend: at strength=0 the output equals the source (identity).
    let out_rgb = mix(src.rgb, charcoal_with_grain, params.strength);

    // Alpha is preserved from the source image.
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
