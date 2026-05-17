// Cat Ghost: tiles a transparent cat image over the source using centered tiling,
// then alpha-composites the overlay at the requested strength.
//
// Centering formula keeps partial edge tiles symmetric:
//   surplus = ceil(canvas / tile_size) * tile_size - canvas
//   uv = fract((coord + surplus * 0.5) / tile_size)

struct CatGhostParams {
    size:     f32,      // x-dimension of each tile in pixels
    strength: f32,      // overlay opacity [0, 1]; 0 is identity
    _padding: vec2<f32>,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: CatGhostParams;
@group(2) @binding(0) var cat_tex:     texture_2d<f32>;
@group(2) @binding(1) var cat_sampler: sampler;

// Aspect ratio of the cat image: height / width = 1498 / 1129.
const CAT_ASPECT: f32 = 1498.0 / 1129.0;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<f32>(gid.xy);

    // Tile dimensions in pixels.
    let tile_w = params.size;
    let tile_h = params.size * CAT_ASPECT;
    let canvas = vec2<f32>(f32(dims.x), f32(dims.y));
    let tile   = vec2<f32>(tile_w, tile_h);

    // Center the tile grid so partial tiles at each edge are equal in width/height.
    let surplus = ceil(canvas / tile) * tile - canvas;
    let uv = fract((coord + surplus * 0.5) / tile);

    let src     = textureLoad(src_texture, vec2<i32>(gid.xy), 0);
    let overlay = textureSampleLevel(cat_tex, cat_sampler, uv, 0.0);

    // Alpha composite: cat over source, scaled by strength.
    // Source alpha is preserved unchanged.
    let effective_alpha = overlay.a * params.strength;
    let out_rgb = src.rgb * (1.0 - effective_alpha) + overlay.rgb * effective_alpha;
    textureStore(dst_texture, vec2<i32>(gid.xy), vec4<f32>(out_rgb, src.a));
}
