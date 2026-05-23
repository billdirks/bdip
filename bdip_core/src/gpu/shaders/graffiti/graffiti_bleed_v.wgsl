// Graffiti — Pass 1b: vertical blur.
//
// Second pass of the separable box blur. Reads the horizontally-blurred scratch
// texture (from graffiti_bleed.wgsl) and applies vertical blur to complete the
// 2D blur that simulates spray-paint overspray. The two-pass separable approach
// produces an isotropic result with cost O(r) per pixel rather than O(r²).
//
// The blur radius formula is identical to Pass 1a so that both passes use the
// same kernel size.
//
// All GraffitiParams fields must be declared in every pass to satisfy WebGPU's
// uniform binding-size validation requirement.

struct GraffitiParams {
    strength:      f32,
    color_levels:  f32,
    edge_strength: f32,
    bleed:         f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: GraffitiParams;

const BLEED_FRACTION: f32 = 0.015;
const RADIUS_CAP:     i32 = 90;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);

    // Compute blur radius (same formula as horizontal pass).
    let base_radius = BLEED_FRACTION * f32(max(dims.x, dims.y));
    let radius      = min(i32(ceil(base_radius * params.bleed)), RADIUS_CAP);

    if radius == 0 {
        let pixel = textureLoad(src_texture, coord, 0);
        textureStore(dst_texture, coord, pixel);
        return;
    }

    // Vertical blur: sample along a vertical line of height 2*radius+1.
    let diameter = 2 * radius + 1;
    let inv_diam = 1.0 / f32(diameter);
    var accum: vec4<f32> = vec4<f32>(0.0);

    for (var dy: i32 = -radius; dy <= radius; dy++) {
        let tap_y = clamp(coord.y + dy, 0, i32(dims.y) - 1);
        accum += textureLoad(src_texture, vec2<i32>(coord.x, tap_y), 0);
    }

    textureStore(dst_texture, coord, accum * inv_diam);
}
