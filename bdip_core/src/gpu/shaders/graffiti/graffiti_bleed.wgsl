// Graffiti — Pass 1a: horizontal blur.
//
// First pass of a separable box blur. Blurs the source horizontally and writes
// the result to a scratch texture. The vertical pass (graffiti_bleed_v.wgsl)
// completes the 2D blur by reading this scratch texture.
//
// The blur radius is derived from the `bleed` parameter scaled against image
// size, with a compile-time cap (RADIUS_CAP) to bound register pressure and
// prevent excessively large kernels on very high-resolution images.
//
// At bleed=0 the radius is 0 and the pass copies the source pixel unchanged,
// so the scratch texture equals the source.
//
// All GraffitiParams fields must be declared in every pass to satisfy WebGPU's
// uniform binding-size validation requirement.

struct GraffitiParams {
    strength:     f32,
    color_levels: f32,
    edge_strength: f32,
    bleed:        f32,
}

// 1 input: source at binding 0, output at binding 1.
@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: GraffitiParams;

// BLEED_FRACTION controls blur radius as a fraction of the longer image dimension.
// At 0.015 on a 6000-px image: radius = ceil(0.015 * 6000) = 90 px.
const BLEED_FRACTION: f32 = 0.015;
const RADIUS_CAP:     i32 = 90;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);

    // Compute blur radius from bleed parameter.
    // bleed=0 → radius=0 (identity copy); bleed=1 → full BLEED_FRACTION radius.
    let base_radius = BLEED_FRACTION * f32(max(dims.x, dims.y));
    let radius      = min(i32(ceil(base_radius * params.bleed)), RADIUS_CAP);

    if radius == 0 {
        // No blur: pass through source unchanged.
        let pixel = textureLoad(src_texture, coord, 0);
        textureStore(dst_texture, coord, pixel);
        return;
    }

    // Horizontal blur: sample along a horizontal line of width 2*radius+1.
    // The vertical pass (graffiti_bleed_v.wgsl) follows to complete the 2D blur.
    let diameter      = 2 * radius + 1;
    let inv_diam      = 1.0 / f32(diameter);
    var accum: vec4<f32> = vec4<f32>(0.0);

    for (var dx: i32 = -radius; dx <= radius; dx++) {
        let tap_x = clamp(coord.x + dx, 0, i32(dims.x) - 1);
        accum += textureLoad(src_texture, vec2<i32>(tap_x, coord.y), 0);
    }

    textureStore(dst_texture, coord, accum * inv_diam);
}
