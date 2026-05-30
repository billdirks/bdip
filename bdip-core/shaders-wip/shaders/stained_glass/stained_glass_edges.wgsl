// Stained Glass — edge compositing pass.
//
// Reads the Voronoi scratch texture produced by the previous pass (RGB = cell
// colour, A = boundary proximity) and composites dark cell borders over it.
// Finally blends the result with the original source image according to
// `params.strength`.
//
// When strength = 0.0 (the identity default) the output equals the source
// image exactly, regardless of cell_size or edge_width.

struct StainedGlassParams {
    strength:   f32,
    cell_size:  f32,
    edge_width: f32,
    _padding:   f32,
}

// Bindings — 2 inputs: source (binding 0), voronoi scratch (binding 1), then
// output at binding 2 (position-indexed: N inputs → output at binding N).
@group(0) @binding(0) var src_texture:     texture_2d<f32>;
@group(0) @binding(1) var voronoi_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture:     texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: StainedGlassParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src    = textureLoad(src_texture,     coord, 0);
    let vor    = textureLoad(voronoi_texture, coord, 0);

    // vor.rgb = flat cell colour; vor.a = boundary proximity in [0, 1].
    let cell_color       = vor.rgb;
    let boundary_prox    = vor.a;

    // Edge mask: a pixel is "on an edge" when its boundary proximity is below
    // the edge_width threshold. smoothstep produces a soft gradient so the
    // border doesn't have a hard aliased look.
    //
    // When edge_width = 0 the smoothstep returns 0 for all pixels → no edges.
    let edge_mask = 1.0 - smoothstep(
        0.0,
        params.edge_width + 1e-6,   // avoid zero-range smoothstep
        boundary_prox,
    );

    // Dark edge colour: near-black with a subtle warm tint typical of lead
    // came (the metal strips in real stained glass).
    let edge_color = vec3<f32>(0.04, 0.03, 0.02);

    // Composite: blend cell colour with edge colour based on the edge mask.
    let stained_rgb = mix(cell_color, edge_color, edge_mask);

    // Final blend: at strength=0 pass through source unchanged (identity).
    let out_rgb = mix(src.rgb, stained_rgb, params.strength);

    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
