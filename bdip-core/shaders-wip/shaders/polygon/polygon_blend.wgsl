// Polygon — Pass 2: strength blend.
//
// Blends the Voronoi facet colour (from the scratch texture produced by pass 1)
// with the original source image according to `params.strength`.
//
// Identity: when strength = 0.0, mix(src, facets, 0.0) = src, so the output
// equals the source image regardless of density or jitter.
//
// All PolygonParams fields must be declared in every pass to satisfy WebGPU's
// uniform binding-size validation requirement.

struct PolygonParams {
    // Blend factor: 0.0 = source unchanged (identity), 1.0 = full effect.
    strength: f32,
    // Not read in this pass; declared to match the shared uniform buffer layout.
    density:  f32,
    jitter:   f32,
    _padding: f32,
}

// Bindings — 2 inputs: source at binding 0, voronoi scratch at binding 1,
// output at binding 2.
@group(0) @binding(0) var src_texture:     texture_2d<f32>;
@group(0) @binding(1) var voronoi_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture:     texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: PolygonParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord   = vec2<i32>(gid.xy);
    let src     = textureLoad(src_texture,     coord, 0);
    let facets  = textureLoad(voronoi_texture, coord, 0);

    // Blend: strength=0.0 → source unchanged (identity); strength=1.0 → full facets.
    let out_rgb = mix(src.rgb, facets.rgb, params.strength);

    // Alpha is preserved from the source image.
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
