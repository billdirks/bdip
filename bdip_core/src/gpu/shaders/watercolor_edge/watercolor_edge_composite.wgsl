// Watercolor Edge — Pass 2: dark-edge composite.
//
// Reads the source image and the Sobel scratch texture from pass 1, then
// applies a dark multiplication mask to produce the watercolor-edge look:
//
//   dark_mask = 1.0 - (edge_intensity * strength)
//   out_rgb   = src.rgb * dark_mask
//
// When strength = 0.0, dark_mask = 1.0 everywhere → output equals source (identity).
// When strength = 1.0 and edge_intensity = 1.0, dark_mask = 0.0 → fully black edge.
//
// The multiplication approach keeps the original hues intact (only darkens them),
// which is the defining characteristic of the watercolor illustration look: colors
// bleed into one another but edges are darkened, not recolored.
//
// All WatercolorEdgeParams fields must be declared in every pass to satisfy
// WebGPU's uniform binding-size validation requirement.

struct WatercolorEdgeParams {
    // Blend factor: 0.0 = source unchanged (identity), 1.0 = full dark-edge effect.
    strength:  f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

// Bindings — 2 inputs: source at binding 0, edge scratch at binding 1, output at binding 2.
@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var edge_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture:  texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: WatercolorEdgeParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(src_texture,  coord, 0);
    let edge  = textureLoad(edge_texture, coord, 0);

    let edge_intensity = edge.r;

    // Build a dark mask: at a strong edge the mask approaches (1 - strength),
    // which multiplies into the source to darken it. At zero edge intensity
    // the mask is always 1.0, leaving the source unchanged regardless of strength.
    let dark_mask = 1.0 - (edge_intensity * params.strength);

    // Multiply the source RGB by the dark mask. Values above 1.0 (linear-light
    // headroom) are intentionally preserved without clamping so downstream shaders
    // can continue to operate in the full floating-point range.
    let out_rgb = src.rgb * dark_mask;

    // Alpha is preserved from the source image.
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
