// Comic Book — halftone pass.
//
// Produces Ben-Day dot shading by comparing pixel luminance against a tiled
// radial-gradient threshold map. Bright pixels let the halftone show through;
// dark pixels are inked. The dot_scale parameter controls cell size in pixels.
//
// Binding layout (1 input → input at 0, output at 1):
//   @group(0) @binding(0) Source
//   @group(0) @binding(1) output
//   @group(1) @binding(0) uniform params
//   @group(2) @binding(0) halftone threshold texture
//   @group(2) @binding(1) halftone sampler

struct ComicBookParams {
    strength:        f32,
    dot_scale:       f32,
    edge_threshold:  f32,
    edge_thickness:  f32,
}

@group(0) @binding(0) var src_texture:       texture_2d<f32>;
@group(0) @binding(1) var dst_texture:       texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params:   ComicBookParams;
@group(2) @binding(0) var halftone_tex:      texture_2d<f32>;
@group(2) @binding(1) var halftone_sampler:  sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<u32>(gid.xy);
    let color = textureLoad(src_texture, coord, 0);
    let luma = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Tile the threshold map: one cell per dot_scale pixels.
    let uv = fract(vec2<f32>(gid.xy) / params.dot_scale);
    let threshold = textureSampleLevel(halftone_tex, halftone_sampler, uv, 0.0).r;

    // Ben-Day: pixel is "paper" (white) when luma >= threshold, "ink" (black) otherwise.
    let dot_val = select(0.0, 1.0, luma >= threshold);

    textureStore(dst_texture, coord, vec4<f32>(dot_val, dot_val, dot_val, color.a));
}
