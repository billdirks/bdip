// Tintype — Pass 1: strong radial vignette
//
// Applies a heavy elliptical vignette to the toned scratch produced by Pass 0.
// The result is written to a second scratch texture for Pass 2 to overlay grit.
//
// Tintype vignettes are more aggressive than daguerreotypes: the corners go
// almost fully black because hand-held plates were often unevenly coated at the
// edges.  The falloff therefore starts closer to the centre (0.30 rather than
// 0.38) and reaches full black earlier (0.70 rather than 0.80).
//
// Bind-group layout (1-input pass):
//   group(0) binding(0): toned scratch (Pass 0 output)
//   group(0) binding(1): vignette scratch output
//   group(1) binding(0): uniform params

struct TintypeParams {
    strength: f32,
    _pad0:    f32,
    _pad1:    f32,
    _pad2:    f32,
}

@group(0) @binding(0) var toned_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform>       params: TintypeParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(toned_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let toned = textureLoad(toned_texture, coord, 0);

    // Normalised UV centred at (0,0) spanning [-0.5, +0.5] on each axis.
    let uv      = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let centered = uv - vec2<f32>(0.5);

    // Elliptical radial distance, corrected for aspect ratio so the vignette
    // is circular in image-space rather than pixel-space.
    let aspect     = f32(dims.x) / f32(dims.y);
    let d          = sqrt(centered.x * centered.x +
                          (centered.y * aspect) * (centered.y * aspect));

    // Aggressive falloff: starts at 0.30, reaches zero at 0.70.
    let vig_start  = 0.30;
    let vig_end    = 0.70;
    let vig_factor = 1.0 - smoothstep(vig_start, vig_end, d);

    // Apply vignette to the toned image — alpha is preserved unchanged.
    let vignetted = toned.rgb * vig_factor;
    textureStore(dst_texture, coord, vec4<f32>(vignetted, toned.a));
}
