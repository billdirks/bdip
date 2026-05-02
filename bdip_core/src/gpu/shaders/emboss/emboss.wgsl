struct EmbossParams {
    strength:  f32,
    direction: f32,
    _padding:  vec2<f32>,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: EmbossParams;

// Rec. 709 luminance coefficients.
const REC709: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

fn luma(c: vec4<f32>) -> f32 {
    return dot(c.rgb, REC709);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let coord = vec2<i32>(global_id.xy);
    let src   = textureLoad(src_texture, coord, 0);

    // Convert the direction angle (degrees) to a unit-vector offset.
    // A step of 1 pixel is used, so the offset components are rounded to
    // the nearest integer to form a valid texel address. The direction
    // controls which axis the "lighting" appears to come from.
    let angle_rad = params.direction * (3.14159265358979 / 180.0);
    let offset    = vec2<i32>(
        i32(round(cos(angle_rad))),
        i32(round(sin(angle_rad))),
    );

    // Clamp neighbour coordinates to image bounds so border pixels produce a
    // sensible result without out-of-bounds reads.
    let fwd_coord = clamp(coord + offset,     vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
    let bwd_coord = clamp(coord - offset,     vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));

    let fwd = textureLoad(src_texture, fwd_coord, 0);
    let bwd = textureLoad(src_texture, bwd_coord, 0);

    // Height difference between opposite neighbours gives a surface-relief signal.
    // Adding 0.5 shifts the neutral (flat) result to mid-gray.
    let height_diff = luma(fwd) - luma(bwd);
    let emboss_luma = clamp(height_diff * params.strength + 0.5, 0.0, 1.0);
    let emboss_rgb  = vec3<f32>(emboss_luma);

    // Blend: strength=0.0 passes the source through unchanged (identity).
    // At full strength the output is the emboss relief as a grayscale image.
    // The blend parameter doubles as both the convolution scale and the mix
    // weight, which keeps the API minimal: one slider, one semantic.
    let t   = clamp(params.strength, 0.0, 1.0);
    let out = vec4<f32>(mix(src.rgb, emboss_rgb, t), src.a);

    textureStore(dst_texture, coord, out);
}
