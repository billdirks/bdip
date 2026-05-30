struct TintParams {
    tint: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: TintParams;

// RGB <-> YIQ conversion matrices for shifting the Q (green-magenta) axis.
fn rgb_to_yiq(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.299,  0.587,  0.114)),
        dot(c, vec3<f32>(0.596, -0.274, -0.322)),
        dot(c, vec3<f32>(0.211, -0.523,  0.312)),
    );
}

fn yiq_to_rgb(yiq: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        yiq.x + 0.956 * yiq.y + 0.621 * yiq.z,
        yiq.x - 0.272 * yiq.y - 0.647 * yiq.z,
        yiq.x - 1.106 * yiq.y + 1.703 * yiq.z,
    );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    let color = textureLoad(src_texture, coord, 0);

    var yiq = rgb_to_yiq(color.rgb);
    // Shift Q axis to correct green (negative) / magenta (positive) color casts.
    yiq.z = yiq.z + params.tint;
    let out_rgb = clamp(yiq_to_rgb(yiq), vec3<f32>(0.0), vec3<f32>(1.0));

    textureStore(dst_texture, coord, vec4<f32>(out_rgb, color.a));
}
