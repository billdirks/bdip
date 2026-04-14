@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

struct ContrastParams {
    contrast_offset: f32,
    // Padding to satisfy WebGPU 16-byte alignment rules for uniforms in structs.
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(1) @binding(0) var<uniform> params: ContrastParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);

    // Scale each channel outward from the linear neutral midpoint (0.5).
    //   contrast_offset =  0.0 → scale = 1.0 → unchanged
    //   contrast_offset =  1.0 → scale = 2.0 → darks pushed to black, lights to white
    //   contrast_offset = -1.0 → scale = 0.0 → entire image flattened to neutral gray
    let scale = 1.0 + params.contrast_offset;
    let new_rgb = clamp(
        (color.rgb - vec3<f32>(0.5)) * scale + vec3<f32>(0.5),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let final_color = vec4<f32>(new_rgb, color.a);

    textureStore(dst_texture, coords, final_color);
}
