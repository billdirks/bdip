struct MirrorReflectionParams {
    // Mirror axis mode encoded as a float integer:
    //   0.0 = none (identity, no mirroring)
    //   1.0 = horizontal (flip left/right along vertical axis)
    //   2.0 = vertical   (flip top/bottom along horizontal axis)
    //   3.0 = both axes
    mode:    f32,
    // Blend factor in [0.0, 1.0]. 0.0 = original image (identity);
    // 1.0 = fully mirrored. Values between blend smoothly.
    blend:   f32,
    _pad0:   f32,
    _pad1:   f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: MirrorReflectionParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Original pixel from the source at the current output coordinate.
    let original = textureLoad(src_texture, vec2<i32>(coord), 0);

    // Compute the mirrored source coordinate by flipping based on mode.
    // mode is treated as an integer flag:
    //   bit 0 (mode & 1): horizontal flip (mirror x)
    //   bit 1 (mode & 2): vertical flip   (mirror y)
    let mode_i = i32(round(params.mode));
    let flip_h = (mode_i & 1) != 0;
    let flip_v = (mode_i & 2) != 0;

    var mirror_x = coord.x;
    var mirror_y = coord.y;
    if flip_h {
        mirror_x = dims.x - 1u - coord.x;
    }
    if flip_v {
        mirror_y = dims.y - 1u - coord.y;
    }

    let mirror_coord = vec2<i32>(i32(mirror_x), i32(mirror_y));
    let mirrored = textureLoad(src_texture, mirror_coord, 0);

    // Blend between original and mirrored. At blend=0 the output equals the
    // original (identity); at blend=1 the output is the fully mirrored image.
    let out = mix(original, mirrored, params.blend);
    textureStore(dst_texture, vec2<i32>(coord), out);
}
