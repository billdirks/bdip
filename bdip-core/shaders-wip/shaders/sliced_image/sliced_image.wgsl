struct SlicedImageParams {
    // Number of horizontal slices to divide the image into. Range [1, 50].
    // At 1, the entire image is one slice and the effect is a uniform X shift.
    slice_count: f32,
    // Horizontal UV offset applied to each slice. Range [0.0, 0.5].
    // At 0.0 no shift is applied (identity). The sign alternates per slice when
    // alternating_direction is 1.0.
    slice_offset: f32,
    // When 1.0, odd-indexed slices shift by +slice_offset and even-indexed slices
    // shift by -slice_offset. When 0.0, all slices shift by +slice_offset.
    alternating_direction: f32,
    _padding: f32,
}

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: SlicedImageParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Normalized UV in [0, 1] with half-pixel offset for pixel centres.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Determine which slice this pixel belongs to. slice_count is clamped to at
    // least 1.0 to avoid division by zero when the user sets it to 0.
    let count = max(params.slice_count, 1.0);
    let slice_index = floor(uv.y * count);

    // Compute the X offset for this slice. When alternating_direction is 1.0 the
    // offset sign flips on every other slice. The modulo check uses 0.5 as a
    // threshold to treat the boolean flag as a continuous f32 parameter: values
    // below 0.5 are treated as "off" and values >= 0.5 as "on".
    var x_offset: f32;
    if params.alternating_direction >= 0.5 {
        // Odd slices (index 1, 3, 5, …) shift right; even slices shift left.
        let sign = select(-1.0, 1.0, (i32(slice_index) % 2) == 1);
        x_offset = sign * params.slice_offset;
    } else {
        x_offset = params.slice_offset;
    }

    // Apply horizontal shift and wrap with fract so the image tiles seamlessly
    // rather than producing black fill at the edges. Wrapping is chosen over black
    // fill because the "sliced" aesthetic expects visible image content in every
    // slice cell; black fill would create empty bands that obscure the effect.
    let src_uv = vec2<f32>(fract(uv.x + x_offset), uv.y);

    // Convert UV back to integer texture coordinates for a nearest-neighbour sample.
    let src_coord = vec2<i32>(clamp(
        vec2<i32>(src_uv * vec2<f32>(dims)),
        vec2<i32>(0, 0),
        vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1),
    ));

    let color = textureLoad(src_texture, src_coord, 0);
    textureStore(dst_texture, coord, color);
}
