// Tilt-Shift — composite pass.
//
// Blends the original (sharp) source with the Gaussian-blurred texture using a
// gradient mask derived from each pixel's vertical distance to the focus band.
//
// Focus band definition:
//   band_top    = focus_center - focus_width * 0.5
//   band_bottom = focus_center + focus_width * 0.5
//
// Blend weight (blend_t) per pixel:
//   - 0.0 inside the focus band       → 100% sharp source
//   - 1.0 at maximum distance outside → 100% blurred
//   - linearly interpolated over a transition zone of width `TRANSITION_FRACTION`
//     on each side of the band edges
//
// The transition zone softens the boundary between sharp and blurred regions,
// avoiding a hard edge at the focus band border. Its width is a fixed fraction
// of the image height.
//
// All five Tilt-Shift WGSL files declare the full TiltShiftParams struct to
// satisfy WebGPU's uniform binding-size validation.

struct TiltShiftParams {
    focus_center:  f32,
    focus_width:   f32,
    blur_strength: f32,
    _padding:      f32,
}

// Bindings — position-indexed (2 inputs → inputs at 0 and 1, output at 2).
@group(0) @binding(0) var input_source:  texture_2d<f32>;
@group(0) @binding(1) var input_blurred: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: TiltShiftParams;

// Transition zone half-width as a fraction of image height. A value of 0.05
// means the blend transitions over 5% of the image height on each band edge,
// which produces a visually smooth falloff on typical images.
const TRANSITION_FRACTION: f32 = 0.05;

// Computes the blend weight [0, 1] for a given normalised vertical position (y_norm).
//   0 = fully inside the focus band (use sharp source)
//   1 = fully outside the focus band (use blurred version)
fn blend_weight(y_norm: f32, band_top: f32, band_bottom: f32, transition: f32) -> f32 {
    // Signed distance from the nearest band edge (positive = outside the band).
    let dist_top    = band_top - y_norm;      // positive above band
    let dist_bottom = y_norm - band_bottom;   // positive below band
    let dist_outside = max(dist_top, dist_bottom);

    // Map distance to [0, 1] over the transition zone.
    // smoothstep provides a C1-continuous falloff.
    return smoothstep(0.0, transition, dist_outside);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);

    // Normalised vertical position: 0.0 at top row, 1.0 at bottom row.
    let y_norm = (f32(gid.y) + 0.5) / f32(dims.y);

    let half_width  = params.focus_width * 0.5;
    let band_top    = params.focus_center - half_width;
    let band_bottom = params.focus_center + half_width;
    let transition  = TRANSITION_FRACTION;

    let blend_t = blend_weight(y_norm, band_top, band_bottom, transition);

    let src     = textureLoad(input_source,  coord, 0);
    let blurred = textureLoad(input_blurred, coord, 0);

    // Linear blend: 0 = sharp, 1 = blurred.
    let out_rgb = mix(src.rgb, blurred.rgb, blend_t);

    // Alpha is copied from the source — tilt-shift must not alter transparency.
    textureStore(output_texture, coord, vec4<f32>(out_rgb, src.a));
}
