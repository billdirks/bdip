// Pencil Sketch — Pass 2: directional blur along edge orientation.
//
// Reads the Sobel scratch texture from pass 1 (edge intensity in .r, gradient
// angle in .g) and produces the final pencil-sketch output:
//   1. Optionally blurs edge intensity along the stroke direction (directional blur).
//   2. Inverts the intensity to produce dark lines on a white background.
//   3. Blends the sketch result back with the original source via `params.strength`.
//
// Identity: when strength = 0.0, the output equals the source image exactly,
// regardless of edge_strength or stroke_softness.
//
// All PencilSketchParams fields must be declared in every pass to satisfy
// WebGPU's uniform binding-size validation requirement.

struct PencilSketchParams {
    // Blend factor: 0.0 = source unchanged (identity), 1.0 = full effect.
    strength:        f32,
    // Multiplier on raw Sobel magnitude. Higher values amplify faint edges.
    edge_strength:   f32,
    // Directional blur extent. 0.0 = no blur; 1.0 = maximum stroke softness.
    stroke_softness: f32,
    _padding:        f32,
}

// Bindings — 2 inputs: source at binding 0, edge scratch at binding 1, output at binding 2.
@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var edge_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture:  texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: PencilSketchParams;

// SIGMA_FRACTION drives the directional blur radius relative to image size.
// At 0.01 fraction on a 6000-px image: sigma ≈ 60 px → radius ≈ 180 px.
const SIGMA_FRACTION: f32 = 0.01;
const RADIUS_CAP:     i32 = 180;

const PI: f32 = 3.14159265358979;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(src_texture,  coord, 0);
    let edge  = textureLoad(edge_texture, coord, 0);

    let raw_intensity = edge.r; // edge intensity from pass 1

    var final_intensity: f32;

    if params.stroke_softness < 0.001 {
        // No directional blur: use raw edge intensity directly.
        final_intensity = raw_intensity;
    } else {
        // Directional blur along the gradient angle stored in edge.g.
        // The stored angle points perpendicular to the edge normal, which is
        // the natural direction for pencil strokes running along an edge line.
        let angle = edge.g * (2.0 * PI); // [0, 1] → [0, 2π]
        let dir   = vec2<f32>(cos(angle), sin(angle));

        // Scale blur sigma by stroke_softness so the effect is proportional
        // to the slider. Below the 0.001 threshold (handled above) the blur
        // radius rounds to 0 and we skip the loop entirely.
        let base_sigma   = SIGMA_FRACTION * f32(max(dims.x, dims.y));
        let sigma        = base_sigma * params.stroke_softness;
        let radius       = min(i32(ceil(3.0 * sigma)), RADIUS_CAP);

        if radius == 0 {
            // Sigma rounds down to zero — treat same as no-blur case.
            final_intensity = raw_intensity;
        } else {
            let two_sigma_sq = 2.0 * sigma * sigma;
            var accum:      f32 = 0.0;
            var weight_sum: f32 = 0.0;

            for (var t: i32 = -radius; t <= radius; t++) {
                let offset = vec2<i32>(vec2<f32>(dir) * f32(t));
                let tap    = textureLoad(
                    edge_texture,
                    clamp(coord + offset, vec2<i32>(0), vec2<i32>(dims) - 1),
                    0,
                );
                let w       = exp(-f32(t * t) / two_sigma_sq);
                accum      += tap.r * w;
                weight_sum += w;
            }
            final_intensity = accum / weight_sum;
        }
    }

    // Convert edge intensity to pencil-on-white-paper appearance:
    // strong edge → dark line (near 0.0); low-edge area → white paper (1.0).
    let sketch_value = 1.0 - clamp(final_intensity, 0.0, 1.0);
    let sketch_rgb   = vec3<f32>(sketch_value, sketch_value, sketch_value);

    // Final blend: at strength=0 the output equals the source (identity).
    let out_rgb = mix(src.rgb, sketch_rgb, params.strength);

    // Alpha is preserved from the source image.
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
