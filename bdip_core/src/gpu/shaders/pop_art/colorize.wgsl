// Pop Art — colorize pass.
//
// Maps the quantized tonal levels from the previous pass to vibrant pop-art
// hues. Each discrete level is assigned a distinct hue by dividing the color
// wheel evenly. Saturation is fixed at 1.0 for maximum vibrancy; lightness
// rises linearly with level so the darkest level is near-black and the
// brightest is vivid. The result is the flat, high-contrast palette associated
// with Andy Warhol-style silkscreen printing.
//
// All Pop Art WGSL files declare the full PopArtParams struct to satisfy
// WebGPU's uniform binding-size validation.

struct PopArtParams {
    strength:  f32,
    levels:    f32,
    dot_scale: f32,
    _padding:  f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: PopArtParams;

// Converts a hue in [0, 1) to an RGB triplet using the standard piecewise
// linear model — no branching, works in linear-light space.
fn hue_to_rgb(h: f32) -> vec3<f32> {
    let r = abs(h * 6.0 - 3.0) - 1.0;
    let g = 2.0 - abs(h * 6.0 - 2.0);
    let b = 2.0 - abs(h * 6.0 - 4.0);
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Full HSL → linear-RGB conversion (saturation=1 collapses to pure hue_to_rgb
// at lightness=0.5, but the formula handles arbitrary s and l correctly).
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    let rgb = hue_to_rgb(h);
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    return (rgb - 0.5) * c + l;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord    = vec2<i32>(gid.xy);
    let quantized = textureLoad(input_texture, coord, 0);

    let L    = clamp(params.levels, 2.0, 8.0);
    let luma = dot(quantized.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Map luma to the nearest discrete level index in [0, L-1].
    let level = round(luma * (L - 1.0));

    // Distribute hues evenly around the color wheel. A small phase offset (0.05)
    // prevents the lowest level from landing on pure red, which reads as harsh.
    let hue = fract(level / L + 0.05);

    // Lightness from near-black at level 0 to a vivid mid-tone at level L-1.
    // Capping at 0.65 keeps all levels visibly saturated (L=1 at HSL = white).
    let lightness = mix(0.08, 0.65, level / (L - 1.0));

    let pop_color = hsl_to_rgb(hue, 1.0, lightness);

    textureStore(output_texture, coord, vec4<f32>(pop_color, quantized.a));
}
