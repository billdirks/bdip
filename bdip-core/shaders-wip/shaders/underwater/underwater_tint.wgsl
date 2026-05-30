// Underwater — tint pass (pass 1 of 2).
//
// Applies a blue/teal color shift with partial desaturation to simulate being
// submerged in water. The output is written to a scratch texture; the caustic
// pass (pass 2) reads this alongside the original source to produce the final
// composite.
//
// Effect formula:
//   1. Convert pixel to luminance (desaturated gray).
//   2. Mix original color toward a blue/teal underwater color target based on
//      `depth`. The target keeps some green to produce a teal rather than pure
//      blue, matching shallow to mid-water appearance.
//   3. The tinted color is stored as-is; the blend against the original source
//      happens in the caustic pass using `strength`.
//
// When `depth=0.0` the mix factor is zero and the tinted output equals the
// source, making this pass a no-op regardless of `strength`.
//
// All Underwater WGSL files declare the full UnderwaterParams struct to satisfy
// WebGPU's uniform binding-size validation (see specs/adding_a_shader.md
// § "Shared-uniform alignment rule").

struct UnderwaterParams {
    depth:             f32,
    caustic_intensity: f32,
    strength:          f32,
    _padding:          f32,
}

@group(0) @binding(0) var input_texture:  texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: UnderwaterParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let src   = textureLoad(input_texture, coord, 0);

    // Luminance (Rec. 709 coefficients) for desaturation.
    let lum = dot(src.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Underwater color target: teal-blue, preserving some luminance so deep
    // pixels aren't crushed to black. The channel weights were chosen to
    // produce a mid-depth ocean teal at depth=0.5.
    //   R: strongly attenuated (water absorbs red first)
    //   G: lightly reduced (water absorbs green less)
    //   B: boosted relative to luminance (water transmits blue deepest)
    let water_rgb = vec3<f32>(
        lum * 0.35,
        lum * 0.70,
        lum * 1.10,
    );

    // Mix the original color toward the underwater color target based on depth.
    // At depth=0 the mix factor is 0 → output equals source (no tint).
    // At depth=1 the mix factor is 1 → output is the pure water color.
    let tinted = mix(src.rgb, water_rgb, params.depth);

    textureStore(output_texture, coord, vec4<f32>(tinted, src.a));
}
