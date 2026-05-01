@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;

struct SelectiveColorParams {
    // Target hue in degrees [0, 360).
    target_hue: f32,
    // Half-width of the hue selection window in degrees (0 = no selection, 180 = full).
    tolerance: f32,
    // Width of the smooth falloff region in degrees.
    feather: f32,
    _padding: f32,
}

@group(1) @binding(0) var<uniform> params: SelectiveColorParams;

// Convert linear-light RGB to HSV.
// H is in [0, 360), S in [0, 1], V in [0, 1].
fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    let r = rgb.r;
    let g = rgb.g;
    let b = rgb.b;
    let cmax = max(r, max(g, b));
    let cmin = min(r, min(g, b));
    let delta = cmax - cmin;

    var h: f32 = 0.0;
    if delta > 0.0001 {
        if cmax == r {
            h = 60.0 * (((g - b) / delta) % 6.0);
        } else if cmax == g {
            h = 60.0 * ((b - r) / delta + 2.0);
        } else {
            h = 60.0 * ((r - g) / delta + 4.0);
        }
    }
    // Ensure H is in [0, 360).
    if h < 0.0 {
        h = h + 360.0;
    }

    let s = select(0.0, delta / cmax, cmax > 0.0001);
    let v = cmax;
    return vec3<f32>(h, s, v);
}

// Compute the shortest angular distance between two hues in degrees.
// Result is always in [0, 180].
fn hue_distance(a: f32, b: f32) -> f32 {
    let diff = abs(a - b) % 360.0;
    return min(diff, 360.0 - diff);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coords = vec2<u32>(global_id.x, global_id.y);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(src_texture, coords, 0);

    // Rec. 709 luminance (linear light).
    let luminance = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let gray = vec3<f32>(luminance);

    // When tolerance is 0 the shader is an identity (everything desaturated, target
    // hue window has zero width so no pixel survives); however the registration
    // default is tolerance=180 to produce a full-color pass-through identity.
    let hsv = rgb_to_hsv(color.rgb);
    let dist = hue_distance(hsv.x, params.target_hue);

    // Color retention weight:
    //   dist <= tolerance               → 1.0 (full color)
    //   tolerance < dist <= tolerance+feather → smooth falloff
    //   dist > tolerance+feather        → 0.0 (full grayscale)
    let inner_edge = params.tolerance;
    let outer_edge = params.tolerance + max(params.feather, 0.001);
    let color_weight = 1.0 - smoothstep(inner_edge, outer_edge, dist);

    let final_rgb = mix(gray, color.rgb, color_weight);
    textureStore(dst_texture, coords, vec4<f32>(final_rgb, color.a));
}
