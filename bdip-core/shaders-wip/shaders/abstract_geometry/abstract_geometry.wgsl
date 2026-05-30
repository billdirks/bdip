// Abstract Geometry shader
//
// Draws a tiled hexagonal grid over the source image. Each cell is flood-filled
// with a noise-derived hue (sampled from the blue-noise texture at the cell
// centre), and the cell boundaries are darkened to form visible edge lines.
//
// The overlay is blended back onto the source with `params.strength` so that
// strength = 0.0 is a pure passthrough (identity).

struct AbstractGeometryParams {
    strength:     f32,
    cell_size:    f32, // hex circumradius at a 1000-px reference width
    edge_width:   f32, // edge half-width as fraction of cell_size
    fill_opacity: f32,
}

@group(0) @binding(0) var src_texture:   texture_2d<f32>;
@group(0) @binding(1) var dst_texture:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: AbstractGeometryParams;
@group(2) @binding(0) var noise_texture: texture_2d<f32>;
@group(2) @binding(1) var noise_sampler: sampler;

// ---------------------------------------------------------------------------
// Hexagonal grid helpers — pointy-top layout
// ---------------------------------------------------------------------------

// Returns the axial (integer) coordinates of the hex cell that contains the
// pixel at `p`, given a circumradius (centre-to-vertex distance) of `r`.
fn hex_cell(p: vec2<f32>, r: f32) -> vec2<i32> {
    // Pointy-top hex: column spacing = sqrt(3)*r, row spacing = 1.5*r.
    let col_spacing = sqrt(3.0) * r;
    let row_spacing = 1.5 * r;

    // Fractional axial coordinates.
    let q = (p.x / col_spacing) - (p.y / row_spacing) * 0.5;
    let s = p.y / row_spacing;

    // Round to nearest hex using cube-coordinate rounding.
    let fq = q;
    let fr = -q - s;
    let fs = s;

    var rq = round(fq);
    var rr = round(fr);
    var rs = round(fs);

    let dq = abs(rq - fq);
    let dr = abs(rr - fr);
    let ds = abs(rs - fs);

    if dq > dr && dq > ds {
        rq = -rr - rs;
    } else if dr > ds {
        rr = -rq - rs;
    } else {
        rs = -rq - rr;
    }

    return vec2<i32>(i32(rq), i32(rs));
}

// Centre of hex cell (aq, as_) in pixel space, given circumradius `r`.
fn hex_centre(aq: i32, as_: i32, r: f32) -> vec2<f32> {
    let col_spacing = sqrt(3.0) * r;
    let row_spacing = 1.5 * r;
    let x = (f32(aq) + f32(as_) * 0.5) * col_spacing;
    let y = f32(as_) * row_spacing;
    return vec2<f32>(x, y);
}

// Smooth signed distance to the boundary of a regular pointy-top hexagon
// centred at the origin with circumradius `r`.  Negative = inside.
fn hex_sdf(p: vec2<f32>, r: f32) -> f32 {
    // iq's analytical hex SDF.
    let k = vec3<f32>(-0.866025404, 0.5, 0.577350269); // cos/sin/tan(30°)
    var q = abs(p);
    q = q - 2.0 * min(dot(k.xy, q), 0.0) * k.xy;
    q = q - vec2<f32>(clamp(q.x, -k.z * r, k.z * r), r);
    return sign(q.y) * length(q);
}

// ---------------------------------------------------------------------------
// Noise-derived cell colour
// ---------------------------------------------------------------------------

// Hash two integers to a pseudo-random float in [0, 1) using a simple
// integer-arithmetic hash.  Used to select which part of the noise texture
// to sample for a given cell, so every cell gets a stable, unique look.
fn cell_hash(aq: i32, as_: i32) -> vec2<f32> {
    var h = u32(aq) * 1664525u + u32(as_) * 22695477u + 1013904223u;
    h ^= h >> 13u;
    h *= 0x9e3779b9u;
    let hx = f32(h & 0xFFFFu) / 65535.0;
    let hy = f32((h >> 16u) & 0xFFFFu) / 65535.0;
    return vec2<f32>(hx, hy);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<u32>(gid.xy);
    let src   = textureLoad(src_texture, coord, 0);

    // Scale cell_size to the actual image width so the pattern is consistent
    // across different image resolutions.
    let r = params.cell_size * (f32(dims.x) / 1000.0);

    let px = vec2<f32>(gid.xy);

    // Identify the hex cell this pixel belongs to.
    let cell = hex_cell(px, r);
    let centre = hex_centre(cell.x, cell.y, r);

    // Distance from this pixel to the hex boundary (negative = inside).
    let d = hex_sdf(px - centre, r);

    // Edge mask: 1.0 right on the boundary (d = 0), falls off symmetrically
    // into the cell interior (d < 0) and the cell exterior (d > 0).
    // Using abs(d) instead of d ensures that deep-interior pixels (large
    // negative d) have edge_mask = 0, not 1.
    let edge_half_px = params.edge_width * r;
    let edge_mask = 1.0 - smoothstep(0.0, max(edge_half_px, 0.5), abs(d));

    // Noise-derived hue for this cell: sample the blue-noise texture at a
    // position derived from the cell index so each cell gets a unique value.
    let uv = fract(cell_hash(cell.x, cell.y));
    let noise_rgb = textureSampleLevel(noise_texture, noise_sampler, uv, 0.0).rgb;

    // Convert the noise sample to a vivid hue by boosting saturation.
    // We use a simple YCbCr-style boost: subtract luma, scale chroma, add back.
    let luma  = dot(noise_rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let chroma = noise_rgb - vec3<f32>(luma);
    let hue_rgb = clamp(vec3<f32>(luma) + chroma * 3.0, vec3<f32>(0.0), vec3<f32>(1.0));

    // Composite: fill the cell interior with hue_rgb at fill_opacity,
    // then darken the edges to black for the grid lines.
    // Inside the hex (d < 0) the cell colour is blended; at the boundary it
    // transitions to black edge lines.
    let inside = clamp(-d / max(r * 0.01, 0.001), 0.0, 1.0); // 1 deep inside, 0 at/outside boundary
    let cell_fill   = mix(src.rgb, hue_rgb, params.fill_opacity * inside);
    let with_edges  = mix(cell_fill, vec3<f32>(0.0), edge_mask);

    // Final blend with source controlled by strength.
    let out_rgb = mix(src.rgb, with_edges, params.strength);
    textureStore(dst_texture, coord, vec4<f32>(out_rgb, src.a));
}
