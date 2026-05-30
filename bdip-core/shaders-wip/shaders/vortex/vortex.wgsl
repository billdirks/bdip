// Vortex: a radial UV twist where the rotation peaks at a parameterized ring
// distance rather than at the image centre. The envelope is Gaussian-shaped,
// so both the centre and the far edges receive less rotation than the ring,
// producing a whirlpool appearance distinct from Swirl's centre-peaked spiral.
//
// The twist magnitude at distance r from centre is:
//
//   theta(r) = twist * strength * exp(-0.5 * ((r - ring_r) / sigma)^2)
//
// where ring_r = radius_scale * half_diagonal and sigma = ring_r / 2.
// At twist=0.0 or strength=0.0 the transform is an identity.

struct VortexParams {
    // Total rotation at the peak ring, in full turns (1.0 = 360 degrees).
    // 0.0 = identity. Positive = counter-clockwise; negative = clockwise.
    twist:        f32,
    // Normalised distance from centre (in half-diagonal units) at which the
    // twist is strongest. 0.5 = half of the half-diagonal. Identity when 0.
    radius_scale: f32,
    // Blend factor in [0, 1] that scales the full twist envelope.
    // 0.0 = identity (no distortion); 1.0 = full effect.
    strength:     f32,
    _padding:     f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> params: VortexParams;

const TAU: f32 = 6.28318530717958647692;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    let coord = vec2<u32>(global_id.x, global_id.y);
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Early-out: identity when twist or strength is zero.
    if params.twist == 0.0 || params.strength == 0.0 {
        let color = textureLoad(src_texture, coord, 0);
        textureStore(dst_texture, coord, color);
        return;
    }

    // Normalised UV in [0, 1] with half-pixel offset for pixel centres.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Aspect-corrected centred coordinates so that distance is circular.
    let aspect = f32(dims.x) / f32(dims.y);
    let centred = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);

    // Half-diagonal in the aspect-corrected space (longest normalised radius).
    let half_diag = length(vec2<f32>(0.5 * aspect, 0.5));

    let dist = length(centred);

    // Ring radius: the distance at which the twist envelope peaks.
    // When radius_scale is 0 the ring collapses to the centre, treated as
    // identity to avoid a degenerate zero-sigma Gaussian.
    let ring_r = params.radius_scale * half_diag;
    if ring_r <= 0.0 {
        let color = textureLoad(src_texture, coord, 0);
        textureStore(dst_texture, coord, color);
        return;
    }

    // Gaussian envelope centred on ring_r. sigma is set to half the ring
    // radius so the envelope covers a meaningful band without being too narrow.
    // This is a design choice: sigma = ring_r / 2 provides a smooth bell that
    // tapers to near-zero before the image centre and at large radii.
    let sigma = ring_r * 0.5;
    let delta = (dist - ring_r) / sigma;
    let envelope = exp(-0.5 * delta * delta);

    // Rotation angle for this pixel (radians). The turns-to-radians
    // conversion (× TAU) is applied here so the user parameter is in turns.
    let theta = params.twist * TAU * params.strength * envelope;

    // Apply a 2-D rotation matrix to the aspect-corrected centred coordinates.
    let s = sin(theta);
    let c = cos(theta);
    let rotated = vec2<f32>(
        c * centred.x - s * centred.y,
        s * centred.x + c * centred.y,
    );

    // Undo aspect correction and map back to [0, 1] UV space.
    let src_uv = vec2<f32>(rotated.x / aspect + 0.5, rotated.y + 0.5);

    // Pixels that map outside [0, 1] after distortion are filled with opaque
    // black. Clamping is avoided because it would replicate edge pixels into
    // the distorted region, creating smearing artifacts at borders.
    if src_uv.x < 0.0 || src_uv.x > 1.0 || src_uv.y < 0.0 || src_uv.y > 1.0 {
        textureStore(dst_texture, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Convert UV back to integer texture coordinates (nearest-neighbour).
    let src_coord = vec2<i32>(src_uv * vec2<f32>(dims));
    let clamped = vec2<i32>(
        clamp(src_coord.x, 0, i32(dims.x) - 1),
        clamp(src_coord.y, 0, i32(dims.y) - 1),
    );

    let color = textureLoad(src_texture, clamped, 0);
    textureStore(dst_texture, coord, color);
}
