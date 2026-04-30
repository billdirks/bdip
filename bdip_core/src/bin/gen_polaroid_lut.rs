/// Generates the Polaroid color-grading LUT (`polaroid_64.bin`).
///
/// The file is a 64³ raw f32 RGB cube (R-fastest iteration order, little-endian)
/// in the same format consumed by `AuxAssetFormat::CubeRaw { size: 64 }`. Run once
/// and check in the result; the generator is kept in the repo so the LUT is
/// reproducible and the grade parameters can be audited or re-tuned.
///
/// Usage: `cargo run --bin gen_polaroid_lut -- <output_path>`
/// Default output: `bdip_core/src/gpu/assets/luts/polaroid_64.bin`
fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bdip_core/src/gpu/assets/luts/polaroid_64.bin".to_string());

    const SIZE: u32 = 64;
    let pixel_count = (SIZE * SIZE * SIZE) as usize;
    let mut data: Vec<u8> = Vec::with_capacity(pixel_count * 3 * 4);

    for b_i in 0..SIZE {
        for g_i in 0..SIZE {
            for r_i in 0..SIZE {
                let r = r_i as f32 / (SIZE - 1) as f32;
                let g = g_i as f32 / (SIZE - 1) as f32;
                let b = b_i as f32 / (SIZE - 1) as f32;

                let (out_r, out_g, out_b) = polaroid_grade(r, g, b);

                data.extend_from_slice(&out_r.to_le_bytes());
                data.extend_from_slice(&out_g.to_le_bytes());
                data.extend_from_slice(&out_b.to_le_bytes());
            }
        }
    }

    std::fs::write(&path, &data).unwrap_or_else(|e| panic!("Failed to write {path}: {e}"));
    println!(
        "Wrote {} bytes ({} cells) to {}",
        data.len(),
        pixel_count,
        path
    );
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn s_curve(x: f32, strength: f32) -> f32 {
    // Smooth S-curve fixing both endpoints (x=0→0, x=1→1).
    // Positive strength: darkens lower midtones, lightens upper midtones.
    let delta = strength * x * (1.0 - x) * (2.0 * x - 1.0);
    (x + delta).clamp(0.0, 1.0)
}

/// Applies the Polaroid color grade in sRGB space.
///
/// Characteristics:
///   - Lifted blacks (faded, analog feel)
///   - Gentle S-curve contrast
///   - Warm yellow-orange cast in midtones and highlights
///   - Subtle teal cast in deep shadows
fn polaroid_grade(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    // Shadow lift: raise the black point. Warm lift (R > G > B) echoes the
    // yellow base of Polaroid integral film chemistry.
    let shadow = smoothstep(0.45, 0.0, luma);
    let r = (r + 0.055 * shadow).clamp(0.0, 1.0);
    let g = (g + 0.050 * shadow).clamp(0.0, 1.0);
    let b = (b + 0.045 * shadow).clamp(0.0, 1.0);

    // Gentle S-curve contrast — preserves the faded feel while adding micro-contrast.
    let r = s_curve(r, 0.10);
    let g = s_curve(g, 0.10);
    let b = s_curve(b, 0.10);

    // Recompute luma after lift and contrast.
    let luma2 = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    // Warm cast: peaks in upper midtones (~0.5 luma), fades toward white.
    // Boosts red and green, reduces blue — the characteristic Polaroid warmth.
    let warm = smoothstep(0.15, 0.65, luma2) * (1.0 - smoothstep(0.75, 1.0, luma2));
    let r = (r * (1.0 + 0.070 * warm)).clamp(0.0, 1.0);
    let g = (g * (1.0 + 0.025 * warm)).clamp(0.0, 1.0);
    let b = (b * (1.0 - 0.060 * warm)).clamp(0.0, 1.0);

    // Teal cast in deep shadows: slight cyan/green tint below luma 0.3.
    let cool = smoothstep(0.30, 0.0, luma2);
    let r = (r - 0.015 * cool).clamp(0.0, 1.0);
    let g = (g + 0.005 * cool).clamp(0.0, 1.0);
    let b = (b + 0.020 * cool).clamp(0.0, 1.0);

    (r, g, b)
}
