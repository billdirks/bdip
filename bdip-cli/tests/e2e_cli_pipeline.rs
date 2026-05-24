use std::path::{Path, PathBuf};
use std::process::Command;

const TEST_TRANSFORMS: &[&str] = &["brightness:0.2", "brightness:0.3"];
const INITIAL_LUMINANCE: u8 = 100;

/// Returns true if at least one pixel in `path` differs from `[r, g, b, a]`.
fn any_pixel_differs(path: &Path, r: u8, g: u8, b: u8, a: u8) -> bool {
    let img = image::open(path)
        .expect("Failed to open output image")
        .into_rgba8();
    img.pixels()
        .any(|p| p[0] != r || p[1] != g || p[2] != b || p[3] != a)
}

fn setup_test_image(tmp_dir: &tempfile::TempDir) -> PathBuf {
    let in_path = tmp_dir.path().join("test_in.png");
    let mut img = image::RgbaImage::new(16, 16);
    for pixel in img.pixels_mut() {
        *pixel = image::Rgba([INITIAL_LUMINANCE, INITIAL_LUMINANCE, INITIAL_LUMINANCE, 255]);
    }
    img.save(&in_path).unwrap();
    in_path
}

fn get_avg_luminance(path: &Path) -> u64 {
    let res = image::open(path)
        .expect("Failed to open output image")
        .into_rgba8();
    let mut total_luminance: u64 = 0;
    for pixel in res.pixels() {
        total_luminance += pixel[0] as u64;
    }
    total_luminance / (res.width() as u64 * res.height() as u64)
}

#[test]
fn test_cli_apply_flow() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let in_path = setup_test_image(&tmp_dir);
    let out_path = tmp_dir.path().join("test_out_apply.png");

    let cargo_bin = assert_cmd::cargo::cargo_bin("bdip-cli");
    let mut cmd = Command::new(cargo_bin);
    cmd.arg(&in_path).arg("--output").arg(&out_path);

    for transform in TEST_TRANSFORMS {
        cmd.arg("--apply").arg(transform);
    }

    let output = cmd.output().expect("Failed to execute bdip-cli");
    if !output.status.success() {
        panic!(
            "CLI --apply failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    assert!(out_path.exists());
    let avg = get_avg_luminance(&out_path);
    assert!(
        avg > INITIAL_LUMINANCE as u64,
        "Luminance {} should be > {}",
        avg,
        INITIAL_LUMINANCE
    );
}

#[test]
fn test_cli_pipeline_flow() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let in_path = setup_test_image(&tmp_dir);
    let out_path = tmp_dir.path().join("test_out_pipeline.png");
    let pipeline_path = tmp_dir.path().join("pipeline.txt");

    // Write the transforms to a pipeline file
    std::fs::write(&pipeline_path, TEST_TRANSFORMS.join("\n")).unwrap();

    let cargo_bin = assert_cmd::cargo::cargo_bin("bdip-cli");
    let mut cmd = Command::new(cargo_bin);
    cmd.arg(&in_path)
        .arg("--output")
        .arg(&out_path)
        .arg("--pipeline")
        .arg(&pipeline_path);

    let output = cmd.output().expect("Failed to execute bdip-cli");
    if !output.status.success() {
        panic!(
            "CLI --pipeline failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    assert!(out_path.exists());
    let avg = get_avg_luminance(&out_path);
    assert!(
        avg > INITIAL_LUMINANCE as u64,
        "Luminance {} should be > {}",
        avg,
        INITIAL_LUMINANCE
    );
}

#[test]
fn test_cli_missing_input_fails() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let out_path = tmp_dir.path().join("should_not_exist.png");

    let cargo_bin = assert_cmd::cargo::cargo_bin("bdip-cli");
    let output = Command::new(cargo_bin)
        .arg("--output")
        .arg(&out_path)
        .arg("--apply")
        .arg("brightness:0.5")
        .output()
        .expect("Failed to execute bdip-cli");

    assert!(
        !output.status.success(),
        "Invocation without a required input file should fail"
    );
    assert!(!out_path.exists());
}

#[test]
fn test_cli_16bit_precision_preserved() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let in_path = tmp_dir.path().join("test_in_16bit.png");
    let mut img = bdip_core::Rgba16Image::new(16, 16);
    for pixel in img.pixels_mut() {
        // Value 32768 would be downsampled to 128 in 8-bit, which upsamples back to 32896.
        *pixel = image::Rgba([32768, 32768, 32768, 65535]);
    }
    img.save(&in_path).unwrap();

    let out_path = tmp_dir.path().join("test_out_16bit.png");

    let cargo_bin = assert_cmd::cargo::cargo_bin("bdip-cli");
    let mut cmd = Command::new(cargo_bin);
    cmd.arg(&in_path)
        .arg("--output")
        .arg(&out_path)
        .arg("--apply")
        .arg("brightness:0.0"); // Identity transform

    let output = cmd.output().expect("Failed to execute bdip-cli");
    if !output.status.success() {
        panic!(
            "CLI --apply failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    assert!(out_path.exists());
    let res = image::open(&out_path)
        .expect("Failed to open output image")
        .into_rgba16();
    let avg_r =
        res.pixels().map(|p| p[0] as u64).sum::<u64>() / (res.width() * res.height()) as u64;

    // Allow precision loss from f16 conversion and the sRGB round-trip (ingest + present),
    // but must clearly be closer to 32768 than 32896.
    assert!(
        (avg_r as i64 - 32768_i64).abs() < 128,
        "Expected ~32768, got {}",
        avg_r
    );
}

#[test]
fn test_headless_multi_apply() {
    let tmp_dir = tempfile::tempdir().unwrap();

    // Use a saturated red image so that the saturation transform has a visible effect.
    let in_path = tmp_dir.path().join("test_in_color.png");
    let mut img = image::RgbaImage::new(16, 16);
    for pixel in img.pixels_mut() {
        *pixel = image::Rgba([200, 80, 80, 255]);
    }
    img.save(&in_path).unwrap();

    let out_path = tmp_dir.path().join("test_out_multi.png");

    let cargo_bin = assert_cmd::cargo::cargo_bin("bdip-cli");
    let output = Command::new(cargo_bin)
        .arg(&in_path)
        .arg("--output")
        .arg(&out_path)
        .arg("--apply")
        .arg("brightness:0.3")
        .arg("--apply")
        .arg("saturation:-0.5")
        .output()
        .expect("Failed to execute bdip-cli");

    if !output.status.success() {
        panic!(
            "CLI multi-apply failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    assert!(out_path.exists());
    // The output must differ from the original [200, 80, 80, 255] pixels — both brightness
    // and saturation transforms alter the values.
    assert!(
        any_pixel_differs(&out_path, 200, 80, 80, 255),
        "Output image should differ from the input after applying brightness + saturation"
    );
}
