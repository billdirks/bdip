use std::path::{Path, PathBuf};
use std::process::Command;

const TEST_TRANSFORMS: &[&str] = &["brightness:0.2", "brightness:0.3"];
const INITIAL_LUMINANCE: u8 = 100;

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

    let cargo_bin = assert_cmd::cargo::cargo_bin("bdip");
    let mut cmd = Command::new(cargo_bin);
    cmd.arg("--headless")
        .arg(&in_path)
        .arg("--output")
        .arg(&out_path);

    for transform in TEST_TRANSFORMS {
        cmd.arg("--apply").arg(transform);
    }

    let output = cmd.output().expect("Failed to execute bdip");
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

    let cargo_bin = assert_cmd::cargo::cargo_bin("bdip");
    let mut cmd = Command::new(cargo_bin);
    cmd.arg("--headless")
        .arg(&in_path)
        .arg("--output")
        .arg(&out_path)
        .arg("--pipeline")
        .arg(&pipeline_path);

    let output = cmd.output().expect("Failed to execute bdip");
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
fn test_cli_headless_without_input_fails() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let out_path = tmp_dir.path().join("should_not_exist.png");

    let cargo_bin = assert_cmd::cargo::cargo_bin("bdip");
    let output = Command::new(cargo_bin)
        .arg("--headless")
        .arg("--output")
        .arg(&out_path)
        .arg("--apply")
        .arg("brightness:0.5")
        .output()
        .expect("Failed to execute bdip");

    assert!(
        !output.status.success(),
        "Headless mode without an input file should fail"
    );
    assert!(!out_path.exists());
}
