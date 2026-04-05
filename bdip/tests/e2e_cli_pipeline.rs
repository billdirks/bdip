use std::process::Command;

#[test]
fn test_cli_pipeline() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let in_path = tmp_dir.path().join("test_in.png");
    let out_path = tmp_dir.path().join("test_out.png");

    // 1. Write an image.
    let mut img = image::RgbaImage::new(16, 16);
    for pixel in img.pixels_mut() {
        *pixel = image::Rgba([50, 50, 50, 255]);
    }
    img.save(&in_path).unwrap();

    // 2. Spawn CLI executable to test multiple parameters (--apply brightness:-0.2 --apply brightness:0.5)
    let cargo_bin = assert_cmd::cargo::cargo_bin("bdip");
    let mut cmd = Command::new(cargo_bin);
    cmd.arg("--headless")
        .arg(&in_path)
        .arg("--output")
        .arg(&out_path)
        .arg("--apply")
        .arg("brightness:-0.2")
        .arg("--apply")
        .arg("brightness:0.5");

    let output = cmd.output().expect("Failed to execute bdip");
    if !output.status.success() {
        panic!("CLI failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    // 3. Assert output was created.
    assert!(out_path.exists());

    // 4. Verify luminance shift mathematically is larger than 50.
    // Base was 50. -0.2 and +0.5 applied sequentially on 0-255.
    let res = image::open(&out_path).unwrap().into_rgba8();
    let mut total_luminance: u64 = 0;
    for pixel in res.pixels() {
        total_luminance += pixel[0] as u64;
    }
    let avg = total_luminance / (16 * 16);
    assert!(avg > 50, "Average luminance {} was not greater than initial 50", avg);
}
