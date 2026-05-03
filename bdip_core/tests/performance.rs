//! GPU-pipeline performance benchmarks at 24 MP — the primary benchmark target from
//! `specs/perf_goal_1.md`.
//!
//! These live as an integration test (separate compile target) so they don't bloat
//! the unit-test build and don't need `#[ignore]` to stay out of `cargo test`.
//! Run them explicitly via `cargo perf-test` (alias for
//! `cargo test --release -p bdip_core --test performance ...`).
//!
//! See `bdip_core/src/gpu/image_pipeline.rs::PhaseTiming` for the three-bucket
//! timing model the helpers below produce. Briefly: `execute` is CPU encode +
//! submit; `gpu_wait` is the wall-clock time blocked in `device.poll(Wait)` after
//! submission (i.e. the true GPU compute time); `readback` is the download path
//! only (copy + map + memcpy).

use bdip_core::Rgba16Image;
use bdip_core::Transform;
use bdip_core::gpu::engine::GpuEngine;
use bdip_core::gpu::image_pipeline::{PassTiming, Renderer};
use bdip_core::gpu::shaders::registry_by_id;
use bdip_core::gpu::texture::upload_texture;
use bdip_core::wgpu;

// Inlined here rather than imported from the unit-test-only `gpu::test_util`
// module, which is `#[cfg(test)]` and therefore invisible to integration tests.
// Five lines of synthetic image generation isn't worth promoting `test_util` to
// the public surface.
fn make_solid_image(w: u32, h: u32, r: u16, g: u16, b: u16) -> Rgba16Image {
    let mut img = Rgba16Image::new(w, h);
    for pixel in img.pixels_mut() {
        *pixel = image::Rgba([r, g, b, 65535]);
    }
    img
}

/// Timing for a single cold-or-warm run of the benchmark pipeline. See module
/// docs for the rationale behind splitting into three buckets (without it, GPU
/// shader time hides inside the readback timer because `download_slice` polls
/// the device).
#[derive(Copy, Clone)]
struct PhaseTiming {
    execute_ms: f64,
    gpu_wait_ms: f64,
    readback_ms: f64,
}

impl PhaseTiming {
    fn critical_path_ms(&self) -> f64 {
        self.execute_ms + self.gpu_wait_ms + self.readback_ms
    }
}

/// Result of a cold+warm benchmark run. Includes wall-clock phase timings
/// and, when the adapter supports `TIMESTAMP_QUERY`, per-pass GPU durations.
struct BenchResult {
    cold: PhaseTiming,
    warm: PhaseTiming,
    cold_pass_timings: Vec<PassTiming>,
    warm_pass_timings: Vec<PassTiming>,
}

/// Runs the cold+warm benchmark pipeline for a given shader and returns
/// per-phase timings plus per-pass GPU timestamps. Cold runs
/// ingest+apply+present+download. Warm reuses the ingested texture and
/// downloads via `download_slice` to match the interactive-editing path.
///
/// GPU timestamp collection adds ~3-5 ms of overhead per call (query set
/// allocation, resolve, synchronous poll + map for readback). This inflates
/// the wall-clock `execute_ms` relative to the untimed `apply` path. The
/// per-pass durations themselves are measured by the GPU and are unaffected.
fn bench_shader_roundtrip(
    engine: &GpuEngine,
    renderer: &mut Renderer,
    uploaded: &wgpu::Texture,
    width: u32,
    height: u32,
    transform: &Transform,
) -> BenchResult {
    use std::time::Instant;

    let use_timestamps = engine.supports_timestamps();

    let wait_for_gpu = || {
        let t = Instant::now();
        engine
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        t.elapsed().as_secs_f64() * 1000.0
    };

    // --- Run 1: cold (shader compilation + initial buffer allocation) ---
    let t_execute = Instant::now();
    let ingested = renderer.ingest(engine, uploaded);
    let (transformed, cold_pass_timings) = if use_timestamps {
        renderer
            .apply_with_timestamps(engine, &ingested, transform)
            .unwrap()
    } else {
        (
            renderer.apply(engine, &ingested, transform).unwrap(),
            Vec::new(),
        )
    };
    let present_buf = renderer.present(engine, &transformed);
    let cold = {
        let execute_ms = t_execute.elapsed().as_secs_f64() * 1000.0;
        let gpu_wait_ms = wait_for_gpu();

        let t_readback = Instant::now();
        renderer
            .download(engine, &present_buf, width, height)
            .unwrap();
        let readback_ms = t_readback.elapsed().as_secs_f64() * 1000.0;

        PhaseTiming {
            execute_ms,
            gpu_wait_ms,
            readback_ms,
        }
    };

    // --- Run 2: warm (pipelines compiled, staging buffer + pixel_vec cached) ---
    let t_execute = Instant::now();
    let (transformed, warm_pass_timings) = if use_timestamps {
        renderer
            .apply_with_timestamps(engine, &ingested, transform)
            .unwrap()
    } else {
        (
            renderer.apply(engine, &ingested, transform).unwrap(),
            Vec::new(),
        )
    };
    let present_buf = renderer.present(engine, &transformed);
    let warm = {
        let execute_ms = t_execute.elapsed().as_secs_f64() * 1000.0;
        let gpu_wait_ms = wait_for_gpu();

        let t_readback = Instant::now();
        renderer
            .download_slice(engine, &present_buf, width, height)
            .unwrap();
        let readback_ms = t_readback.elapsed().as_secs_f64() * 1000.0;

        PhaseTiming {
            execute_ms,
            gpu_wait_ms,
            readback_ms,
        }
    };

    BenchResult {
        cold,
        warm,
        cold_pass_timings,
        warm_pass_timings,
    }
}

/// Prints a uniform perf report for a shader's cold+warm run. `label` and
/// `pass_count` are pulled from the shader registration so the header shows,
/// e.g., "Clarity (3 passes)".
fn print_perf_report(label: &str, pass_count: usize, result: &BenchResult, warm_target_ms: f64) {
    let cold = &result.cold;
    let warm = &result.warm;
    let header = format!("--- 24 MP GPU roundtrip — {label} ({pass_count} passes) ---");
    eprintln!("{header}");
    eprintln!(
        "  run 1 execute  (cpu encode+submit):     {:>8.2} ms",
        cold.execute_ms
    );
    eprintln!(
        "  run 1 gpu wait (ingest+apply+present):  {:>8.2} ms",
        cold.gpu_wait_ms
    );
    eprintln!(
        "  run 1 readback (copy+map+memcpy):       {:>8.2} ms",
        cold.readback_ms
    );
    eprintln!(
        "  run 1 critical path:                    {:>8.2} ms",
        cold.critical_path_ms()
    );
    print_pass_timings("  run 1", &result.cold_pass_timings);
    eprintln!(
        "  run 2 execute  (cpu encode+submit):     {:>8.2} ms",
        warm.execute_ms
    );
    eprintln!(
        "  run 2 gpu wait (apply+present):         {:>8.2} ms",
        warm.gpu_wait_ms
    );
    eprintln!(
        "  run 2 readback (copy+map+memcpy):       {:>8.2} ms",
        warm.readback_ms
    );
    eprintln!(
        "  run 2 critical path:                    {:>8.2} ms  (target: <{:.0} ms warm)",
        warm.critical_path_ms(),
        warm_target_ms
    );
    print_pass_timings("  run 2", &result.warm_pass_timings);
    eprintln!("{}", "-".repeat(header.len()));
}

fn print_pass_timings(prefix: &str, timings: &[PassTiming]) {
    if timings.is_empty() {
        return;
    }
    for t in timings {
        let ms = t.duration_ns / 1_000_000.0;
        eprintln!("{prefix} gpu pass {:<20} {:>8.2} ms", t.label, ms);
    }
    let total_ms: f64 = timings.iter().map(|t| t.duration_ns).sum::<f64>() / 1_000_000.0;
    eprintln!(
        "{prefix} gpu passes total:              {:>8.2} ms",
        total_ms
    );
}

/// Returns the display name and pass count registered for `shader_id`. Used
/// by the perf tests to keep report headers in sync with the registry.
fn shader_display_info(shader_id: &'static str) -> (&'static str, usize) {
    let reg =
        registry_by_id(shader_id).unwrap_or_else(|| panic!("Unknown shader ID: '{shader_id}'"));
    (reg.meta.display_name, reg.meta.passes.len())
}

const PERF_WIDTH: u32 = 5000;
const PERF_HEIGHT: u32 = 4800;
// There is some pipeline overhead associated with timing shader passes
// so this is 5ms longer than our target.
const PERF_WARM_TARGET_MS: f64 = 30.0;
const PERF_COLD_TARGET_MS: f64 = 80.0;

/// Times the GPU-critical path on a 24 MP synthetic image — the primary target
/// size from perf_goal_1.md. Uses the single-pass `brightness` shader to
/// exercise the ingest → apply → present → download path with minimal shader
/// cost, and asserts both the upload budget and cold/warm critical paths.
///
/// Two runs isolate warm-pipeline performance from one-time startup costs.
/// See `PhaseTiming` (in `bdip_core::gpu::image_pipeline`) for what each timing
/// bucket measures.
#[test]
fn perf_gpu_roundtrip_24mp() {
    use std::time::Instant;

    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    // 5000×4800 = 24,000,000 pixels (~24 MP), matching the primary benchmark
    // target in perf_goal_1.md. Generated synthetically — no test asset needed.
    let img = make_solid_image(PERF_WIDTH, PERF_HEIGHT, 32767, 32767, 32767);

    let t_upload = Instant::now();
    let uploaded = upload_texture(&engine.device, &engine.queue, &img);
    let upload_ms = t_upload.elapsed().as_secs_f64() * 1000.0;

    let transform = Transform {
        shader_id: "brightness",
        values: vec![0.1],
    };
    let result = bench_shader_roundtrip(
        &engine,
        &mut renderer,
        &uploaded,
        img.width(),
        img.height(),
        &transform,
    );

    let (label, pass_count) = shader_display_info(transform.shader_id);
    eprintln!("  gpu upload:                             {upload_ms:>8.2} ms");
    print_perf_report(label, pass_count, &result, PERF_WARM_TARGET_MS);

    assert!(
        upload_ms < PERF_WARM_TARGET_MS,
        "Upload time exceeded {PERF_WARM_TARGET_MS:.0}ms target: {upload_ms:.2}ms"
    );
    assert!(
        result.cold.critical_path_ms() < PERF_COLD_TARGET_MS,
        "Run 1 (cold) critical path exceeded {PERF_COLD_TARGET_MS:.0}ms target: {:.2}ms",
        result.cold.critical_path_ms()
    );
    assert!(
        result.warm.critical_path_ms() < PERF_WARM_TARGET_MS,
        "Run 2 (warm) critical path exceeded {PERF_WARM_TARGET_MS:.0}ms target: {:.2}ms",
        result.warm.critical_path_ms()
    );
}

/// Times the GPU critical path on a 24 MP image with the Clarity multi-pass
/// shader (blur_h, blur_v, combine). See `perf_gpu_roundtrip_24mp` and
/// `PhaseTiming` for the measurement model.
#[test]
fn perf_gpu_roundtrip_24mp_clarity() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(PERF_WIDTH, PERF_HEIGHT, 32767, 32767, 32767);
    let uploaded = upload_texture(&engine.device, &engine.queue, &img);

    let transform = Transform {
        shader_id: "clarity",
        values: vec![0.5],
    };
    let result = bench_shader_roundtrip(
        &engine,
        &mut renderer,
        &uploaded,
        img.width(),
        img.height(),
        &transform,
    );

    let (label, pass_count) = shader_display_info(transform.shader_id);
    print_perf_report(label, pass_count, &result, PERF_WARM_TARGET_MS);

    assert!(
        result.warm.critical_path_ms() < PERF_WARM_TARGET_MS,
        "{label} warm critical path exceeded {PERF_WARM_TARGET_MS:.0} ms target: {:.2} ms",
        result.warm.critical_path_ms()
    );
}

/// Times the GPU critical path on a 24 MP image with the Comic Book multi-pass
/// shader (edges, halftone with aux texture, combine). The halftone pass uses a
/// nearest-neighbor auxiliary texture, validating Group 2 bind group setup cost
/// and aux cache-hit behavior on the warm run.
#[test]
fn perf_gpu_roundtrip_24mp_comic_book() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(PERF_WIDTH, PERF_HEIGHT, 32767, 32767, 32767);
    let uploaded = upload_texture(&engine.device, &engine.queue, &img);

    let transform = Transform {
        shader_id: "comic_book",
        values: vec![1.0f32, 16.0, 0.10, 0.15],
    };
    let result = bench_shader_roundtrip(
        &engine,
        &mut renderer,
        &uploaded,
        img.width(),
        img.height(),
        &transform,
    );

    let (label, pass_count) = shader_display_info(transform.shader_id);
    // Comic Book runs 3 full-scale passes on 24 MP (edges, halftone, combine), making it
    // the most GPU-intensive shader in the current plan. Rather than asserting a fixed wall
    // time (which would be hardware-dependent), this test benchmarks Group 2 bind group
    // setup cost and aux cache-hit behavior on the warm run. Inspect the printed timings
    // to track regressions across hardware or changes.
    print_perf_report(label, pass_count, &result, PERF_WARM_TARGET_MS);

    assert!(
        result.warm.critical_path_ms() < PERF_WARM_TARGET_MS,
        "{label} warm critical path exceeded {PERF_WARM_TARGET_MS:.0} ms target: {:.2} ms",
        result.warm.critical_path_ms()
    );
}

/// Times the GPU critical path on a 24 MP image with the Color LUT single-pass
/// shader. Uses the identity LUT at full intensity to benchmark the Group 2
/// bind group setup cost and the `get_or_upload` cache-hit path on the warm run.
/// This is the first shader in the plan that exercises the 3D texture pipeline.
#[test]
fn perf_gpu_roundtrip_24mp_color_lut() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(PERF_WIDTH, PERF_HEIGHT, 32767, 32767, 32767);
    let uploaded = upload_texture(&engine.device, &engine.queue, &img);

    let transform = Transform {
        shader_id: "color_lut",
        values: vec![1.0f32],
    };
    let result = bench_shader_roundtrip(
        &engine,
        &mut renderer,
        &uploaded,
        img.width(),
        img.height(),
        &transform,
    );

    let (label, pass_count) = shader_display_info(transform.shader_id);
    print_perf_report(label, pass_count, &result, PERF_WARM_TARGET_MS);

    assert!(
        result.warm.critical_path_ms() < PERF_WARM_TARGET_MS,
        "{label} warm critical path exceeded {PERF_WARM_TARGET_MS:.0} ms target: {:.2} ms",
        result.warm.critical_path_ms()
    );
}

/// Times the GPU critical path on a 24 MP image with the Cartoon multi-pass
/// shader (smooth_h, smooth_v, quantize, edges, combine). See
/// `perf_gpu_roundtrip_24mp` and `PhaseTiming` for the measurement model.
#[test]
fn perf_gpu_roundtrip_24mp_cartoon() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(PERF_WIDTH, PERF_HEIGHT, 32767, 32767, 32767);
    let uploaded = upload_texture(&engine.device, &engine.queue, &img);

    // Cartoon default params: strength=0.0, levels=8.0, edge_threshold=0.15,
    // edge_softness=0.10, edge_darkness=1.0. Defaults exercise the full 5-pass
    // pipeline under realistic conditions without assuming specific output.
    let transform = Transform {
        shader_id: "cartoon",
        values: vec![0.0f32, 8.0, 0.15, 0.10, 1.0],
    };
    let result = bench_shader_roundtrip(
        &engine,
        &mut renderer,
        &uploaded,
        img.width(),
        img.height(),
        &transform,
    );

    let (label, pass_count) = shader_display_info(transform.shader_id);
    print_perf_report(label, pass_count, &result, PERF_WARM_TARGET_MS);

    assert!(
        result.warm.critical_path_ms() < PERF_WARM_TARGET_MS,
        "{label} warm critical path exceeded {PERF_WARM_TARGET_MS:.0} ms target: {:.2} ms",
        result.warm.critical_path_ms()
    );
}

/// Times the GPU critical path on a 24 MP image with the Pop Art multi-pass
/// shader (quantize, colorize, combine). All three passes run at full resolution,
/// making this a useful data point for tracking 3-pass full-scale pipeline cost.
#[test]
fn perf_gpu_roundtrip_24mp_pop_art() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(PERF_WIDTH, PERF_HEIGHT, 32767, 32767, 32767);
    let uploaded = upload_texture(&engine.device, &engine.queue, &img);

    let transform = Transform {
        shader_id: "pop_art",
        values: vec![1.0f32, 4.0, 12.0],
    };
    let result = bench_shader_roundtrip(
        &engine,
        &mut renderer,
        &uploaded,
        img.width(),
        img.height(),
        &transform,
    );

    let (label, pass_count) = shader_display_info(transform.shader_id);
    print_perf_report(label, pass_count, &result, PERF_WARM_TARGET_MS);

    assert!(
        result.warm.critical_path_ms() < PERF_WARM_TARGET_MS,
        "{label} warm critical path exceeded {PERF_WARM_TARGET_MS:.0} ms target: {:.2} ms",
        result.warm.critical_path_ms()
    );
}

/// Times the GPU critical path on a 24 MP image with the Tilt-Shift multi-pass
/// shader (down, blur_h, blur_v, up, composite). The separable Gaussian passes run
/// at 4× downsampled resolution, making this a data point for the cost of the
/// downsample→blur→upsample strategy plus a masked-blend composite at 24 MP.
#[test]
fn perf_gpu_roundtrip_24mp_tilt_shift() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(PERF_WIDTH, PERF_HEIGHT, 32767, 32767, 32767);
    let uploaded = upload_texture(&engine.device, &engine.queue, &img);

    // focus_center=0.5, focus_width=0.3, blur_strength=1.0: the top and bottom
    // 35% of the image are fully blurred, exercising the maximum kernel radius.
    let transform = Transform {
        shader_id: "tilt_shift",
        values: vec![0.5f32, 0.3, 1.0],
    };
    let result = bench_shader_roundtrip(
        &engine,
        &mut renderer,
        &uploaded,
        img.width(),
        img.height(),
        &transform,
    );

    let (label, pass_count) = shader_display_info(transform.shader_id);
    print_perf_report(label, pass_count, &result, PERF_WARM_TARGET_MS);

    assert!(
        result.warm.critical_path_ms() < PERF_WARM_TARGET_MS,
        "{label} warm critical path exceeded {PERF_WARM_TARGET_MS:.0} ms target: {:.2} ms",
        result.warm.critical_path_ms()
    );
}

/// Times the GPU critical path on a 24 MP image with the Bokeh Shapes multi-pass
/// shader (polygon blur pass + blend pass). The blur pass iterates an integer-offset
/// kernel up to 50 px in radius, making this a data point for the cost of a dense
/// per-pixel gather kernel at full resolution.
#[test]
fn perf_gpu_roundtrip_24mp_bokeh_shapes() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(PERF_WIDTH, PERF_HEIGHT, 32767, 32767, 32767);
    let uploaded = upload_texture(&engine.device, &engine.queue, &img);

    // radius=20, sides=6 (hexagon), strength=1.0: exercises a medium-radius
    // hexagonal kernel at full blend strength.
    let transform = Transform {
        shader_id: "bokeh_shapes",
        values: vec![20.0f32, 6.0, 1.0],
    };
    let result = bench_shader_roundtrip(
        &engine,
        &mut renderer,
        &uploaded,
        img.width(),
        img.height(),
        &transform,
    );

    let (label, pass_count) = shader_display_info(transform.shader_id);
    print_perf_report(label, pass_count, &result, PERF_WARM_TARGET_MS);

    assert!(
        result.warm.critical_path_ms() < PERF_WARM_TARGET_MS,
        "{label} warm critical path exceeded {PERF_WARM_TARGET_MS:.0} ms target: {:.2} ms",
        result.warm.critical_path_ms()
    );
}

/// Times the GPU critical path on a 24 MP image with the Polaroid multi-pass
/// shader (grade pass with 3D LUT aux texture, then border pass). Exercises
/// the 3D LUT cache-hit path alongside the scratch texture pool.
#[test]
fn perf_gpu_roundtrip_24mp_polaroid() {
    let engine = GpuEngine::new().unwrap();
    let mut renderer = Renderer::new(&engine);

    let img = make_solid_image(PERF_WIDTH, PERF_HEIGHT, 32767, 32767, 32767);
    let uploaded = upload_texture(&engine.device, &engine.queue, &img);

    let transform = Transform {
        shader_id: "polaroid",
        values: vec![1.0f32, 1.0],
    };
    let result = bench_shader_roundtrip(
        &engine,
        &mut renderer,
        &uploaded,
        img.width(),
        img.height(),
        &transform,
    );

    let (label, pass_count) = shader_display_info(transform.shader_id);
    print_perf_report(label, pass_count, &result, PERF_WARM_TARGET_MS);

    assert!(
        result.warm.critical_path_ms() < PERF_WARM_TARGET_MS,
        "{label} warm critical path exceeded {PERF_WARM_TARGET_MS:.0} ms target: {:.2} ms",
        result.warm.critical_path_ms()
    );
}
