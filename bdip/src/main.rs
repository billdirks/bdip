use bdip_core::gpu::shaders::{ParamKind, Transform, all_registrations, registry_by_id};
use bdip_core::gpu::texture::{download_presentation_buffer, upload_texture};
use bdip_core::gpu::{engine::GpuEngine, pipeline::Renderer};
use clap::Parser;

mod cli;
mod timing;
mod ui;

fn parse_transform(s: &str) -> anyhow::Result<Transform> {
    let parts: Vec<&str> = s.split(':').collect();
    let name = parts[0].to_lowercase();

    let reg = registry_by_id(&name).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown transformation: '{}'. Available: {}",
            name,
            all_registrations()
                .map(|r| r.meta.id)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;

    let value = match &reg.meta.param {
        ParamKind::Slider { .. } => {
            if parts.len() != 2 {
                return Err(anyhow::anyhow!(
                    "{} requires a float value. E.g., {}:0.5",
                    reg.meta.display_name,
                    reg.meta.id
                ));
            }
            parts[1].parse::<f32>()?
        }
        ParamKind::Toggle => 0.0,
    };

    Ok(Transform {
        shader_id: reg.meta.id,
        value,
    })
}

fn main() -> anyhow::Result<()> {
    let args = cli::Cli::parse();

    if args.headless {
        let output_path = args
            .output
            .ok_or_else(|| anyhow::anyhow!("--output is required in headless mode"))?;

        let mut transforms = Vec::new();

        if let Some(pipeline_path) = args.pipeline {
            let config = std::fs::read_to_string(pipeline_path)?;
            for line in config.lines() {
                let s = line.trim();
                if !s.is_empty() && !s.starts_with('#') {
                    transforms.push(parse_transform(s)?);
                }
            }
        } else {
            for t in args.apply {
                transforms.push(parse_transform(&t)?);
            }
        }

        let input_path = args
            .input
            .ok_or_else(|| anyhow::anyhow!("An input file is required in headless mode"))?;
        println!("Running headless processing on {:?}", input_path);

        let mut timer = timing::PipelineTimer::new(args.timings);

        let mut img = bdip_core::io::load_image(&input_path)?;
        timer.lap("disk read");

        let engine = GpuEngine::new()?;
        let mut renderer = Renderer::new(&engine);

        let uploaded_texture = upload_texture(&engine.device, &engine.queue, &img);
        timer.lap("gpu upload");

        let mut current_texture = renderer.ingest(&engine, &uploaded_texture);
        for transform in &transforms {
            println!("Applying {:?}", transform);
            current_texture = renderer.apply(&engine, &current_texture, transform);
        }
        timer.lap("gpu execute");

        let presentation_buffer = renderer.present(&engine, &current_texture);
        let (width, height) = img.dimensions();
        img = download_presentation_buffer(
            &engine.device,
            &engine.queue,
            &presentation_buffer,
            width,
            height,
        )?;
        timer.lap("gpu readback");

        bdip_core::io::save_image(&img, &output_path)?;
        timer.lap("disk write");

        timer.report();
        println!("Saved output to {:?}", output_path);
    } else {
        ui::run(args.input)?;
    }

    Ok(())
}
