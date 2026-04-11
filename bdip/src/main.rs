use bdip_core::gpu::texture::{download_texture, upload_texture};
use bdip_core::{gpu::engine::GpuEngine, gpu::pipeline::Renderer, Transformation};
use clap::Parser;

mod cli;
mod ui_spike;

fn parse_transform(s: &str) -> anyhow::Result<Transformation> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts[0].to_lowercase().as_str() {
        "brightness" => {
            if parts.len() != 2 {
                return Err(anyhow::anyhow!(
                    "Brightness requires a float value. E.g., brightness:0.5"
                ));
            }
            let val = parts[1].parse::<f32>()?;
            Ok(Transformation::Brightness(val))
        }
        _ => Err(anyhow::anyhow!(
            "Unsupported or unknown transformation: {}",
            parts[0]
        )),
    }
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
        let mut img = bdip_core::io::load_image(&input_path)?;

        let engine = GpuEngine::new()?;
        let renderer = Renderer::new(&engine);

        let uploaded_texture = upload_texture(&engine.device, &engine.queue, &img);
        let mut current_texture = renderer.ingest(&engine, &uploaded_texture);

        for transform in transforms {
            match transform {
                Transformation::Brightness(val) => {
                    println!("Applying Brightness {}", val);
                    current_texture = renderer.apply_brightness(&engine, &current_texture, val);
                }
                _ => {
                    println!("Unknown transform {:?}", transform);
                }
            }
        }

        let presentation_texture = renderer.present(&engine, &current_texture);
        let (width, height) = img.dimensions();
        img = download_texture(
            &engine.device,
            &engine.queue,
            &presentation_texture,
            width,
            height,
        )?;

        bdip_core::io::save_image(&img, &output_path)?;
        println!("Saved output to {:?}", output_path);
    } else {
        println!("Starting UI Spike...");
        ui_spike::run(args.input)?;
    }

    Ok(())
}
