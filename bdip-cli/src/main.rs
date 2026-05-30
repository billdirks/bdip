use bdip_core::gpu::shaders::{ParamKind, Transform, all_registrations, registry_by_id};
use bdip_core::gpu::texture::{download_presentation_buffer, upload_texture};
use bdip_core::gpu::{engine::GpuEngine, image_pipeline::Renderer};
use clap::Parser;
use std::io::IsTerminal;
use std::path::PathBuf;

mod timing;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    /// Input image file path
    pub input: Option<PathBuf>,

    /// Output file path
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Apply a transformation. Can be used multiple times. E.g., -a brightness:0.5 -a invert
    #[arg(short, long = "apply", conflicts_with = "pipeline")]
    pub apply: Vec<String>,

    /// Path to a text file containing line-by-line transformations
    #[arg(short, long)]
    pub pipeline: Option<PathBuf>,

    /// Print per-stage pipeline timings to stderr after processing
    #[arg(long, default_value_t = false)]
    pub timings: bool,

    /// Print CLI usage and parameter descriptions for a shader, then exit
    #[arg(long, value_name = "SHADER_ID", exclusive = true)]
    pub describe_shader: Option<String>,

    /// List all available shaders and their descriptions, then exit
    #[arg(long, exclusive = true)]
    pub list_shaders: bool,
}

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

    let values = match &reg.meta.param {
        ParamKind::Sliders(defs) => {
            let value_strs: Vec<&str> = parts[1..].to_vec();
            if value_strs.len() != defs.len() {
                return Err(anyhow::anyhow!(
                    "{} requires {} value(s). E.g., {}:{}",
                    reg.meta.display_name,
                    defs.len(),
                    reg.meta.id,
                    defs.iter()
                        .map(|d| format!("{}", d.default))
                        .collect::<Vec<_>>()
                        .join(":")
                ));
            }
            value_strs
                .iter()
                .map(|s| s.parse::<f32>().map_err(|e| anyhow::anyhow!(e)))
                .collect::<anyhow::Result<Vec<f32>>>()?
        }
        ParamKind::Toggle => vec![],
    };

    Ok(Transform {
        shader_id: reg.meta.id,
        values,
    })
}

fn describe_shader(shader_id: &str) -> anyhow::Result<()> {
    let reg = registry_by_id(shader_id).ok_or_else(|| {
        let available = all_registrations()
            .map(|r| r.meta.id)
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::anyhow!("Unknown shader '{}'. Available: {}", shader_id, available)
    })?;

    let meta = &reg.meta;
    println!("Shader:  {}", meta.display_name);
    println!("ID:      {}", meta.id);
    println!();
    println!("{}", meta.description);
    println!();

    match &meta.param {
        ParamKind::Sliders(defs) => {
            let positional: String = defs
                .iter()
                .map(|d| format!("<{}>", d.name.to_lowercase().replace(' ', "_")))
                .collect::<Vec<_>>()
                .join(":");
            println!("Usage:   --apply {}:{}", meta.id, positional);
            println!();
            println!("Parameters:");
            for def in defs.iter() {
                println!(
                    "  {}  (range: {}..{}, default: {})",
                    def.name, def.min, def.max, def.default
                );
                println!("    {}", def.description);
            }
        }
        ParamKind::Toggle => {
            println!("Usage:   --apply {}", meta.id);
            println!();
            println!("No parameters — the effect is either on or off.");
        }
    }

    Ok(())
}

fn list_shaders() {
    let mut shaders: Vec<_> = all_registrations().collect();
    shaders.sort_by_key(|r| r.meta.id);

    let is_tty = std::io::stdout().is_terminal();

    for reg in shaders {
        if is_tty {
            println!("\x1b[1m{}\x1b[0m: {}", reg.meta.id, reg.meta.description);
        } else {
            println!("{}: {}", reg.meta.id, reg.meta.description);
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();

    if args.list_shaders {
        list_shaders();
        return Ok(());
    }

    if let Some(ref shader_id) = args.describe_shader {
        return describe_shader(shader_id);
    }

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

    if transforms.is_empty() {
        return Err(anyhow::anyhow!(
            "At least one transform (--apply or --pipeline) is required"
        ));
    }

    let input_path = args
        .input
        .ok_or_else(|| anyhow::anyhow!("<INPUT> is required when processing an image"))?;

    let output_path = args
        .output
        .ok_or_else(|| anyhow::anyhow!("--output is required"))?;

    println!("Running processing on {:?}", input_path);

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
        current_texture = renderer.apply(&engine, &current_texture, transform)?;
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
    println!("Saved output to {:?}", output_path);

    timer.report();

    Ok(())
}
