use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Input image file path (optional for UI mode)
    pub input: Option<PathBuf>,

    /// Run the application in headless mode without UI
    #[arg(long, default_value_t = false)]
    pub headless: bool,

    /// Output file path (required for headless mode)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Apply a transformation. Can be used multiple times. E.g., -a brightness:0.5 -a invert
    #[arg(short, long = "apply", conflicts_with = "pipeline")]
    pub apply: Vec<String>,

    /// Path to a text file containing line-by-line transformations
    #[arg(short, long)]
    pub pipeline: Option<PathBuf>,

    /// Print per-stage pipeline timings to stderr after headless processing
    #[arg(long, default_value_t = false)]
    pub timings: bool,
}
