use clap::Parser;
use std::path::PathBuf;

mod ui;

#[derive(Parser)]
struct Args {
    pub input: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ui::run(args.input)
}
