use anyhow::Context;
use std::path::PathBuf;

mod app;
mod canvas;
mod menu_bar;
mod message;
mod scheduler;
mod sidebar;
mod style;

use app::BdipApp;

fn app_theme(_: &BdipApp) -> iced::Theme {
    iced::Theme::Dark
}

pub fn run(input: Option<PathBuf>) -> anyhow::Result<()> {
    iced::application(
        move || BdipApp::new(input.clone()),
        BdipApp::update,
        BdipApp::view,
    )
    .title("bdip")
    .theme(app_theme)
    .subscription(BdipApp::subscription)
    .run()
    .context("Failed to run iced application")?;
    Ok(())
}
