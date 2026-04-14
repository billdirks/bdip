use iced::widget::{container, text};
use iced::{Background, Color, Theme};

/// Style for history entries that have been undone. Renders text in a dimmed
/// gray to signal that these entries are inactive and would be restored by Redo.
pub fn dimmed_text(theme: &Theme) -> text::Style {
    let palette = theme.extended_palette();
    text::Style {
        color: Some(Color {
            a: 0.4,
            ..palette.background.strong.text
        }),
    }
}

/// Style for the error banner displayed at the top of the canvas area.
pub fn error_banner(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgb(0.55, 0.10, 0.10))),
        text_color: Some(Color::WHITE),
        ..Default::default()
    }
}
