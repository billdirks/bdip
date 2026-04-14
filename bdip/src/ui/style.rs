use iced::widget::text;
use iced::{Color, Theme};

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
