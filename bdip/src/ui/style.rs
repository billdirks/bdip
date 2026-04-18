use iced::widget::{button, container, text};
use iced::{Background, Border, Color, Shadow, Theme};

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

/// Background style for the menu bar strip.
pub fn menu_bar_container(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();
    container::Style {
        background: Some(Background::Color(palette.background.weak.color)),
        ..Default::default()
    }
}

/// Background and border style for the open pulldown panel.
pub fn menu_pulldown_container(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();
    container::Style {
        background: Some(Background::Color(palette.background.weak.color)),
        border: Border {
            color: palette.background.strong.color,
            width: 1.0,
            radius: 2.0.into(),
        },
        ..Default::default()
    }
}

/// Style for the "File" menu-bar button. Shows a subtle highlight when the menu is open,
/// and a slightly brighter highlight on hover.
pub fn menu_file_button(theme: &Theme, status: button::Status, is_open: bool) -> button::Style {
    let palette = theme.extended_palette();
    let text_color = Color {
        a: 0.85,
        ..palette.background.weak.text
    };
    let open_bg = Color {
        a: 0.18,
        ..palette.background.strong.text
    };
    let hover_bg = Color {
        a: 0.25,
        ..palette.background.strong.text
    };
    let base = button::Style {
        text_color,
        border: Border::default(),
        shadow: Shadow::default(),
        snap: true,
        background: if is_open {
            Some(Background::Color(open_bg))
        } else {
            None
        },
    };
    match status {
        button::Status::Active => base,
        button::Status::Hovered | button::Status::Pressed => button::Style {
            background: Some(Background::Color(hover_bg)),
            ..base
        },
        button::Status::Disabled => base,
    }
}

/// Style for pulldown-menu item buttons: transparent background that highlights on hover,
/// and dimmed text when disabled.
pub fn menu_item_button(theme: &Theme, status: button::Status) -> button::Style {
    let palette = theme.extended_palette();
    let resting_text = Color {
        a: 0.85,
        ..palette.background.weak.text
    };
    let base = button::Style {
        background: None,
        text_color: resting_text,
        border: Border::default(),
        shadow: Shadow::default(),
        snap: true,
    };
    match status {
        button::Status::Active => base,
        button::Status::Hovered | button::Status::Pressed => button::Style {
            background: Some(Background::Color(palette.primary.strong.color)),
            text_color: palette.primary.strong.text,
            ..base
        },
        button::Status::Disabled => button::Style {
            text_color: Color {
                a: 0.4,
                ..palette.background.strong.text
            },
            ..base
        },
    }
}

pub const SECTION_HEADER_SIZE: f32 = 12.0;
pub const SECTION_HEADER_SPACING: f32 = 8.0;
pub const SECTION_SIDEBAR_PADDING: f32 = 8.0;

/// Style for sidebar section header titles.
pub fn section_header_text(theme: &Theme) -> text::Style {
    let palette = theme.extended_palette();
    text::Style {
        color: Some(palette.background.strong.text),
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
