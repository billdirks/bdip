use iced::theme::palette::Extended;
use iced::widget::{button, container, text};
use iced::{Background, Border, Color, Shadow, Theme};

/// Alpha applied to menu bar and pulldown text (File label, item rows, inline shortcuts).
/// Slightly off-white so the menu bar doesn't feel glaring against a dark background.
pub const MENU_TEXT_ALPHA: f32 = 0.85;

/// Alpha applied to disabled or inactive text (undone history entries, disabled menu items).
pub const DIMMED_TEXT_ALPHA: f32 = 0.4;

/// Alpha applied to the loaded-filename label in the menu bar.
/// Softer than MENU_TEXT_ALPHA so it reads as an orientation cue, not an action.
pub const FILENAME_TEXT_ALPHA: f32 = 0.5;

/// Background color shared by all menu chrome surfaces (menu bar strip and pulldown panel).
fn menu_surface_color(palette: &Extended) -> Color {
    palette.background.weak.color
}

/// Base text color shared by all text rendered on menu chrome surfaces (File label,
/// pulldown item rows, and the filename label).
fn menu_text_base(palette: &Extended) -> Color {
    palette.background.weak.text
}

/// Base text color for disabled or inactive text (undone history entries, disabled menu items).
fn disabled_text_base(palette: &Extended) -> Color {
    palette.background.strong.text
}

/// Base for all in-window button styles: no background, no border, no shadow, pixel-snapped.
/// Callers override `text_color` (and `background` when needed) via struct update syntax.
fn no_chrome_button_base() -> button::Style {
    button::Style {
        background: None,
        text_color: Color::TRANSPARENT, // always overridden by caller
        border: Border::default(),
        shadow: Shadow::default(),
        snap: true,
    }
}

/// Style for history entries that have been undone. Renders text in a dimmed
/// gray to signal that these entries are inactive and would be restored by Redo.
pub fn dimmed_text(theme: &Theme) -> text::Style {
    let palette = theme.extended_palette();
    text::Style {
        color: Some(Color {
            a: DIMMED_TEXT_ALPHA,
            ..disabled_text_base(palette)
        }),
    }
}

/// Background style for the menu bar strip.
pub fn menu_bar_container(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();
    container::Style {
        background: Some(Background::Color(menu_surface_color(palette))),
        ..Default::default()
    }
}

/// Background and border style for the open pulldown panel.
pub fn menu_pulldown_container(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();
    container::Style {
        background: Some(Background::Color(menu_surface_color(palette))),
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
        a: MENU_TEXT_ALPHA,
        ..menu_text_base(palette)
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
        background: if is_open {
            Some(Background::Color(open_bg))
        } else {
            None
        },
        ..no_chrome_button_base()
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
        a: MENU_TEXT_ALPHA,
        ..menu_text_base(palette)
    };
    let base = button::Style {
        text_color: resting_text,
        ..no_chrome_button_base()
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
                a: DIMMED_TEXT_ALPHA,
                ..disabled_text_base(palette)
            },
            ..base
        },
    }
}

/// Style for an inline link-styled button: no background or border, near-white resting text
/// that brightens fully on hover to signal interactivity.
pub fn link_button(theme: &Theme, status: button::Status) -> button::Style {
    let palette = theme.extended_palette();
    let base = button::Style {
        text_color: Color {
            a: MENU_TEXT_ALPHA,
            ..palette.background.base.text
        },
        ..no_chrome_button_base()
    };
    match status {
        button::Status::Hovered | button::Status::Pressed => button::Style {
            text_color: palette.background.base.text,
            ..base
        },
        _ => base,
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

/// Style for the loaded filename shown on the right side of the menu bar.
/// Dimmed so it reads as an orientation cue rather than an interactive element.
pub fn filename_text(theme: &Theme) -> text::Style {
    let palette = theme.extended_palette();
    text::Style {
        color: Some(Color {
            a: FILENAME_TEXT_ALPHA,
            ..menu_text_base(palette)
        }),
    }
}

/// Ghost/outline button: thin border, no fill, muted text. Use for low-priority actions
/// that should not compete visually with the primary canvas content.
pub fn ghost_button(theme: &Theme, status: button::Status) -> button::Style {
    let palette = theme.extended_palette();
    let resting_text = Color {
        a: MENU_TEXT_ALPHA,
        ..palette.background.base.text
    };
    let base = button::Style {
        background: None,
        text_color: resting_text,
        border: Border {
            color: palette.background.strong.color,
            width: 1.0,
            radius: 4.0.into(),
        },
        shadow: Shadow::default(),
        snap: true,
    };
    match status {
        button::Status::Active => base,
        button::Status::Hovered | button::Status::Pressed => button::Style {
            background: Some(Background::Color(Color {
                a: 0.08,
                ..disabled_text_base(palette)
            })),
            ..base
        },
        button::Status::Disabled => button::Style {
            text_color: Color {
                a: DIMMED_TEXT_ALPHA,
                ..disabled_text_base(palette)
            },
            border: Border {
                color: Color {
                    a: DIMMED_TEXT_ALPHA,
                    ..palette.background.strong.color
                },
                ..base.border
            },
            ..base
        },
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
