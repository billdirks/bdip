use iced::widget::{button, column, container, pick_list, rule, scrollable, slider, text};
use iced::{Element, Length};

use super::app::BdipApp;
use super::message::{Message, TransformOption};
use super::style;

/// Approximate height to show ~5 history entries (each ~28px including spacing).
const HISTORY_MAX_HEIGHT: f32 = 155.0;

const TRANSFORM_OPTIONS: &[TransformOption] =
    &[TransformOption::Brightness, TransformOption::Saturation];

pub fn view(app: &BdipApp) -> Element<'_, Message> {
    let transform_picker = pick_list(
        TRANSFORM_OPTIONS,
        Some(app.selected_transform.clone()),
        Message::TransformSelected,
    );

    let transform_control: Element<'_, Message> = match app.selected_transform {
        TransformOption::Brightness | TransformOption::Saturation | TransformOption::Contrast => {
            let s = slider(
                -1.0_f32..=1.0_f32,
                app.preview_value,
                Message::SliderChanged,
            )
            .step(0.01)
            .on_release(Message::SliderReleased);
            let value_label = text(format!("{:.2}", app.preview_value));
            column![s, value_label].spacing(4).into()
        }
        TransformOption::Grayscale | TransformOption::Invert => {
            button("Apply").on_press(Message::ApplyParameterless).into()
        }
    };

    let history_section = history_view(app);

    column![transform_picker, transform_control, history_section]
        .spacing(16)
        .padding(8)
        .width(Length::Fill)
        .into()
}

fn history_view(app: &BdipApp) -> Element<'_, Message> {
    let can_undo = app.history.can_undo();
    let can_redo = app.history.can_redo();

    let undo_btn = {
        let b = button("Undo (⌘Z)");
        if can_undo {
            b.on_press(Message::Undo)
        } else {
            b
        }
    };

    let redo_btn = {
        let b = button("Redo (⌘⇧Z)");
        if can_redo {
            b.on_press(Message::Redo)
        } else {
            b
        }
    };

    let controls = iced::widget::row![undo_btn, redo_btn].spacing(8);

    // Build the scrollable history list.
    let mut list: iced::widget::Column<'_, Message> = column![].spacing(4);

    // Active entries — newest first.
    let applied = app.history.applied_transforms();
    for t in applied.iter().rev() {
        list = list.push(text(t.to_string()));
    }

    // Divider + dimmed redo entries — only when there are undone items.
    let redo = app.history.redo_transforms();
    if !redo.is_empty() {
        list = list.push(rule::horizontal(1));
        for t in redo {
            list = list.push(text(t.to_string()).style(style::dimmed_text));
        }
    }

    let history_scroll =
        container(scrollable(list.width(Length::Fill)).height(Length::Fixed(HISTORY_MAX_HEIGHT)))
            .width(Length::Fill);

    column![controls, history_scroll]
        .spacing(8)
        .width(Length::Fill)
        .into()
}
