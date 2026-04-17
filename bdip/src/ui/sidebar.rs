use iced::widget::{
    button, column, container, pick_list, row, rule, scrollable, slider, text, toggler,
};
use iced::{Element, Length};

use bdip_core::gpu::shaders::{ParamKind, ShaderOption, registry_by_id, sorted_registrations};

use super::app::BdipApp;
use super::message::Message;
use super::style;

pub fn view(app: &BdipApp) -> Element<'_, Message> {
    let transform_section = transform_view(app);
    let history_section = history_view(app);

    // Each section gets half the sidebar height.
    column![
        container(transform_section)
            .width(Length::Fill)
            .height(Length::FillPortion(1)),
        rule::horizontal(1),
        container(history_section)
            .width(Length::Fill)
            .height(Length::FillPortion(1)),
    ]
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

fn transform_view(app: &BdipApp) -> Element<'_, Message> {
    let options: Vec<ShaderOption> = sorted_registrations()
        .into_iter()
        .map(|reg| ShaderOption {
            id: reg.meta.id,
            display_name: reg.meta.display_name,
        })
        .collect();

    let transform_picker = pick_list(
        options,
        Some(app.selected_transform.clone()),
        Message::TransformSelected,
    );

    let selected_reg = registry_by_id(app.selected_transform.id);
    let transform_control: Element<'_, Message> = match selected_reg.map(|r| &r.meta.param) {
        Some(ParamKind::Slider { min, max, .. }) => {
            let s = slider(*min..=*max, app.preview_value, Message::SliderChanged)
                .step(0.01)
                .on_release(Message::SliderReleased);
            let value_label = text(format!("{:.2}", app.preview_value));
            column![s, value_label].spacing(4).into()
        }
        Some(ParamKind::Toggle) | None => {
            let is_active = app.is_transform_active(&app.selected_transform);
            row![
                text("Apply"),
                toggler(is_active).on_toggle(|_| Message::ToggleParameterless),
            ]
            .spacing(8)
            .align_y(iced::Alignment::Center)
            .into()
        }
    };

    column![transform_picker, transform_control]
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

    // The scrollable expands to fill whatever height the container gives this section.
    let history_scroll = scrollable(list.width(Length::Fill)).height(Length::Fill);

    column![controls, history_scroll]
        .spacing(8)
        .padding(8)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}
