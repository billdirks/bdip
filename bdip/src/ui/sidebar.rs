use iced::widget::{
    button, column, container, pick_list, row, rule, scrollable, slider, text, toggler,
};
use iced::{Element, Length};

use bdip_core::gpu::shaders::{ParamKind, ShaderOption, registry_by_id, sorted_registrations};

use super::app::{BdipApp, current_values_for};
use super::message::Message;
use super::style;

/// Decimal places required to represent any slider value exactly in decimal
/// notation. Used by the sidebar value readout and by pipeline export.
pub const SLIDER_DECIMAL_PLACES: u32 = 2;

/// Step size for all parameter sliders, derived from `SLIDER_DECIMAL_PLACES`
/// so the two cannot desync. Values snap to multiples of this step starting
/// from integer anchors.
pub const SLIDER_STEP: f32 = 1.0 / 10_i32.pow(SLIDER_DECIMAL_PLACES) as f32;

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
    let transform_control: Element<'_, Message> = match selected_reg.map(|r| &r.meta) {
        Some(meta) if matches!(meta.param, ParamKind::Sliders(_)) => {
            let ParamKind::Sliders(defs) = &meta.param else {
                unreachable!()
            };
            let base_vals =
                current_values_for(app.selected_transform.id, &app.history, &meta.param);
            let mut col = column![].spacing(8);
            for (i, def) in defs.iter().enumerate() {
                let display_val = match &app.preview_slider {
                    Some(ps) if ps.param_index == i => ps.value,
                    _ => base_vals.get(i).copied().unwrap_or(def.default),
                };
                let label_row = row![
                    text(def.name),
                    text(format!(
                        "{:.*}",
                        SLIDER_DECIMAL_PLACES as usize, display_val
                    )),
                ]
                .spacing(8);
                let s = slider(def.min..=def.max, display_val, move |val| {
                    Message::SliderChanged {
                        param_index: i,
                        value: val,
                    }
                })
                .step(SLIDER_STEP)
                .on_release(Message::SliderReleased {
                    param_index: i,
                    value: display_val,
                });
                col = col.push(label_row);
                col = col.push(s);
            }
            col.into()
        }
        _ => {
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

    let header = text("TRANSFORMATIONS")
        .size(style::SECTION_HEADER_SIZE)
        .font(iced::Font {
            weight: iced::font::Weight::Bold,
            ..iced::Font::DEFAULT
        })
        .style(style::section_header_text);

    let content = column![transform_picker, transform_control].spacing(16);

    column![header, content]
        .spacing(style::SECTION_HEADER_SPACING)
        .padding(style::SECTION_SIDEBAR_PADDING)
        .width(Length::Fill)
        .into()
}

fn history_view(app: &BdipApp) -> Element<'_, Message> {
    let can_undo = app.history.can_undo();
    let can_redo = app.history.can_redo();

    let undo_btn = {
        let b = button("Undo (⌘Z)").style(style::ghost_button);
        if can_undo {
            b.on_press(Message::Undo)
        } else {
            b
        }
    };

    let redo_btn = {
        let b = button("Redo (⌘⇧Z)").style(style::ghost_button);
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

    let header = text("HISTORY")
        .size(style::SECTION_HEADER_SIZE)
        .font(iced::Font {
            weight: iced::font::Weight::Bold,
            ..iced::Font::DEFAULT
        })
        .style(style::section_header_text);

    column![header, controls, history_scroll]
        .spacing(style::SECTION_HEADER_SPACING)
        .padding(style::SECTION_SIDEBAR_PADDING)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}
