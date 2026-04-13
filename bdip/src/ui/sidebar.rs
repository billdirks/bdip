use iced::widget::{button, column, pick_list, slider, text};
use iced::{Element, Length};

use super::app::BdipApp;
use super::message::{Message, TransformOption};

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

    let history_label = text("History");

    column![transform_picker, transform_control, history_label]
        .spacing(16)
        .padding(8)
        .width(Length::Fill)
        .into()
}
