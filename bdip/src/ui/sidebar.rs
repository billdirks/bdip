use iced::widget::{column, pick_list, text};
use iced::{Element, Length};

use super::app::BdipApp;
use super::message::{Message, TransformOption};

/// The two shaders available in Phase 5. Contrast, Grayscale, and Invert are
/// added in PRs 6–8 once their GPU shaders are implemented.
const PHASE5_OPTIONS: &[TransformOption] =
    &[TransformOption::Brightness, TransformOption::Saturation];

pub fn view(app: &BdipApp) -> Element<'_, Message> {
    let transform_picker = pick_list(
        PHASE5_OPTIONS,
        Some(app.selected_transform.clone()),
        Message::TransformSelected,
    );

    let history_label = text("History");

    column![transform_picker, history_label]
        .spacing(16)
        .padding(8)
        .width(Length::Fill)
        .into()
}
