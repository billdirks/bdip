use iced::widget::{button, row};
use iced::{Element, Length};

use super::app::BdipApp;
use super::message::Message;

pub fn view(app: &BdipApp) -> Element<'_, Message> {
    let load_btn = button("Load Image").on_press(Message::LoadImagePressed);

    // Save is only active when an image is loaded.
    let save_btn = if app.image_handle.is_some() {
        button("Save Image").on_press(Message::SaveImagePressed)
    } else {
        button("Save Image")
    };

    row![load_btn, save_btn]
        .spacing(8)
        .padding(8)
        .width(Length::Fill)
        .into()
}
