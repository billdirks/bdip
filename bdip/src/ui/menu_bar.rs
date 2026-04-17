use iced::widget::{button, column, container, text};
use iced::{Element, Length};

use super::app::BdipApp;
use super::message::Message;
use super::style;

/// Renders the top menu bar strip containing the "File" label.
pub fn view(app: &BdipApp) -> Element<'_, Message> {
    let is_open = app.menu_open;
    let file_btn = button(text("File"))
        .padding(iced::Padding::default().top(6).bottom(4).left(12).right(12))
        .style(move |theme, status| style::menu_file_button(theme, status, is_open))
        .on_press(Message::ToggleFileMenu);

    container(file_btn)
        .width(Length::Fill)
        .style(style::menu_bar_container)
        .into()
}

/// Renders the open pulldown panel. Only called when `app.menu_open` is true.
pub fn pulldown(app: &BdipApp) -> Element<'_, Message> {
    let load_btn = button(text("Load Image"))
        .padding([3, 16])
        .width(Length::Fill)
        .style(style::menu_item_button);
    let load_item = if !app.is_loading {
        load_btn.on_press(Message::LoadImagePressed)
    } else {
        load_btn
    };

    let save_btn = button(text("Save Image"))
        .padding([3, 16])
        .width(Length::Fill)
        .style(style::menu_item_button);
    let save_item = if app.image_handle.is_some() && !app.is_saving {
        save_btn.on_press(Message::SaveImagePressed)
    } else {
        save_btn
    };

    container(column![load_item, save_item].width(Length::Fixed(130.0)))
        .padding([10, 0])
        .style(style::menu_pulldown_container)
        .into()
}
