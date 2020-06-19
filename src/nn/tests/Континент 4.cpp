
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Континент 4/Установка") {
	{
		stb::Image image("Континент 4/Установка/Select language.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Please select Language").size() == 1);
		CHECK(tensor.match("English").size() == 1);
		CHECK(tensor.match("Russian").size() == 1);
		CHECK(tensor.match("Autodeploy").size() == 1);
		CHECK(tensor.match("Automatic boot in 30 seconds").size() == 1);
	}
	{
		stb::Image image("Континент 4/Установка/Начальный экран.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Установить Континент 4.1.0.919").size() == 1);
		CHECK(tensor.match("Тест памяти").size() == 1);
	}
	{
		stb::Image image("Континент 4/Установка/Выбор платформы.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Выберите тип платформы").size() == 1);
		CHECK(tensor.match("Настраиваемая").size() == 1);
	}
	{
		stb::Image image("Континент 4/Установка/Ввод номера шлюза.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Установка").size() == 1);
		CHECK(tensor.match("Введите идентификатор шлюза").size() == 1);
	}
}

TEST_CASE("Континент 4/Локальное меню") {
	{
		stb::Image image("Континент 4/Локальное меню/Выбор метода аутентификации.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Выбор метода аутентификации").size() == 1);
		CHECK(tensor.match("Соболь").size() == 1);
		CHECK(tensor.match("Учётная запись/пароль").size() == 1);
		CHECK(tensor.match("Отмена").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Главное меню до инициализации.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Главное меню").size() == 1);
		CHECK(tensor.match("Сведения").size() == 1);
		CHECK(tensor.match("Сведения").match_foreground(&image, "blue").match_background(&image, "gray").size() == 1);
		CHECK(tensor.match("Сведения").match_foreground(&image, "gray").match_background(&image, "blue").size() == 0);
		CHECK(tensor.match("Инициализация").size() == 1);
		CHECK(tensor.match("Инициализация").match_foreground(&image, "gray").match_background(&image, "blue").size() == 1);
		CHECK(tensor.match("Инициализация").match_foreground(&image, "blue").match_background(&image, "gray").size() == 0);
		CHECK(tensor.match("Журналы").size() == 1);
		CHECK(tensor.match("Инструменты").size() == 1);
		CHECK(tensor.match("Настройки").size() == 1);
		CHECK(tensor.match("Change Language/Сменить язык").size() == 1);
		CHECK(tensor.match("Завершение работы устройства").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Главное меню после инициализации.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Повторная инициализация").size() == 1);
		CHECK(tensor.match("Сертификаты").size() == 1);
		CHECK(tensor.match("Настройка ЦУС").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Инициализация прошла успешно.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Успешно.").size() == 1);
		CHECK(tensor.match("Нажмите Enter").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Инициализировать устройство как.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Инициализировать устройство как:").size() == 1);
		CHECK(tensor.match("[ Начать инициализацию ]").size() == 1);
		CHECK(tensor.match("[ Начать инициализацию ]").match_foreground(&image, "blue").match_background(&image, "gray").size() == 1);
		CHECK(tensor.match("[ Начать инициализацию ]").match_foreground(&image, "gray").match_background(&image, "blue").size() == 0);
		CHECK(tensor.match("[ Отмена ]").size() == 1);
		CHECK(tensor.match("[ Отмена ]").match_foreground(&image, "gray").match_background(&image, "blue").size() == 1);
		CHECK(tensor.match("[ Отмена ]").match_foreground(&image, "blue").match_background(&image, "gray").size() == 0);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Меню Инструменты.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Инструменты").size() == 1);
		CHECK(tensor.match("Экспорт конфигурации УБ на носитель").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Меню Настройки.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Меню настроек").size() == 1);
		CHECK(tensor.match("Применение локальной политики").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Меню Сертификаты.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Сертификаты УЦ").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Очистить локальные журналы.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Очистить локальные журналы?").size() == 1);
		CHECK(tensor.match("Да").size() == 1);
		CHECK(tensor.match("Да").match_foreground(&image, "blue").match_background(&image, "gray").size() == 1);
		CHECK(tensor.match("Да").match_foreground(&image, "gray").match_background(&image, "blue").size() == 0);
		CHECK(tensor.match("Нет").size() == 1);
		CHECK(tensor.match("Нет").match_foreground(&image, "gray").match_background(&image, "blue").size() == 1);
		CHECK(tensor.match("Нет").match_foreground(&image, "blue").match_background(&image, "gray").size() == 0);
	}
}
