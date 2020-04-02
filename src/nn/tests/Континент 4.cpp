
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Континент 4/Установка") {
	{
		stb::Image image("Континент 4/Установка/Начальный экран.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Установить Континент 4.1.0.919").size() == 1);
		CHECK(tensor.match(&image, "Тест памяти").size() == 1);
	}
	{
		stb::Image image("Континент 4/Установка/Выбор платформы.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Выберите тип платформы").size() == 1);
		CHECK(tensor.match(&image, "Настраиваемая").size() == 1);
	}
	{
		stb::Image image("Континент 4/Установка/Ввод номера шлюза.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Установка").size() == 1);
		CHECK(tensor.match(&image, "Введите идентификатор шлюза").size() == 1);
	}
}

TEST_CASE("Континент 4/Локальное меню") {
	{
		stb::Image image("Континент 4/Локальное меню/Выбор метода аутентификации.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Выбор метода аутентификации").size() == 1);
		CHECK(tensor.match(&image, "Соболь").size() == 1);
		CHECK(tensor.match(&image, "Учётная запись/пароль").size() == 1);
		CHECK(tensor.match(&image, "Отмена").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Главное меню до инициализации.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Главное меню").size() == 1);
		CHECK(tensor.match(&image, "Сведения").size() == 1);
		CHECK(tensor.match(&image, "Сведения").match_foreground(&image, "blue").match_background(&image, "gray").size() == 1);
		CHECK(tensor.match(&image, "Сведения").match_foreground(&image, "gray").match_background(&image, "blue").size() == 0);
		CHECK(tensor.match(&image, "Инициализация").size() == 1);
		CHECK(tensor.match(&image, "Инициализация").match_foreground(&image, "gray").match_background(&image, "blue").size() == 1);
		CHECK(tensor.match(&image, "Инициализация").match_foreground(&image, "blue").match_background(&image, "gray").size() == 0);
		CHECK(tensor.match(&image, "Журналы").size() == 1);
		CHECK(tensor.match(&image, "Инструменты").size() == 1);
		CHECK(tensor.match(&image, "Настройки").size() == 1);
		CHECK(tensor.match(&image, "Change Language/Сменить язык").size() == 1);
		CHECK(tensor.match(&image, "Завершение работы устройства").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Главное меню после инициализации.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Повторная инициализация").size() == 1);
		CHECK(tensor.match(&image, "Сертификаты").size() == 1);
		CHECK(tensor.match(&image, "Настройка ЦУС").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Инициализация прошла успешно.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Успешно.").size() == 1);
		CHECK(tensor.match(&image, "Нажмите Enter").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Инициализировать устройство как.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Инициализировать устройство как:").size() == 1);
		CHECK(tensor.match(&image, "[ Начать инициализацию ]").size() == 1);
		CHECK(tensor.match(&image, "[ Начать инициализацию ]").match_foreground(&image, "blue").match_background(&image, "gray").size() == 1);
		CHECK(tensor.match(&image, "[ Начать инициализацию ]").match_foreground(&image, "gray").match_background(&image, "blue").size() == 0);
		CHECK(tensor.match(&image, "[ Отмена ]").size() == 1);
		CHECK(tensor.match(&image, "[ Отмена ]").match_foreground(&image, "gray").match_background(&image, "blue").size() == 1);
		CHECK(tensor.match(&image, "[ Отмена ]").match_foreground(&image, "blue").match_background(&image, "gray").size() == 0);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Меню Инструменты.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Инструменты").size() == 1);
		CHECK(tensor.match(&image, "Экспорт конфигурации УБ на носитель").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Меню Настройки.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Меню настроек").size() == 1);
		CHECK(tensor.match(&image, "Применение локальной политики").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Меню Сертификаты.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Сертификаты УЦ").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Очистить локальные журналы.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Очистить локальные журналы?").size() == 1);
		CHECK(tensor.match(&image, "Да").size() == 1);
		CHECK(tensor.match(&image, "Да").match_foreground(&image, "blue").match_background(&image, "gray").size() == 1);
		CHECK(tensor.match(&image, "Да").match_foreground(&image, "gray").match_background(&image, "blue").size() == 0);
		CHECK(tensor.match(&image, "Нет").size() == 1);
		CHECK(tensor.match(&image, "Нет").match_foreground(&image, "gray").match_background(&image, "blue").size() == 1);
		CHECK(tensor.match(&image, "Нет").match_foreground(&image, "blue").match_background(&image, "gray").size() == 0);
	}
}
