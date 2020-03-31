
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Континент 4/Установка") {
	{
		stb::Image image("Континент 4/Установка/Начальный экран.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Установить Континент 4.1.0.919").size() == 1);
		CHECK(ocr.search("Тест памяти").size() == 1);
	}
	{
		stb::Image image("Континент 4/Установка/Выбор платформы.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Выберите тип платформы").size() == 1);
		CHECK(ocr.search("Настраиваемая").size() == 1);
	}
	{
		stb::Image image("Континент 4/Установка/Ввод номера шлюза.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Установка").size() == 1);
		CHECK(ocr.search("Введите идентификатор шлюза").size() == 1);
	}
}

TEST_CASE("Континент 4/Локальное меню") {
	{
		stb::Image image("Континент 4/Локальное меню/Выбор метода аутентификации.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Выбор метода аутентификации").size() == 1);
		CHECK(ocr.search("Соболь").size() == 1);
		CHECK(ocr.search("Учётная запись/пароль").size() == 1);
		CHECK(ocr.search("Отмена").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Главное меню до инициализации.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Главное меню").size() == 1);
		CHECK(ocr.search("Сведения").size() == 1);
		CHECK(ocr.search("Сведения", "blue", "gray").size() == 1);
		CHECK(ocr.search("Сведения", "gray", "blue").size() == 0);
		CHECK(ocr.search("Инициализация").size() == 1);
		CHECK(ocr.search("Инициализация", "gray", "blue").size() == 1);
		CHECK(ocr.search("Инициализация", "blue", "gray").size() == 0);
		CHECK(ocr.search("Журналы").size() == 1);
		CHECK(ocr.search("Инструменты").size() == 1);
		CHECK(ocr.search("Настройки").size() == 1);
		CHECK(ocr.search("Change Language/Сменить язык").size() == 1);
		CHECK(ocr.search("Завершение работы устройства").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Главное меню после инициализации.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Повторная инициализация").size() == 1);
		CHECK(ocr.search("Сертификаты").size() == 1);
		CHECK(ocr.search("Настройка ЦУС").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Инициализация прошла успешно.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Успешно.").size() == 1);
		CHECK(ocr.search("Нажмите Enter").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Инициализировать устройство как.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Инициализировать устройство как:").size() == 1);
		CHECK(ocr.search("[ Начать инициализацию ]").size() == 1);
		CHECK(ocr.search("[ Начать инициализацию ]", "blue", "gray").size() == 1);
		CHECK(ocr.search("[ Начать инициализацию ]", "gray", "blue").size() == 0);
		CHECK(ocr.search("[ Отмена ]").size() == 1);
		CHECK(ocr.search("[ Отмена ]", "gray", "blue").size() == 1);
		CHECK(ocr.search("[ Отмена ]", "blue", "gray").size() == 0);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Меню Инструменты.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Инструменты").size() == 1);
		CHECK(ocr.search("Экспорт конфигурации УБ на носитель").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Меню Настройки.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Меню настроек").size() == 1);
		CHECK(ocr.search("Применение локальной политики").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Меню Сертификаты.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Сертификаты УЦ").size() == 1);
	}
	{
		stb::Image image("Континент 4/Локальное меню/Очистить локальные журналы.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Очистить локальные журналы?").size() == 1);
		CHECK(ocr.search("Да").size() == 1);
		CHECK(ocr.search("Да", "blue", "gray").size() == 1);
		CHECK(ocr.search("Да", "gray", "blue").size() == 0);
		CHECK(ocr.search("Нет").size() == 1);
		CHECK(ocr.search("Нет", "gray", "blue").size() == 1);
		CHECK(ocr.search("Нет", "blue", "gray").size() == 0);
	}
}
