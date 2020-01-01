
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Континент 4/Установка") {
	nn::OCR& ocr = nn::OCR::instance();

	{
		auto result = ocr.run("Континент 4/Установка/Начальный экран.png");
		CHECK(result.search("Установить Континент 4.1.0.919").size() == 1);
		CHECK(result.search("Тест памяти").size() == 1);
	}
	{
		auto result = ocr.run("Континент 4/Установка/Выбор платформы.png");
		CHECK(result.search("Выберите тип платформы").size() == 1);
		CHECK(result.search("Настраиваемая").size() == 1);
	}
	{
		auto result = ocr.run("Континент 4/Установка/Ввод номера шлюза.png");
		CHECK(result.search("Установка").size() == 1);
		CHECK(result.search("Введите идентификатор шлюза").size() == 1);
	}
}

TEST_CASE("Континент 4/Локальное меню") {
	nn::OCR& ocr = nn::OCR::instance();

	{
		auto result = ocr.run("Континент 4/Локальное меню/Выбор метода аутентификации.png");
		CHECK(result.search("Выбор метода аутентификации").size() == 1);
		CHECK(result.search("Соболь").size() == 1);
		CHECK(result.search("Учётная запись/пароль").size() == 1);
		CHECK(result.search("Отмена").size() == 1);
	}
	{
		auto result = ocr.run("Континент 4/Локальное меню/Главное меню до инициализации.png");
		CHECK(result.search("Главное меню").size() == 1);
		CHECK(result.search("Инициализация").size() == 1);
		CHECK(result.search("Журналы").size() == 1);
		CHECK(result.search("Инструменты").size() == 1);
		CHECK(result.search("Настройки").size() == 1);
		CHECK(result.search("Change Language/Сменить язык").size() == 1);
		CHECK(result.search("Завершение работы устройства").size() == 1);
	}
	{
		auto result = ocr.run("Континент 4/Локальное меню/Главное меню после инициализации.png");
		CHECK(result.search("Повторная инициализация").size() == 1);
		CHECK(result.search("Сертификаты").size() == 1);
		CHECK(result.search("Настройка ЦУС").size() == 1);
	}
	{
		auto result = ocr.run("Континент 4/Локальное меню/Инициализация прошла успешно.png");
		CHECK(result.search("Успешно.").size() == 1);
		CHECK(result.search("Нажмите Enter").size() == 1);
	}
	{
		auto result = ocr.run("Континент 4/Локальное меню/Инициализировать устройство как.png");
		CHECK(result.search("Инициализировать устройство как:").size() == 1);
		CHECK(result.search("Начать инициализацию").size() == 1);
	}
	{
		auto result = ocr.run("Континент 4/Локальное меню/Меню Инструменты.png");
		CHECK(result.search("Инструменты").size() == 1);
		CHECK(result.search("Экспорт конфигурации УБ на носитель").size() == 1);
	}
	{
		auto result = ocr.run("Континент 4/Локальное меню/Меню Настройки.png");
		CHECK(result.search("Меню настроек").size() == 1);
		CHECK(result.search("Применение локальной политики").size() == 1);
	}
	{
		auto result = ocr.run("Континент 4/Локальное меню/Меню Сертификаты.png");
		CHECK(result.search("Сертификаты УЦ").size() == 1);
	}
}
