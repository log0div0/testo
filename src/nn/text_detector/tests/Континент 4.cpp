
#include <catch.hpp>
#include "../TextDetector.hpp"

TEST_CASE("Континент 4/Установка") {
	TextDetector& detector = TextDetector::instance();

	{
		stb::Image img("Континент 4/Установка/Начальный экран.png");
		CHECK(detector.detect(img, "Установить Континент 4.1.0.919").size() == 1);
		CHECK(detector.detect(img, "Тест памяти").size() == 1);
	}
	{
		stb::Image img("Континент 4/Установка/Выбор платформы.png");
		CHECK(detector.detect(img, "Выберите тип платформы").size() == 1);
		CHECK(detector.detect(img, "Настраиваемая").size() == 1);
	}
	{
		stb::Image img("Континент 4/Установка/Ввод номера шлюза.png");
		CHECK(detector.detect(img, "Установка").size() == 1);
		CHECK(detector.detect(img, "Введите идентификатор шлюза").size() == 1);
	}
}

TEST_CASE("Континент 4/Локальное меню") {
	TextDetector& detector = TextDetector::instance();

	{
		stb::Image img("Континент 4/Локальное меню/Выбор метода аутентификации.png");
		CHECK(detector.detect(img, "Выбор метода аутентификации").size() == 1);
		CHECK(detector.detect(img, "Соболь").size() == 1);
		CHECK(detector.detect(img, "Учётная запись/пароль").size() == 1);
		CHECK(detector.detect(img, "Отмена").size() == 1);
	}
	{
		stb::Image img("Континент 4/Локальное меню/Главное меню до инициализации.png");
		CHECK(detector.detect(img, "Главное меню").size() == 1);
		CHECK(detector.detect(img, "Инициализация").size() == 1);
		CHECK(detector.detect(img, "Журналы").size() == 1);
		CHECK(detector.detect(img, "Инструменты").size() == 1);
		CHECK(detector.detect(img, "Настройки").size() == 1);
		CHECK(detector.detect(img, "Change Language/Сменить язык").size() == 1);
		CHECK(detector.detect(img, "Завершение работы устройства").size() == 1);
	}
	{
		stb::Image img("Континент 4/Локальное меню/Главное меню после инициализации.png");
		CHECK(detector.detect(img, "Повторная инициализация").size() == 1);
		CHECK(detector.detect(img, "Сертификаты").size() == 1);
		CHECK(detector.detect(img, "Настройка ЦУС").size() == 1);
	}
	{
		stb::Image img("Континент 4/Локальное меню/Инициализация прошла успешно.png");
		CHECK(detector.detect(img, "Успешно.").size() == 1);
		CHECK(detector.detect(img, "Нажмите Enter").size() == 1);
	}
	{
		stb::Image img("Континент 4/Локальное меню/Инициализировать устройство как.png");
		CHECK(detector.detect(img, "Инициализировать устройство как:").size() == 1);
		CHECK(detector.detect(img, "Начать инициализацию").size() == 1);
	}
	{
		stb::Image img("Континент 4/Локальное меню/Меню Инструменты.png");
		CHECK(detector.detect(img, "Инструменты").size() == 1);
		CHECK(detector.detect(img, "Экспорт конфигурации УБ на носитель").size() == 1);
	}
	{
		stb::Image img("Континент 4/Локальное меню/Меню Настройки.png");
		CHECK(detector.detect(img, "Меню настроек").size() == 1);
		CHECK(detector.detect(img, "Применение локальной политики").size() == 1);
	}
	{
		stb::Image img("Континент 4/Локальное меню/Меню Сертификаты.png");
		CHECK(detector.detect(img, "Сертификаты УЦ").size() == 1);
	}
}
