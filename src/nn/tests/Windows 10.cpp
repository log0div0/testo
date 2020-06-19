
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Windows 10/Установка") {
	{
		stb::Image image("Windows 10/Установка/Продолжить на выбранном языке.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Продолжить на выбранном языке?").size() == 1);
		CHECK(tensor.match("Да").size() == 1);
		CHECK(tensor.match("Русский").size() == 1);
		CHECK(tensor.match("English (United States)").size() == 1);
		CHECK(tensor.match("Добро пожаловать").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Давайте начнём с региона.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Давайте начнем с региона. Это правильно?").size() == 1);
		CHECK(tensor.match("Парагвай").size() == 1);
		CHECK(tensor.match("Перу").size() == 1);
		CHECK(tensor.match("Польша").size() == 1);
		CHECK(tensor.match("Португалия").size() == 1);
		CHECK(tensor.match("Пуэрто-Рико").size() == 1);
		CHECK(tensor.match("Реюньон").size() == 1);
		CHECK(tensor.match("Россия").size() == 1);
		CHECK(tensor.match("Да").size() == 2);
		CHECK(tensor.match("Основы").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Лицензионное соглашение о Windows 10.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Лицензионное соглашение о Windows 10").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Если вы подключитесь к Интернету.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Если вы подключитесь к Интернету").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Войдите с помощью учетной записи.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Войдите с помощью учетной записи").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Кто будет использовать этот компьютер.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Кто будет использовать этот компьютер").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Удобная работа на разных устройствах.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Удобная работа на разных устройствах").size() == 1);
	}
}

TEST_CASE("Windows 10/Валидата CSP") {
	{
		stb::Image image("Windows 10/Валидата CSP/Выберите считыватель ключа.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Программа конфигурации Валидата CSP").size() == 1);
		CHECK(tensor.match("ДСЧ").size() == 1);
		CHECK(tensor.match("Считыватели ключа").size() == 1);
		CHECK(tensor.match("Ключи").size() == 1);
		CHECK(tensor.match("Сертификаты").size() == 1);
		CHECK(tensor.match("Совместимость").size() == 1);
		CHECK(tensor.match("Сервис").size() == 1);
		CHECK(tensor.match("Выберите считыватель ключа").size() == 1);
		CHECK(tensor.match("Считыватель съёмного диска").size() == 1);
		CHECK(tensor.match("Считыватель 'Соболь'").size() == 1);
		CHECK(tensor.match("Считыватель 'Аккорд'").size() == 1);
		CHECK(tensor.match("Считыватель реестра").size() == 1);
		CHECK(tensor.match("Считыватель ruToken").size() == 1);
		CHECK(tensor.match("Считыватель eToken").size() == 1);
		CHECK(tensor.match("Считыватель vdToken").size() == 2);
		CHECK(tensor.match("Отмена").size() == 2);
		CHECK(tensor.match("Применить").size() == 1);
		CHECK(tensor.match("Validata").size() == 1);
	}
}