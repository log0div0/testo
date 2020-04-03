
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
