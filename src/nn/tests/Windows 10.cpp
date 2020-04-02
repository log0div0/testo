
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Windows 10/Установка") {
	{
		stb::Image image("Windows 10/Установка/Продолжить на выбранном языке.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Продолжить на выбранном языке?").size() == 1);
		CHECK(tensor.match(&image, "Да").size() == 1);
		CHECK(tensor.match(&image, "Русский").size() == 1);
		CHECK(tensor.match(&image, "English (United States)").size() == 1);
		CHECK(tensor.match(&image, "Добро пожаловать").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Давайте начнём с региона.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Давайте начнем с региона. Это правильно?").size() == 1);
		CHECK(tensor.match(&image, "Парагвай").size() == 1);
		CHECK(tensor.match(&image, "Перу").size() == 1);
		CHECK(tensor.match(&image, "Польша").size() == 1);
		CHECK(tensor.match(&image, "Португалия").size() == 1);
		CHECK(tensor.match(&image, "Пуэрто-Рико").size() == 1);
		CHECK(tensor.match(&image, "Реюньон").size() == 1);
		CHECK(tensor.match(&image, "Россия").size() == 1);
		CHECK(tensor.match(&image, "Да").size() == 2);
		CHECK(tensor.match(&image, "Основы").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Лицензионное соглашение о Windows 10.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Лицензионное соглашение о Windows 10").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Если вы подключитесь к Интернету.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Если вы подключитесь к Интернету").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Войдите с помощью учетной записи.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Войдите с помощью учетной записи").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Кто будет использовать этот компьютер.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Кто будет использовать этот компьютер").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Удобная работа на разных устройствах.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Удобная работа на разных устройствах").size() == 1);
	}
}
