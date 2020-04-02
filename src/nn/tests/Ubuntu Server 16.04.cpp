
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Ubuntu Server 16.04/Установка") {
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Начальный экран.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Install Ubuntu Server").size() == 2);
		CHECK(tensor.match(&image, "Install Ubuntu Server with the HWE kernel").size() == 1);
		CHECK(tensor.match(&image, "Test memory").size() == 1);
		CHECK(tensor.match(&image, "Boot from first hard disk").size() == 1);
	}
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Выбор языка.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Select a language").size() == 1);
	}
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Выбор локации.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Select your location").size() == 1);
	}
}
