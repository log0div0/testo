
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Ubuntu Server 16.04/Установка") {
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Начальный экран.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Install Ubuntu Server").size() == 2);
		CHECK(tensor.match("Install Ubuntu Server with the HWE kernel").size() == 1);
		CHECK(tensor.match("Test memory").size() == 1);
		CHECK(tensor.match("Boot from first hard disk").size() == 1);
	}
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Выбор языка.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Select a language").size() == 1);
	}
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Выбор локации.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Select your location").size() == 1);
	}
}
