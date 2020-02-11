
#include <catch.hpp>
#include "nn/Context.hpp"

TEST_CASE("Ubuntu Server 16.04/Установка") {
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Начальный экран.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Install Ubuntu Server").size() == 2);
		CHECK(context.ocr().search("Install Ubuntu Server with the HWE kernel").size() == 1);
		CHECK(context.ocr().search("Test memory").size() == 1);
		CHECK(context.ocr().search("Boot from first hard disk").size() == 1);
	}
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Выбор языка.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Select a language").size() == 1);
	}
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Выбор локации.png");
		nn::Context context(&image);
		CHECK(context.ocr().search("Select your location").size() == 1);
	}
}
