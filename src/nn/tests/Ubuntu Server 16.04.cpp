
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Ubuntu Server 16.04/Установка") {
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Начальный экран.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Install Ubuntu Server").size() == 2);
		CHECK(ocr.search("Install Ubuntu Server with the HWE kernel").size() == 1);
		CHECK(ocr.search("Test memory").size() == 1);
		CHECK(ocr.search("Boot from first hard disk").size() == 1);
	}
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Выбор языка.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Select a language").size() == 1);
	}
	{
		stb::Image image("Ubuntu Server 16.04/Установка/Выбор локации.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Select your location").size() == 1);
	}
}
