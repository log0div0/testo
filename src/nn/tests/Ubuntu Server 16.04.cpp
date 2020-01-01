
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Ubuntu Server 16.04/Установка") {
	nn::OCR& ocr = nn::OCR::instance();

	{
		auto result = ocr.run("Ubuntu Server 16.04/Установка/Начальный экран.png");
		CHECK(result.search("Install Ubuntu Server").size() == 2);
		CHECK(result.search("Install Ubuntu Server with the HWE kernel").size() == 1);
		CHECK(result.search("Test memory").size() == 1);
		CHECK(result.search("Boot from first hard disk").size() == 1);
	}
	{
		auto result = ocr.run("Ubuntu Server 16.04/Установка/Выбор языка.png");
		CHECK(result.search("Select a language").size() == 1);
	}
	{
		auto result = ocr.run("Ubuntu Server 16.04/Установка/Выбор локации.png");
		CHECK(result.search("Select your location").size() == 1);
	}
}
