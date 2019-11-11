
#include <catch.hpp>
#include "../TextDetector.hpp"

TEST_CASE("Ubuntu Server 16.04/Установка") {
	TextDetector& detector = TextDetector::instance();

	{
		stb::Image img("Ubuntu Server 16.04/Установка/Начальный экран.png");
		CHECK(detector.detect(img, "Install Ubuntu Server").size() == 2);
		CHECK(detector.detect(img, "Install Ubuntu Server with the HWE kernel").size() == 1);
		CHECK(detector.detect(img, "Test memory").size() == 1);
		CHECK(detector.detect(img, "Boot from first hard disk").size() == 1);
	}
	{
		stb::Image img("Ubuntu Server 16.04/Установка/Выбор языка.png");
		CHECK(detector.detect(img, "Select a language").size() == 1);
	}
	{
		stb::Image img("Ubuntu Server 16.04/Установка/Выбор локации.png");
		CHECK(detector.detect(img, "Select your location").size() == 1);
	}
}
