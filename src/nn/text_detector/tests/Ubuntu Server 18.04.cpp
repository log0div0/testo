
#include <catch.hpp>
#include "../TextDetector.hpp"

TEST_CASE("Ubuntu Server 18.04/Установка") {
	TextDetector& detector = TextDetector::instance();

	{
		stb::Image img("Ubuntu Server 18.04/Установка/Выбор языка.png");
		CHECK(detector.detect(img, "Language").size() == 2);
		CHECK(detector.detect(img, "English").size() == 1);
		CHECK(detector.detect(img, "Русский").size() == 1);
		CHECK(detector.detect(img, "F1 Help").size() == 1);
		CHECK(detector.detect(img, "F2 Language").size() == 1);
		CHECK(detector.detect(img, "F3 Keymap").size() == 1);
		CHECK(detector.detect(img, "F4 Modes").size() == 1);
		CHECK(detector.detect(img, "F5 Accessibility").size() == 1);
		CHECK(detector.detect(img, "F6 Other Options").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/Начальный экран.png");
		CHECK(detector.detect(img, "Install Ubuntu Server").size() == 2);
		CHECK(detector.detect(img, "Install Ubuntu Server with the HWE kernel").size() == 1);
		CHECK(detector.detect(img, "Check disc for defects").size() == 1);
		CHECK(detector.detect(img, "Test memory").size() == 1);
		CHECK(detector.detect(img, "Boot from first hard disk").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/Опять выбор языка.png");
		CHECK(detector.detect(img, "Welcome !").size() == 1);
		CHECK(detector.detect(img, "Добро пожаловать !").size() == 1);
		CHECK(detector.detect(img, "Please choose your preferred language.").size() == 1);
		CHECK(detector.detect(img, "English").size() == 1);
		CHECK(detector.detect(img, "Русский").size() == 1);
		CHECK(detector.detect(img, "Use UP, DOWN and ENTER keys to select your language.").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/Приветствие.png");
		CHECK(detector.detect(img, "Install Ubuntu").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/Настройка сети.png");
		CHECK(detector.detect(img, "Network connections").size() == 1);
		CHECK(detector.detect(img, "192.168.122.219/24").size() == 1);
		CHECK(detector.detect(img, "52:54:00:45:12:e7").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/Настройка прокси.png");
		CHECK(detector.detect(img, "Configure proxy").size() == 1);
		CHECK(detector.detect(img, "\"http://[[user][:pass]@]host[:port]/\"").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/Установка закончена.png");
		CHECK(detector.detect(img, "Installation complete!").size() == 1);
		CHECK(detector.detect(img, "Reboot Now").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/Пожалуйста, извлеките CD-ROM.png");
		CHECK(detector.detect(img, "Please remove the installation medium, then press ENTER:").size() == 1);
	}
}

TEST_CASE("Ubuntu Server 18.04/Консоль") {
	TextDetector& detector = TextDetector::instance();

	{
		stb::Image img("Ubuntu Server 18.04/Консоль/Установка питона.png");
		CHECK(detector.detect(img, "Setting up daemon").size() == 1);
	}
}
