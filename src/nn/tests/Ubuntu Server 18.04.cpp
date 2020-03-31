
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Ubuntu Server 18.04/Установка") {
	{
		stb::Image image("Ubuntu Server 18.04/Установка/Выбор языка.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Language").size() == 2);
		CHECK(ocr.search("English").size() == 1);
		CHECK(ocr.search("Русский").size() == 1);
		CHECK(ocr.search("F1 Help").size() == 1);
		CHECK(ocr.search("F2 Language").size() == 1);
		CHECK(ocr.search("F3 Keymap").size() == 1);
		CHECK(ocr.search("F4 Modes").size() == 1);
		CHECK(ocr.search("F5 Accessibility").size() == 1);
		CHECK(ocr.search("F6 Other Options").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Начальный экран.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Install Ubuntu Server").size() == 2);
		CHECK(ocr.search("Install Ubuntu Server with the HWE kernel").size() == 1);
		CHECK(ocr.search("Check disc for defects").size() == 1);
		CHECK(ocr.search("Test memory").size() == 1);
		CHECK(ocr.search("Boot from first hard disk").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Опять выбор языка.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Welcome!").size() == 1);
		CHECK(ocr.search("Добро пожаловать!").size() == 1);
		CHECK(ocr.search("Please choose your preferred language.").size() == 1);
		CHECK(ocr.search("English").size() == 1);
		CHECK(ocr.search("Русский").size() == 1);
		CHECK(ocr.search("Use UP, DOWN and ENTER keys to select your language.").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Приветствие.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Install Ubuntu").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Настройка сети.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Network connections").size() == 1);
		CHECK(ocr.search("192.168.122.219/24").size() == 1);
		CHECK(ocr.search("52:54:00:45:12:e7").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Настройка прокси.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Configure proxy").size() == 1);
		CHECK(ocr.search("\"http://[[user][:pass]@]host[:port]/\"").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Установка закончена.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Installation complete!").size() == 1);
		CHECK(ocr.search("Reboot Now").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Пожалуйста, извлеките CD-ROM.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Please remove the installation medium, then press ENTER:").size() == 1);
	}
}

TEST_CASE("Ubuntu Server 18.04/Консоль") {
	{
		stb::Image image("Ubuntu Server 18.04/Консоль/Установка питона.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Setting up daemon").size() == 1);
		CHECK(ocr.search("Unpacking python").size() == 2);
		CHECK(ocr.search("Unpacking python2.7").size() == 1);
		CHECK(ocr.search("Selecting previously unselected package").size() == 3);
		CHECK(ocr.search("root@client:/home/user#").size() == 1);
	}
}
