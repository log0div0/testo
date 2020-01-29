
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Ubuntu Server 18.04/Установка") {
	nn::OCR& ocr = nn::OCR::instance();

	{
		auto result = ocr.run("Ubuntu Server 18.04/Установка/Выбор языка.png");
		CHECK(result.search("Language").size() == 2);
		CHECK(result.search("English").size() == 1);
		CHECK(result.search("Русский").size() == 1);
		CHECK(result.search("F1 Help").size() == 1);
		CHECK(result.search("F2 Language").size() == 1);
		CHECK(result.search("F3 Keymap").size() == 1);
		CHECK(result.search("F4 Modes").size() == 1);
		CHECK(result.search("F5 Accessibility").size() == 1);
		CHECK(result.search("F6 Other Options").size() == 1);
	}

	{
		auto result = ocr.run("Ubuntu Server 18.04/Установка/Начальный экран.png");
		CHECK(result.search("Install Ubuntu Server").size() == 2);
		CHECK(result.search("Install Ubuntu Server with the HWE kernel").size() == 1);
		CHECK(result.search("Check disc for defects").size() == 1);
		CHECK(result.search("Test memory").size() == 1);
		CHECK(result.search("Boot from first hard disk").size() == 1);
	}

	{
		auto result = ocr.run("Ubuntu Server 18.04/Установка/Опять выбор языка.png");
		CHECK(result.search("Welcome!").size() == 1);
		CHECK(result.search("Добро пожаловать!").size() == 1);
		CHECK(result.search("Please choose your preferred language.").size() == 1);
		CHECK(result.search("English").size() == 1);
		CHECK(result.search("Русский").size() == 1);
		CHECK(result.search("Use UP, DOWN and ENTER keys to select your language.").size() == 1);
	}

	{
		auto result = ocr.run("Ubuntu Server 18.04/Установка/Приветствие.png");
		CHECK(result.search("Install Ubuntu").size() == 1);
	}

	{
		auto result = ocr.run("Ubuntu Server 18.04/Установка/Настройка сети.png");
		CHECK(result.search("Network connections").size() == 1);
		CHECK(result.search("192.168.122.219/24").size() == 1);
		CHECK(result.search("52:54:00:45:12:e7").size() == 1);
	}

	{
		auto result = ocr.run("Ubuntu Server 18.04/Установка/Настройка прокси.png");
		CHECK(result.search("Configure proxy").size() == 1);
		CHECK(result.search("\"http://[[user][:pass]@]host[:port]/\"").size() == 1);
	}

	{
		auto result = ocr.run("Ubuntu Server 18.04/Установка/Установка закончена.png");
		CHECK(result.search("Installation complete!").size() == 1);
		CHECK(result.search("Reboot Now").size() == 1);
	}

	{
		auto result = ocr.run("Ubuntu Server 18.04/Установка/Пожалуйста, извлеките CD-ROM.png");
		CHECK(result.search("Please remove the installation medium, then press ENTER:").size() == 1);
	}
}

TEST_CASE("Ubuntu Server 18.04/Консоль") {
	nn::OCR& ocr = nn::OCR::instance();

	{
		auto result = ocr.run("Ubuntu Server 18.04/Консоль/Установка питона.png");
		CHECK(result.search("Setting up daemon").size() == 1);
		CHECK(result.search("Unpacking python").size() == 2);
		CHECK(result.search("Unpacking python2.7").size() == 1);
		CHECK(result.search("Selecting previously unselected package").size() == 3);
		CHECK(result.search("root@client:/home/user#").size() == 1);
	}
}
