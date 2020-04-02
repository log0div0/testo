
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Ubuntu Server 18.04/Установка") {
	{
		stb::Image image("Ubuntu Server 18.04/Установка/Выбор языка.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Language").size() == 2);
		CHECK(tensor.match(&image, "English").size() == 1);
		CHECK(tensor.match(&image, "Русский").size() == 1);
		CHECK(tensor.match(&image, "F1 Help").size() == 1);
		CHECK(tensor.match(&image, "F2 Language").size() == 1);
		CHECK(tensor.match(&image, "F3 Keymap").size() == 1);
		CHECK(tensor.match(&image, "F4 Modes").size() == 1);
		CHECK(tensor.match(&image, "F5 Accessibility").size() == 1);
		CHECK(tensor.match(&image, "F6 Other Options").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Начальный экран.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Install Ubuntu Server").size() == 2);
		CHECK(tensor.match(&image, "Install Ubuntu Server with the HWE kernel").size() == 1);
		CHECK(tensor.match(&image, "Check disc for defects").size() == 1);
		CHECK(tensor.match(&image, "Test memory").size() == 1);
		CHECK(tensor.match(&image, "Boot from first hard disk").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Опять выбор языка.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Welcome!").size() == 1);
		CHECK(tensor.match(&image, "Добро пожаловать!").size() == 1);
		CHECK(tensor.match(&image, "Please choose your preferred language.").size() == 1);
		CHECK(tensor.match(&image, "English").size() == 1);
		CHECK(tensor.match(&image, "Русский").size() == 1);
		CHECK(tensor.match(&image, "Use UP, DOWN and ENTER keys to select your language.").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Приветствие.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Install Ubuntu").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Настройка сети.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Network connections").size() == 1);
		CHECK(tensor.match(&image, "192.168.122.219/24").size() == 1);
		CHECK(tensor.match(&image, "52:54:00:45:12:e7").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Настройка прокси.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Configure proxy").size() == 1);
		CHECK(tensor.match(&image, "\"http://[[user][:pass]@]host[:port]/\"").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Установка закончена.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Installation complete!").size() == 1);
		CHECK(tensor.match(&image, "Reboot Now").size() == 1);
	}

	{
		stb::Image image("Ubuntu Server 18.04/Установка/Пожалуйста, извлеките CD-ROM.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Please remove the installation medium, then press ENTER:").size() == 1);
	}
}

TEST_CASE("Ubuntu Server 18.04/Консоль") {
	{
		stb::Image image("Ubuntu Server 18.04/Консоль/Установка питона.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match(&image, "Setting up daemon").size() == 1);
		CHECK(tensor.match(&image, "Unpacking python").size() == 2);
		CHECK(tensor.match(&image, "Unpacking python2.7").size() == 1);
		CHECK(tensor.match(&image, "Selecting previously unselected package").size() == 3);
		CHECK(tensor.match(&image, "root@client:/home/user#").size() == 1);
	}
}
