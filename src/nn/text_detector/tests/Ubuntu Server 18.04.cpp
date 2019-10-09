
#include <catch.hpp>
#include "../TextDetector.hpp"

TEST_CASE("Ubuntu Server 18.04/Установка") {
	TextDetector detector;

	{
		stb::Image img("Ubuntu Server 18.04/Установка/01 - Выбор языка.png");
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
		stb::Image img("Ubuntu Server 18.04/Установка/02 - Начальный экран.png");
		CHECK(detector.detect(img, "Install Ubuntu Server").size() == 2);
		CHECK(detector.detect(img, "Install Ubuntu Server with the HWE kernel").size() == 1);
		CHECK(detector.detect(img, "Check disc for defects").size() == 1);
		CHECK(detector.detect(img, "Test memory").size() == 1);
		CHECK(detector.detect(img, "Boot from first hard disk").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/03 - Опять выбор языка.png");
		CHECK(detector.detect(img, "Welcome !").size() == 1);
		CHECK(detector.detect(img, "Добро пожаловать !").size() == 1);
		CHECK(detector.detect(img, "Please choose your preferred language.").size() == 1);
		CHECK(detector.detect(img, "English").size() == 1);
		CHECK(detector.detect(img, "Русский").size() == 1);
		CHECK(detector.detect(img, "Use UP, DOWN and ENTER keys to select your language.").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/04 - Выбор раскладки.png");
		CHECK(detector.detect(img, "Keyboard configuration").size() == 1);
		CHECK(detector.detect(img, "English (US)").size() == 2);
		CHECK(detector.detect(img, "Done").size() == 1);
		CHECK(detector.detect(img, "Back").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/05 - MAAS.png");
		CHECK(detector.detect(img, "Ubuntu 18.04").size() == 1);
		CHECK(detector.detect(img, "Welcome to Ubuntu!").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/06 - Настройка сети.png");
		CHECK(detector.detect(img, "Network connections").size() == 1);
		CHECK(detector.detect(img, "192.168.122.219/24").size() == 1);
		CHECK(detector.detect(img, "52:54:00:45:12:e7").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/07 - Настройка прокси.png");
		CHECK(detector.detect(img, "Configure proxy").size() == 1);
		CHECK(detector.detect(img, "\"http://[[user][:pass]@]host[:port]/\"").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/08 - Настройка зеркала.png");
		CHECK(detector.detect(img, "Configure Ubuntu archive mirror").size() == 1);
		CHECK(detector.detect(img, "http://archive.ubuntu.com/ubuntu").size() == 2);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/09 - Настройка диска.png");
		CHECK(detector.detect(img, "Filesystem setup").size() == 1);
		CHECK(detector.detect(img, "Use An Entire Disk").size() == 2);
		CHECK(detector.detect(img, "Use An Entire Disk And Set Up LVM").size() == 1);
		CHECK(detector.detect(img, "Manual").size() == 3);
		CHECK(detector.detect(img, "Back").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/10 - Выбор диска.png");
		CHECK(detector.detect(img, "Filesystem setup").size() == 1);
		CHECK(detector.detect(img, "QEMU_HARDDISK_QM00001").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/11 - Сводка.png");
		CHECK(detector.detect(img, "Filesystem setup").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/12 - Подтверждение на затирание диска.png");
		CHECK(detector.detect(img, "Confirm destructive action").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/13 - Настройка пользователя.png");
		CHECK(detector.detect(img, "Profile setup").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/14 - Настройка SSH.png");
		CHECK(detector.detect(img, "SSH Setup").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/15 - Выбор дополнительных пакетов для установки.png");
		CHECK(detector.detect(img, "Featured Server Snaps").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/16 - Установка закончена.png");
		CHECK(detector.detect(img, "Installation complete!").size() == 1);
		CHECK(detector.detect(img, "Reboot Now").size() == 1);
	}

	{
		stb::Image img("Ubuntu Server 18.04/Установка/17 - Извлеките CD-ROM.png");
		CHECK(detector.detect(img, "Please remove the installation medium, then press ENTER:").size() == 1);
	}
}
