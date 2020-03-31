
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Windows 10/Установка") {
	{
		stb::Image image("Windows 10/Установка/Продолжить на выбранном языке.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Продолжить на выбранном языке?").size() == 1);
		CHECK(ocr.search("Да").size() == 1);
		CHECK(ocr.search("Русский").size() == 1);
		CHECK(ocr.search("English (United States)").size() == 1);
		CHECK(ocr.search("Добро пожаловать").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Давайте начнём с региона.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Давайте начнем с региона. Это правильно?").size() == 1);
		CHECK(ocr.search("Парагвай").size() == 1);
		CHECK(ocr.search("Перу").size() == 1);
		CHECK(ocr.search("Польша").size() == 1);
		CHECK(ocr.search("Португалия").size() == 1);
		CHECK(ocr.search("Пуэрто-Рико").size() == 1);
		CHECK(ocr.search("Реюньон").size() == 1);
		CHECK(ocr.search("Россия").size() == 1);
		CHECK(ocr.search("Да").size() == 2);
		CHECK(ocr.search("Основы").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Лицензионное соглашение о Windows 10.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Лицензионное соглашение о Windows 10").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Если вы подключитесь к Интернету.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Если вы подключитесь к Интернету").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Войдите с помощью учетной записи.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Войдите с помощью учетной записи").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Кто будет использовать этот компьютер.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Кто будет использовать этот компьютер").size() == 1);
	}
	{
		stb::Image image("Windows 10/Установка/Удобная работа на разных устройствах.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Удобная работа на разных устройствах").size() == 1);
	}
}
