
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Windows 7/Установка") {
	nn::OCR& ocr = nn::OCR::instance();

	{
		auto result = ocr.run("Windows 7/Установка/Выбор языка.png");
		CHECK(result.search("My language is English").size() == 1);
		CHECK(result.search("Мой язык - русский").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Выбор языка 2.png");
		CHECK(result.search("Устанавливаемый язык:").size() == 1);
		CHECK(result.search("Формат времени и денежных единиц:").size() == 1);
		CHECK(result.search("Раскладка клавиатуры или метод ввода:").size() == 1);
		CHECK(result.search("Выберите нужный язык и другие параметры, а затем нажмите кнопку \"Далее\".").size() == 1);
	}
	{
		auto result = ocr.run("Windows 7/Установка/Windows завершает применение параметров.png");
		CHECK(result.search("Настройка Windows").size() == 1);
		CHECK(result.search("Windows завершает применение параметров").size() == 1);
	}
}
