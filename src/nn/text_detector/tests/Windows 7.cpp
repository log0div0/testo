
#include <catch.hpp>
#include "../TextDetector.hpp"

TEST_CASE("Windows 7/Установка") {
	TextDetector& detector = TextDetector::instance();

	{
		stb::Image img("Windows 7/Установка/Выбор языка.png");
		CHECK(detector.detect(img, "My language is English").size() == 1);
		CHECK(detector.detect(img, "Мой язык - русский").size() == 1);
	}
	{
		stb::Image img("Windows 7/Установка/Выбор языка 2.png");
		CHECK(detector.detect(img, "Устанавливаемый язык:").size() == 1);
		CHECK(detector.detect(img, "Формат времени и денежных единиц:").size() == 1);
		CHECK(detector.detect(img, "Раскладка клавиатуры или метод ввода:").size() == 1);
		CHECK(detector.detect(img, "Выберите нужный язык и другие параметры, а затем нажмите кнопку \"Далее\".").size() == 1);
	}
}
