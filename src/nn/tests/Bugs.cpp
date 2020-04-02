
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Bugs") {
	{
		stb::Image image("Bugs/0.png");
		nn::OCR ocr(&image);
		CHECK(ocr.search("Все пакеты имеют последние версии").size() == 1);
	}
}
