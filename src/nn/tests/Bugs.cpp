
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Bugs") {
	{
		stb::Image image("Bugs/0.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Все пакеты имеют последние версии").size() == 1);
	}
}
