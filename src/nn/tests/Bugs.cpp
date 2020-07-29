
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("Bugs") {
	{
		stb::Image image("Bugs/0.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Все пакеты имеют последние версии").size() == 1);
	}
	{
		stb::Image image("Bugs/1.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Введите пароль администратора").size() == 1);
	}
	{
		stb::Image image("Bugs/2.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("Type to search").size() == 1);
	}
}
