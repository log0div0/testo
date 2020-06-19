
#include <catch.hpp>
#include "nn/OCR.hpp"

TEST_CASE("CentOS 7/Установка") {
	{
		stb::Image image("CentOS 7/Установка/Задание пароля.png");
		nn::Tensor tensor = nn::find_text(&image);
		CHECK(tensor.match("ROOT PASSWORD").size() == 1);
		CHECK(tensor.match("Root password").size() == 1);
	}
}
