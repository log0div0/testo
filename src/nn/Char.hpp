
#pragma once

#include "Rect.hpp"
#include <string>
#include <vector>
#include <stb/Image.hpp>

namespace nn {

struct Char {
	Rect rect;
	std::vector<std::string> codes;
	std::string foreground;
	std::string background;

	bool match(const std::string& text);
	bool match_foreground(const stb::Image<stb::RGB>* image, const std::string& color);
	bool match_background(const stb::Image<stb::RGB>* image, const std::string& color);
};

}
