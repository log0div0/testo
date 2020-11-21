
#pragma once

#include "Char.hpp"

namespace nn {

struct TextLine {
	Rect rect;
	std::vector<Char> chars;

	std::vector<TextLine> match(const std::string& text);
	bool match_foreground(const stb::Image<stb::RGB>* image, const std::string& color);
	bool match_background(const stb::Image<stb::RGB>* image, const std::string& color);
};

}
