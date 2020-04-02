
#pragma once

#include "Char.hpp"

namespace nn {

struct Word {
	Rect rect;
};

struct TextLine {
	Rect rect;
	std::vector<Char> chars;
	std::vector<Word> words; // tmp

	std::vector<TextLine> match(const std::string& text);
	bool match_foreground(const stb::Image* image, const std::string& color);
	bool match_background(const stb::Image* image, const std::string& color);
};

}
