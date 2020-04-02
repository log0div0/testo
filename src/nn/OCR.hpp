
#pragma once

#include "Rect.hpp"
#include <stb/Image.hpp>
#include <string>
#include <vector>

namespace nn {

struct Char {
	Rect rect;
	std::vector<std::string> codes;
	std::string foreground;
	std::string background;

	bool match(const std::string& text);
	bool match_foreground(const stb::Image* image, const std::string& color);
	bool match_background(const stb::Image* image, const std::string& color);
};

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

struct Tensor {
	std::vector<TextLine> textlines;

	size_t size() const {
		return textlines.size();
	}

	Tensor match(const std::string& text);
	Tensor match_foreground(const stb::Image* image, const std::string& color);
	Tensor match_background(const stb::Image* image, const std::string& color);

	std::vector<Rect> rects() const;
};

Tensor find_text(const stb::Image* image);

}
