
#pragma once

#include "Rect.hpp"
#include <stb/Image.hpp>
#include <string>
#include <vector>

namespace nn {

struct Char {
	const stb::Image* image = nullptr;
	Rect rect;
	std::vector<std::string> codes;

	bool match(const std::string& query);
};

struct Word {
	const stb::Image* image = nullptr;
	Rect rect;
};

struct TextLine {
	const stb::Image* image = nullptr;
	Rect rect;
	std::vector<Char> chars;
	std::vector<Word> words; // tmp

	std::vector<Rect> search(const std::vector<std::string>& query);
};

struct OCR {
	const stb::Image* image = nullptr;
	std::vector<TextLine> textlines;

	OCR(const stb::Image* image_);
	std::vector<Rect> search(const std::string& query, const std::string& color = {}, const std::string& backgroundColor = {});
};

}
