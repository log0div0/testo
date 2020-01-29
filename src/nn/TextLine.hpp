
#pragma once

#include "Rect.hpp"
#include <string>
#include <vector>

namespace nn {

struct Char {
	Rect rect;
	std::vector<std::string> alternatives;
	bool match(const std::string& query);
};

struct Word {
	Rect rect;
};

struct TextLine {
	Rect rect;
	std::vector<Char> chars;
	std::vector<Rect> search(const std::vector<std::string>& query);

// tmp
	std::vector<Word> words;
};

}
