
#pragma once

#include "Rect.hpp"
#include <string>
#include <vector>

namespace nn {

struct Char {
	std::vector<std::string> alternatives;
	bool match(const std::string& query);
};

struct Word {
	Rect rect;
	std::vector<Char> chars;

	bool match(const std::vector<std::string>& query);
	bool match_begin(const std::vector<std::string>& query);
	bool match_end(const std::vector<std::string>& query);
};

struct TextLine {
	Rect rect;
	std::vector<Word> words;
};

}
