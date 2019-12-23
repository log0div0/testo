
#pragma once

#include "Word.hpp"

namespace nn {

struct TextLine {
	Rect rect;
	std::string text;
	std::vector<Word> words;
};

}
