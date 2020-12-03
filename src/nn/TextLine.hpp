
#pragma once

#include "Rect.hpp"
#include "TextRecognizerCache.hpp"

namespace nn {

struct TextLine {
	Rect rect;

	TextRecognizerCache text_recognizer_cache;
};

}
