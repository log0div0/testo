
#pragma once

#include "nlohmann/json.hpp"

#include "Rect.hpp"
#include "TextRecognizerCache.hpp"

namespace nn {

struct TextLine {
	Rect rect;

	TextRecognizerCache text_recognizer_cache;
};

inline void to_json(nlohmann::json& j, const nn::TextLine& textline) {
	return to_json(j, textline.rect);
}

inline void from_json(const nlohmann::json& j, nn::TextLine& textline) {
	return from_json(j, textline.rect);
}

}
