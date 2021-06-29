
#pragma once

#include "Rect.hpp"

namespace nn {

struct Img {
	Rect rect;
};

inline void to_json(nlohmann::json& j, const nn::Img& img) {
	return to_json(j, img.rect);
}

inline void from_json(const nlohmann::json& j, nn::Img& img) {
	return from_json(j, img.rect);
}

}
