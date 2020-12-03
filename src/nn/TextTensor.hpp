
#pragma once

#include "TextLine.hpp"
#include "Tensor.hpp"
#include <stb/Image.hpp>

namespace nn {

struct TextTensor: Tensor<TextLine> {
	TextTensor from_left(size_t i) const { return nn::from_left(*this, i); }
	TextTensor from_top(size_t i) const { return nn::from_top(*this, i); }
	TextTensor from_right(size_t i) const { return nn::from_right(*this, i); }
	TextTensor from_bottom(size_t i) const { return nn::from_bottom(*this, i); }

	TextTensor match_text(const stb::Image<stb::RGB>* image, const std::string& text);
	TextTensor match_color(const stb::Image<stb::RGB>* image, const std::string& fg, const std::string& bg);
};

TextTensor find_text(const stb::Image<stb::RGB>* image);

}
