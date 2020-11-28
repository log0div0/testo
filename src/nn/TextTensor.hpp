
#pragma once

#include "TextLine.hpp"
#include "Tensor.hpp"

namespace nn {

struct TextTensor: Tensor<TextLine> {
	TextTensor from_left(size_t i) const { return nn::from_left(*this, i); }
	TextTensor from_top(size_t i) const { return nn::from_top(*this, i); }
	TextTensor from_right(size_t i) const { return nn::from_right(*this, i); }
	TextTensor from_bottom(size_t i) const { return nn::from_bottom(*this, i); }

	TextTensor match(const std::string& text);
	TextTensor match_foreground(const stb::Image<stb::RGB>* image, const std::string& color);
	TextTensor match_background(const stb::Image<stb::RGB>* image, const std::string& color);
};

TextTensor find_text(const stb::Image<stb::RGB>* image);

}
