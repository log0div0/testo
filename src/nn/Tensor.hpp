
#pragma once

#include "TextLine.hpp"
#include <stdexcept>

namespace nn {

struct ErrorContinue: std::runtime_error {
	ErrorContinue(const char* reason): std::runtime_error(reason) {}
};

struct Tensor {
	std::vector<TextLine> textlines;

	size_t size() const {
		return textlines.size();
	}

	Tensor match(const std::string& text);
	Tensor match_foreground(const stb::Image* image, const std::string& color);
	Tensor match_background(const stb::Image* image, const std::string& color);

	Tensor from_top(size_t i) const;
	Tensor from_bottom(size_t i) const;
	Tensor from_left(size_t i) const;
	Tensor from_right(size_t i) const;

	int32_t x() const;
	int32_t y() const;

	Point left_top() const;
	Point left_bottom() const;
	Point right_top() const;
	Point right_bottom() const;
	Point center() const;
	Point center_top() const;
	Point center_bottom() const;
	Point left_center() const;
	Point right_center() const;
};

}
