
#pragma once

#include "TextLine.hpp"
#include <stb/Image.hpp>

namespace nn {

struct TextColorPicker {
	static TextColorPicker& instance();

	TextColorPicker(const TextColorPicker&) = delete;
	TextColorPicker& operator=(const TextColorPicker&) = delete;

	bool run(const stb::Image<stb::RGB>* image, const TextLine& textline, const std::string& fg, const std::string& bg);

private:
	TextColorPicker() = default;
};

}
