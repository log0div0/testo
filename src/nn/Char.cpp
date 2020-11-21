
#include "Char.hpp"
#include "TextColorPicker.hpp"

namespace nn {

bool Char::match(char32_t query) {
	for (auto& codepoint: codepoints) {
		if (codepoint == query) {
			return true;
		}
	}
	return false;
}

bool Char::match_foreground(const stb::Image<stb::RGB>* image, const std::string& color) {
	if (!color.size()) {
		return true;
	}
	if (!foreground.size()) {
		TextColorPicker::instance().run(image, *this);
	}
	return foreground == color;
}

bool Char::match_background(const stb::Image<stb::RGB>* image, const std::string& color) {
	if (!color.size()) {
		return true;
	}
	if (!background.size()) {
		TextColorPicker::instance().run(image, *this);
	}
	return background == color;
}

}
