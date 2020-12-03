
#include "TextColorPicker.hpp"
#include <vector>

std::vector<std::string> colors = {
	"white",
	"gray",
	"black",
	"red",
	"orange",
	"yellow",
	"green",
	"cyan",
	"blue",
	"purple"
};

namespace nn {

TextColorPicker& TextColorPicker::instance() {
	static TextColorPicker instance;
	return instance;
}

bool TextColorPicker::run(const stb::Image<stb::RGB>* image, const TextLine& textline, const std::string& fg, const std::string& bg) {
	throw std::runtime_error("Implement TextColorPicker::run");
}

}
