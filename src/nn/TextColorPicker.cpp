
#include "TextColorPicker.hpp"
#include <iostream>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>
#include <cmath>

#define IN_H 32
#define IN_W 16

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

void TextColorPicker::run(const stb::Image<stb::RGB>* image, Char& char_) {
	run_nn(image, char_);
	return run_postprocessing(char_);
}

void TextColorPicker::run_nn(const stb::Image<stb::RGB>* image, const Char& char_) {

	if (!in_c || !out_c) {
		in_c = 3;
		out_c = colors.size() + colors.size();

		in.resize(IN_W, IN_H, in_c);
		out.resize(out_c);
	}

	stb::Image<stb::RGB> char_img = image->sub_img(
		char_.rect.left, char_.rect.top,
		char_.rect.width(), char_.rect.height()
	).resize(IN_W, IN_H);

	in.set(char_img, false);

	model.run({&in}, {&out});
}

void TextColorPicker::run_postprocessing(Char& char_) {
	{
		int max_pos = -1;
		float max_value = std::numeric_limits<float>::lowest();
		for (size_t i = 0; i < colors.size(); ++i) {
			if (max_value < out[i]) {
				max_value = out[i];
				max_pos = i;
			}
		}
		char_.foreground = colors.at(max_pos);
	}
	{
		int max_pos = -1;
		float max_value = std::numeric_limits<float>::lowest();
		for (size_t i = 0; i < colors.size(); ++i) {
			if (max_value < out[colors.size()+i]) {
				max_value = out[colors.size()+i];
				max_pos = i;
			}
		}
		char_.background = colors.at(max_pos);
	}
}

}
