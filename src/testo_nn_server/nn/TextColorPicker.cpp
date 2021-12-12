
#include "TextColorPicker.hpp"
#include <vector>
#include <iostream>
#include <cmath>

#define IN_H 32

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
	run_nn(image, textline);
	return run_postprocessing(textline, fg, bg);
}

void TextColorPicker::run_nn(const stb::Image<stb::RGB>* image, const TextLine& textline) {

	int ratio = ceilf(float(textline.rect.width()) / float(textline.rect.height()));
	int new_in_w = ratio * IN_H * 2;

	if (in_w != new_in_w) {
		in_w = new_in_w;
		in_c = 3;
		out_w = new_in_w / 4 + 1;
		out_c = colors.size() * 2;

		in.resize(in_w, IN_H, in_c);
		out.resize(out_w, out_c);
	}

	stb::Image<stb::RGB> textline_img = image->sub_image(
		textline.rect.left, textline.rect.top,
		textline.rect.width(), textline.rect.height()
	).resize(in_w, IN_H);

	in.set(textline_img, true);

	model.run({&in}, {&out});
}

bool TextColorPicker::run_postprocessing(const TextLine& textline, const std::string& fg, const std::string& bg) {
	if (!match_color(textline, fg, 0)) {
		return false;
	}
	if (!match_color(textline, bg, colors.size())) {
		return false;
	}
	return true;
}

int get_color_index(const std::string& color) {
	for (size_t i = 0; i < colors.size(); ++i) {
		if (colors[i] == color) {
			return i;
		}
	}
	throw std::runtime_error("Unknown color: " + color);
}

template <typename T>
int get_max_index(T* t, int size) {
	int max_index = 0;
	for (int i = 1; i < size; ++i) {
		if (t[max_index] < t[i]) {
			max_index = i;
		}
	}
	return max_index;
}

bool TextColorPicker::match_color(const TextLine& textline, const std::string& color, int c_off) {
	if (color.size() == 0) {
		return true;
	}
	std::vector<int> color_scores(colors.size(), 0);
	for (int x = 0; x < out_w; ++x) {
		int max_index = get_max_index(out[x] + c_off, colors.size());
		color_scores[max_index] += 1;
	}

	// for (size_t i = 0; i < color_scores.size(); ++i) {
	// 	std::cout << colors[i] << " " << color_scores[i] << std::endl;
	// }

	int color_index = get_color_index(color);
	return color_scores.at(color_index) != 0;
}

}
