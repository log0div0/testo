
#include "TextRecognizer.hpp"
#include <iostream>
#include <algorithm>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>
#include <cmath>

#define IN_H 32

std::vector<std::u32string> symbols = {
	U"0OoОо",
	U"1",
	U"2",
	U"3ЗзЭэ",
	U"4",
	U"5",
	U"6б",
	U"7",
	U"8",
	U"9",
	U"!",
	U"?",
	U"#",
	U"$",
	U"%",
	U"&",
	U"@",
	U"([{",
	U"<",
	U")]}",
	U">",
	U"+",
	U"-",
	U"*",
	U"/",
	U"\\",
	U".,",
	U":;",
	U"\"'",
	U"^",
	U"~",
	U"=",
	U"|lI",
	U"_",
	U"AА",
	U"aа",
	U"BВв",
	U"bЬьЪъ",
	U"CcСс",
	U"D",
	U"d",
	U"EЕЁ",
	U"eеё",
	U"F",
	U"f",
	U"G",
	U"g",
	U"HНн",
	U"h",
	U"i",
	U"J",
	U"j",
	U"KКк",
	U"k",
	U"L",
	U"MМм",
	U"m",
	U"N",
	U"n",
	U"PpРр",
	U"R",
	U"r",
	U"Q",
	U"q",
	U"Ss",
	U"TТт",
	U"t",
	U"U",
	U"u",
	U"Vv",
	U"Ww",
	U"XxХх",
	U"Y",
	U"yУу",
	U"Zz",
	U"Б",
	U"Гг",
	U"Дд",
	U"Жж",
	U"ИиЙй",
	U"Лл",
	U"Пп",
	U"Фф",
	U"Цц",
	U"Чч",
	U"ШшЩщ",
	U"Ыы",
	U"Юю",
	U"Яя"
};

namespace nn {

TextRecognizer& TextRecognizer::instance() {
	static TextRecognizer instance;
	return instance;
}

TextRecognizer::TextRecognizer() {
	for (size_t i = 0; i < symbols.size(); ++i) {
		symbols_indexes.push_back(i);
	}
}

void TextRecognizer::recognize(const stb::Image<stb::RGB>* image, TextLine& textline) {
	run_nn(image, textline);
	return run_postprocessing(textline);
}

void TextRecognizer::run_nn(const stb::Image<stb::RGB>* image, TextLine& textline) {

	int ratio = ceilf(float(textline.rect.width()) / float(textline.rect.height()));
	int new_in_w = ratio * IN_H * 2;

	if (in_w != new_in_w) {
		in_w = new_in_w;
		in_c = 3;
		out_w = new_in_w / 4 + 1;
		out_c = symbols.size() + 1;

		in.resize(in_w, IN_H, in_c);
		out.resize(out_w, out_c);
	}

	stb::Image<stb::RGB> textline_img = image->sub_img(
		textline.rect.left, textline.rect.top,
		textline.rect.width(), textline.rect.height()
	).resize(in_w, IN_H);

	in.set(textline_img, true);

	model.run({&in}, {&out});
}

#define THRESHOLD -10.0

void TextRecognizer::run_postprocessing(TextLine& textline) {
	float ratio = float(textline.rect.width()) / out_w;
	int prev_max_pos = -1;
	for (int x = 0; x < out_w; ++x) {
		int max_pos = -1;
		float max_value = std::numeric_limits<float>::lowest();
		for (int c = 0; c < out_c; ++c) {
			if (max_value < out[x][c]) {
				max_value = out[x][c];
				max_pos = c;
			}
		}
		if (max_pos == 0) {
			prev_max_pos = max_pos;
			continue;
		}
		if (prev_max_pos == max_pos) {
			textline.chars.back().rect.right = textline.rect.left + std::ceil(x * ratio);
			continue;
		}
		prev_max_pos = max_pos;

		std::sort(symbols_indexes.begin(), symbols_indexes.end(), [&](size_t a, size_t b) {
			return out[x][a + 1] > out[x][b + 1];
		});

		auto end = std::find_if(symbols_indexes.begin(), symbols_indexes.end(), [&](size_t code) {
			return out[x][code + 1] < THRESHOLD;
		});

		Char char_;
		char_.rect.top = textline.rect.top;
		char_.rect.bottom = textline.rect.bottom;
		char_.rect.left = textline.rect.left + std::floor(x * ratio);
		char_.rect.right = textline.rect.left + std::ceil(x * ratio);
		for (auto it = symbols_indexes.begin(); it != end; ++it) {
			char_.codepoints += symbols.at(*it);
		}
		if (char_.codepoints.size() == 0) {
			throw std::runtime_error("TextRecognizer error");
		}
		textline.chars.push_back(char_);
	}
	for (size_t i = 1; i < textline.chars.size(); ++i) {
		int pos = (textline.chars[i-1].rect.right + textline.chars[i].rect.right) / 2;
		textline.chars[i-1].rect.right = textline.chars[i].rect.left = pos;
	}
	if (textline.chars.size()) {
		textline.chars.front().rect.left = textline.rect.left;
		textline.chars.back().rect.right = textline.rect.right;
	}
}

}
