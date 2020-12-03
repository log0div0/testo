
#include "TextRecognizer.hpp"
#include <iostream>
#include <algorithm>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>
#include <cmath>
#include <locale>
#include <codecvt>

#define IN_H 32

std::vector<std::u32string> symbols = {
	U"",
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

std::vector<TextLine> TextRecognizer::recognize(const stb::Image<stb::RGB>* image, const TextLine& textline, const std::string& query) {
	run_nn(image, textline);
	return run_postprocessing(textline, query);
}

void TextRecognizer::run_nn(const stb::Image<stb::RGB>* image, const TextLine& textline) {

	int ratio = ceilf(float(textline.rect.width()) / float(textline.rect.height()));
	int new_in_w = ratio * IN_H * 2;

	if (in_w != new_in_w) {
		in_w = new_in_w;
		in_c = 3;
		out_w = new_in_w / 4 + 1;
		out_c = symbols.size();

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

std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;

struct Matcher {
	struct Prediction {
		bool maybe_blank = false;
		std::u32string codepoints;

		bool match(char32_t codepoint) {
			for (char32_t cp: codepoints) {
				if (cp == codepoint) {
					return true;
				}
			}
			return false;
		}
	};

	std::vector<Prediction> predictions;

	int match(size_t x, const std::u32string& query) {
		size_t y = 0;
		while (x < predictions.size() && (y < query.size())) {
			if (predictions[x].match(query[y])) {
				++y;
				++x;
				continue;
			}
			if ((y > 0) && predictions[x].match(query[y-1])) {
				++x;
				continue;
			}
			if (predictions[x].maybe_blank) {
				++x;
				continue;
			}
			break;
		}
		if (y == query.size()) {
			return x;
		} else {
			return -1;
		}
	}

	void print() {
		for (size_t i = 0; i < predictions.size(); ++i) {
			auto prediction = predictions[i];
			std::cout << i << " ";
			if (prediction.maybe_blank) {
				std::cout << conv.to_bytes(U'•');
			}
			for (char32_t codepoint: prediction.codepoints) {
				std::cout << conv.to_bytes(codepoint);
			}
			std::cout << std::endl;
		}
	}
};

std::vector<TextLine> TextRecognizer::run_postprocessing(const TextLine& textline, const std::string& query) {
	if (query.size() == 0) {
		throw std::runtime_error("Empty query in TextRecognizer");
	}
	if (query.size() > (size_t)out_w) {
		return {};
	}
	// std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
	// static int f = 0;
	// if (f++ != 2) {
	// 	return {};
	// }
	Matcher matcher;
	matcher.predictions.resize(out_w);
	for (int x = 0; x < out_w; ++x) {
		std::sort(symbols_indexes.begin(), symbols_indexes.end(), [&](size_t a, size_t b) {
			return out[x][a] > out[x][b];
		});

		auto end = std::find_if(symbols_indexes.begin(), symbols_indexes.end(), [&](size_t code) {
			return out[x][code] < THRESHOLD;
		});

		if (end == symbols_indexes.begin()) {
			throw std::runtime_error("end == symbols_indexes.begin()");
		}

		for (auto it = symbols_indexes.begin(); it != end; ++it) {
			const std::u32string& tmp = symbols.at(*it);
			if (tmp.size()) {
				matcher.predictions[x].codepoints += tmp;
			} else {
				matcher.predictions[x].maybe_blank = true;
			}
		}
	}

	float ratio = float(textline.rect.width()) / out_w;
	std::u32string u32query;
	for (char32_t ch: conv.from_bytes(query)) {
		if (ch != U' ') {
			u32query.push_back(ch);
		}
	}

	// matcher.print();

	std::vector<TextLine> result;
	for (size_t x = 0; x < (out_w - query.size()); ++x) {
		while (x < (out_w - query.size())) {
			if (matcher.predictions[x].codepoints.size()) {
				break;
			}
			++x;
		}
		int pos = matcher.match(x, u32query);
		if (pos < 0) {
			continue;
		}
		TextLine new_textline;
		new_textline.rect.top = textline.rect.top;
		new_textline.rect.bottom = textline.rect.bottom;
		new_textline.rect.left = textline.rect.left + std::floor(x * ratio);
		new_textline.rect.right = textline.rect.left + std::ceil(pos * ratio);
		result.push_back(new_textline);
		x = pos;
	}
	return result;
}

}
