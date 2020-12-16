
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

namespace nn {

const std::vector<std::u32string> TextRecognizer::symbols = {
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
	U"-_",
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

TextRecognizer& TextRecognizer::instance() {
	static TextRecognizer instance;
	return instance;
}

TextRecognizer::TextRecognizer() {
	for (size_t i = 0; i < symbols.size(); ++i) {
		symbols_indexes.push_back(i);
	}
}

std::vector<TextLine> TextRecognizer::recognize(const stb::Image<stb::RGB>* image, TextLine& textline, const std::string& query) {
	run_nn(image, textline);
	return run_postprocessing(textline, query);
}

#define THRESHOLD -11.0

void TextRecognizer::run_nn(const stb::Image<stb::RGB>* image, TextLine& textline) {
	if (textline.text_recognizer_cache.predictions.size()) {
		return;
	}

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

	stb::Image<stb::RGB> textline_img = image->sub_image(
		textline.rect.left, textline.rect.top,
		textline.rect.width(), textline.rect.height()
	).resize(in_w, IN_H);

	in.set(textline_img, true);

	model.run({&in}, {&out});

	textline.text_recognizer_cache.predictions.resize(out_w);
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

		if ((end - symbols_indexes.begin()) > 5) {
			end = symbols_indexes.begin() + 5;
		}

		for (auto it = symbols_indexes.begin(); it != end; ++it) {
			const std::u32string& tmp = symbols.at(*it);
			if (tmp.size()) {
				textline.text_recognizer_cache.predictions[x].codepoints += tmp;
			} else {
				textline.text_recognizer_cache.predictions[x].maybe_blank = true;
			}
		}
	}
}

static std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;

std::vector<TextLine> TextRecognizer::run_postprocessing(const TextLine& textline, const std::string& query) {
	const TextRecognizerCache& cache = textline.text_recognizer_cache;

	if (query.size() == 0) {
		throw std::runtime_error("Empty query in TextRecognizer");
	}
	if (query.size() > cache.predictions.size()) {
		return {};
	}

	// static int f = 0;
	// std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << (f++) << std::endl;
	// if (f != 12) {
	// 	return {};
	// }
	// cache.print();

	float ratio = float(textline.rect.width()) / cache.predictions.size();
	std::u32string u32query;
	for (char32_t ch: conv.from_bytes(query)) {
		if (ch != U' ') {
			u32query.push_back(ch);
		}
	}

	std::vector<TextLine> result;
	for (size_t x = 0; x < (cache.predictions.size() - query.size()); ++x) {
		while (x < (cache.predictions.size() - query.size())) {
			if (cache.predictions[x].codepoints.size()) {
				break;
			}
			++x;
		}
		int pos = cache.match(x, u32query);
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
