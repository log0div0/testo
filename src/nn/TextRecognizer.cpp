
#include "TextRecognizer.hpp"
#include <iostream>
#include <algorithm>
#include <utf8.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>
#include <cmath>

#define IN_H 32

std::vector<std::string> char_groups = {
	"0OoОо",
	"1",
	"2",
	"3ЗзЭэ",
	"4",
	"5",
	"6б",
	"7",
	"8",
	"9",
	"!",
	"?",
	"#",
	"$",
	"%",
	"&",
	"@",
	"([{",
	"<",
	")]}",
	">",
	"+",
	"-",
	"*",
	"/",
	"\\",
	".,",
	":;",
	"\"'",
	"^",
	"~",
	"=",
	"|lI",
	"_",
	"AА",
	"aа",
	"BВв",
	"bЬьЪъ",
	"CcСс",
	"D",
	"d",
	"EЕЁ",
	"eеё",
	"F",
	"f",
	"G",
	"g",
	"HНн",
	"h",
	"i",
	"J",
	"j",
	"KКк",
	"k",
	"L",
	"MМм",
	"m",
	"N",
	"n",
	"PpРр",
	"R",
	"r",
	"Q",
	"q",
	"Ss",
	"TТт",
	"t",
	"U",
	"u",
	"Vv",
	"Ww",
	"XxХх",
	"Y",
	"yУу",
	"Zz",
	"Б",
	"Гг",
	"Дд",
	"Жж",
	"ИиЙй",
	"Лл",
	"Пп",
	"Фф",
	"Цц",
	"Чч",
	"ШшЩщ",
	"Ыы",
	"Юю",
	"Яя"
};

namespace nn {

TextRecognizer& TextRecognizer::instance() {
	static TextRecognizer instance;
	return instance;
}

TextRecognizer::TextRecognizer() {
	for (size_t i = 0; i < char_groups.size(); ++i) {
		symbols.push_back(utf8::split_to_chars(char_groups[i]));
		symbols_indexes.push_back(i);
	}
	session = LoadModel("TextRecognizer");
}

TextRecognizer::~TextRecognizer() {

}

void TextRecognizer::recognize(const stb::Image<stb::RGB>* image, TextLine& textline) {
	run_nn(image, textline);
	return run_postprocessing(textline);
}

void TextRecognizer::run_nn(const stb::Image<stb::RGB>* image, TextLine& textline) {

	float ratio = float(textline.rect.width()) / float(textline.rect.height());
	int new_in_w = std::floor(ratio * IN_H);
	if (new_in_w % 2) {
		++new_in_w;
	}
	new_in_w *= 2;

	if (in_w != new_in_w) {
		in_w = new_in_w;
		in_c = 3;
		out_w = new_in_w / 4 + 1;
		out_c = symbols.size() + 1;

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

		std::array<int64_t, 4> in_shape = {1, in_c, IN_H, in_w};
		std::array<int64_t, 3> out_shape = {out_w, 1, out_c};

		in.resize(in_c * IN_H * in_w);
		out.resize(out_w * out_c);

		in_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, in.data(), in.size(), in_shape.data(), in_shape.size()));
		out_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, out.data(), out.size(), out_shape.data(), out_shape.size()));
	}

	int textline_h = textline.rect.height();
	int textline_w = textline.rect.width();
	textline_img.resize(textline_h * textline_w * in_c);
	for (int y = 0; y < textline_h; ++y) {
		for (int x = 0; x < textline_w; ++x) {
			for (int c = 0; c < in_c; ++c) {
				int src_index = (textline.rect.top + y) * image->w * image->c + (textline.rect.left + x) * image->c + c;
				int dst_index = y * textline_w * in_c + x * in_c + c;
				textline_img[dst_index] = image->data[src_index];
			}
		}
	}

	textline_img_resized.resize(IN_H * in_w * in_c);
	if (!stbir_resize_uint8(
		textline_img.data(), textline_w, textline_h, 0,
		textline_img_resized.data(), in_w, IN_H, 0,
		in_c)
	) {
		throw std::runtime_error("stbir_resize_uint8 failed");
	}

	// std::string path = "tmp/" + std::to_string(b) + ".png";
	// if (!stbi_write_png(path.c_str(), in_w, IN_H, in_c, textline_img_resized.data(), in_w*in_c)) {
	// 	throw std::runtime_error("Cannot save image " + path + " because " + stbi_failure_reason());
	// }

	for (int y = 0; y < IN_H; ++y) {
		for (int x = 0; x < in_w; ++x) {
			for (int c = 0; c < in_c; ++c) {
				int src_index = y * in_w * in_c + x * in_c + c;
				int dst_index = c * IN_H * in_w + y * in_w + x;
				in[dst_index] = float(textline_img_resized[src_index]) / 255.0;
			}
		}
	}

	const char* in_names[] = {"input"};
	const char* out_names[] = {"output"};

	session->Run(Ort::RunOptions{nullptr}, in_names, &*in_tensor, 1, out_names, &*out_tensor, 1);
}

#define THRESHOLD -10.0

void TextRecognizer::run_postprocessing(TextLine& textline) {
	float ratio = float(textline.rect.width()) / out_w;
	int prev_max_pos = -1;
	for (int x = 0; x < out_w; ++x) {
		int max_pos = -1;
		float max_value = std::numeric_limits<float>::lowest();
		for (int c = 0; c < out_c; ++c) {
			int index = x * out_c + c;
			if (max_value < out[index]) {
				max_value = out[index];
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
			return out[x * out_c + a + 1] > out[x * out_c + b + 1];
		});

		auto end = std::find_if(symbols_indexes.begin(), symbols_indexes.end(), [&](size_t code) {
			return out[x * out_c + code + 1] < THRESHOLD;
		});

		Char char_;
		char_.rect.top = textline.rect.top;
		char_.rect.bottom = textline.rect.bottom;
		char_.rect.left = textline.rect.left + std::floor(x * ratio);
		char_.rect.right = textline.rect.left + std::ceil(x * ratio);
		for (auto it = symbols_indexes.begin(); it != end; ++it) {
			for (auto& code: symbols.at(*it)) {
				char_.codes.push_back(code);
			}
		}
		if (char_.codes.size() == 0) {
			throw std::runtime_error("TextRecognizer error");
		}
		textline.chars.push_back(char_);
	}
	for (size_t i = 1; i < textline.chars.size(); ++i) {
		textline.chars[i-1].rect.right = textline.chars[i].rect.left;
	}
	if (textline.chars.size()) {
		textline.chars.back().rect.right = textline.rect.right;
	}
}

}
