
#include "TextRecognizer.hpp"
#include <iostream>
#include <algorithm>
#include <utf8.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>
#include <cmath>

extern unsigned char TextRecognizer_onnx[];
extern unsigned int TextRecognizer_onnx_len;

#define IN_H 32

std::string common = "0123456789!?\"'#$%&@()[]{}<>+-*/\\.,:;^~=|_";
std::string english = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
std::string russian = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";
std::string alphabet = common + english + russian;

namespace nn {

TextRecognizer::TextRecognizer() {
	symbols = utf8::split_to_chars(alphabet);
	for (size_t i = 0; i < symbols.size(); ++i) {
		symbols_indexes.push_back(i);
	}
	session = LoadModel(TextRecognizer_onnx, TextRecognizer_onnx_len);
}

TextRecognizer::~TextRecognizer() {

}

std::vector<Char> TextRecognizer::recognize(const stb::Image& image, Word& word) {
	if (!image.data) {
		return {};
	}

	run_nn(image, word);
	return decode_word(word);
}

void TextRecognizer::run_nn(const stb::Image& image, const Word& word) {

	float ratio = float(word.rect.width()) / float(word.rect.height());
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

	int word_h = word.rect.height();
	int word_w = word.rect.width();
	word_grey.resize(word_h * word_w * in_c);
	for (int y = 0; y < word_h; ++y) {
		for (int x = 0; x < word_w; ++x) {
			for (int c = 0; c < in_c; ++c) {
				int src_index = (word.rect.top + y) * image.width * image.channels + (word.rect.left + x) * image.channels + c;
				int dst_index = y * word_w * in_c + x * in_c + c;
				word_grey[dst_index] = image.data[src_index];
			}
		}
	}

	word_grey_resized.resize(IN_H * in_w * in_c);
	if (!stbir_resize_uint8(
		word_grey.data(), word_w, word_h, 0,
		word_grey_resized.data(), in_w, IN_H, 0,
		in_c)
	) {
		throw std::runtime_error("stbir_resize_uint8 failed");
	}

	// std::string path = "tmp/" + std::to_string(b) + ".png";
	// if (!stbi_write_png(path.c_str(), in_w, IN_H, in_c, word_grey_resized.data(), in_w*in_c)) {
	// 	throw std::runtime_error("Cannot save image " + path + " because " + stbi_failure_reason());
	// }

	for (int y = 0; y < IN_H; ++y) {
		for (int x = 0; x < in_w; ++x) {
			for (int c = 0; c < in_c; ++c) {
				int src_index = y * in_w * in_c + x * in_c + c;
				int dst_index = c * IN_H * in_w + y * in_w + x;
				in[dst_index] = float(word_grey_resized[src_index]) / 255.0;
			}
		}
	}

	const char* in_names[] = {"input"};
	const char* out_names[] = {"output"};

	session->Run(Ort::RunOptions{nullptr}, in_names, &*in_tensor, 1, out_names, &*out_tensor, 1);
}

#define THRESHOLD -10.0

std::vector<Char> TextRecognizer::decode_word(Word& word) {
	std::vector<Char> result;
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
		char_.rect = word.rect;
		for (auto it = symbols_indexes.begin(); it != end; ++it) {
			char_.alternatives.push_back(symbols.at(*it));
		}
		if (char_.alternatives.size() == 0) {
			throw std::runtime_error("What the fuck?");
		}
		result.push_back(char_);
	}
	return result;
}

}
