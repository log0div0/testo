
#include "TextRecognizer.hpp"
#include <iostream>
#include <utf8.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>

extern unsigned char TextRecognizer_onnx[];
extern unsigned int TextRecognizer_onnx_len;

#define MAX_HEIGHT 32

std::string common = "0123456789!?\"'#$%&@()[]{}<>+-*/\\.,:;^~=|_";
std::string english = "abcdefghijklmnopqrstuvwxyz";
std::string ENGLISH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
std::string russian = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя";
std::string RUSSIAN = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";
std::string alphabet = common + english + ENGLISH + russian + RUSSIAN;

namespace nn {

static inline bool is_n_times_div_by_2(int value, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		if ((value % 2) != 0) {
			return false;
		}
		value /= 2;
	}
	return true;
}

static inline int nearest_n_times_div_by_2(int value, size_t n) {
	while (true) {
		if (is_n_times_div_by_2(value, n)) {
			return value;
		}
		value += 1;
	}
}

/* ITU-R Recommendation 601-2 (assuming nonlinear RGB) */
#define L24(rgb)\
	((rgb)[0]*19595 + (rgb)[1]*38470 + (rgb)[2]*7471)

static inline uint8_t rgb2l(uint8_t* in) {
	return L24(in) >> 16;
}

TextRecognizer::TextRecognizer() {
	symbols = utf8::split_to_chars(alphabet);
	session = LoadModel(TextRecognizer_onnx, TextRecognizer_onnx_len);
}

TextRecognizer::~TextRecognizer() {

}

void TextRecognizer::recognize(const stb::Image& image, std::vector<Word>& words) {
	if (!image.data) {
		return;
	}
	if (!words.size()) {
		return;
	}

	run_nn(image, words);
	decode_words(words);
}

void TextRecognizer::run_nn(const stb::Image& image, const std::vector<Word>& words) {

	widths.resize(words.size());

	for (size_t b = 0; b < words.size(); ++b) {
		const Rect& rect = words[b].rect;
		float ratio = float(rect.width()) / float(rect.height());
		int height = MAX_HEIGHT;
		int width = ratio * height;
		widths[b] = width;
	}

	int max_width = nearest_n_times_div_by_2(*std::max_element(widths.begin(), widths.end()), 2);

	if ((size_t(batch_size) < words.size()) || (in_w < max_width)) {
		batch_size = words.size();
		in_w = max_width;
		in_c = 1;
		out_w = max_width / 4 + 1;
		out_c = symbols.size() + 1;

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

		std::array<int64_t, 4> in_shape = {batch_size, in_c, MAX_HEIGHT, in_w};
		std::array<int64_t, 3> out_shape = {out_w, batch_size, out_c};

		in.resize(batch_size * in_c * MAX_HEIGHT * in_w);
		out.resize(out_w * batch_size * out_c);

		in_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, in.data(), in.size(), in_shape.data(), in_shape.size()));
		out_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, out.data(), out.size(), out_shape.data(), out_shape.size()));
	}

	std::fill(in.begin(), in.end(), 0);
	std::fill(out.begin(), out.end(), 0);

	for (size_t b = 0; b < words.size(); ++b) {
		const Rect& rect = words[b].rect;

		int h = rect.height();
		int w = rect.width();
		word_grey.resize(w * h);
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				int src_index = (rect.top + y) * image.width * image.channels + (rect.left + x) * image.channels;
				int dst_index = y * w + x;
				word_grey[dst_index] = rgb2l(&image.data[src_index]);
			}
		}

		word_grey_resized.resize(widths[b] * MAX_HEIGHT);
		if (!stbir_resize_uint8(
			word_grey.data(), w, h, 0,
			word_grey_resized.data(), widths[b], MAX_HEIGHT, 0,
			in_c)
		) {
			throw std::runtime_error("stbir_resize_uint8 failed");
		}

		// std::string path = "tmp/" + std::to_string(b) + ".png";
		// if (!stbi_write_png(path.c_str(), widths[b], MAX_HEIGHT, 1, word_grey_resized.data(), widths[b]*1)) {
		// 	throw std::runtime_error("Cannot save image " + path + " because " + stbi_failure_reason());
		// }

		for (int y = 0; y < MAX_HEIGHT; ++y) {
			for (int x = 0; x < widths[b]; ++x) {
				for (int c = 0; c < in_c; ++c) {
					int src_index = y * widths[b] * in_c + x * in_c + c;
					int dst_index = b * in_c * MAX_HEIGHT * max_width + c * MAX_HEIGHT * max_width + y * max_width + x;
					in[dst_index] = float(word_grey_resized[src_index]) / 255.0 - 0.5;
				}
			}
		}
	}

	const char* in_names[] = {"input"};
	const char* out_names[] = {"output"};

	session->Run(Ort::RunOptions{nullptr}, in_names, &*in_tensor, 1, out_names, &*out_tensor, 1);
}

void TextRecognizer::decode_words(std::vector<Word>& words) {
	for (size_t b = 0; b < words.size(); ++b) {
		std::string& text = words[b].text;
		int prev_max_pos = -1;
		for (int x = 0; (x < (widths[b] / 4)) && (x < out_w); ++x) {
			int max_pos = -1;
			float max_value = std::numeric_limits<float>::lowest();
			for (int c = 0; c < out_c; ++c) {
				int index = x * batch_size * out_c + b * out_c + c;
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
			text += symbols.at(max_pos - 1);
		}
	}
}

}
