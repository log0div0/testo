
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

std::vector<std::string> TextRecognizer::recognize(const stb::Image& image, const std::vector<Rect>& rects) {
	if (!image.data) {
		return {};
	}
	if (!rects.size()) {
		return {};
	}

	run_nn(image, rects);
	throw std::runtime_error("Implement me");
}

void TextRecognizer::run_nn(const stb::Image& image, const std::vector<Rect>& rects) {

	widths.resize(rects.size());

	for (size_t i = 0; i < rects.size(); ++i) {
		const Rect& rect = rects[i];
		float ratio = float(rect.width()) / float(rect.height());
		int height = MAX_HEIGHT;
		int width = ratio * height;
		widths[i] = width;
	}

	int max_width = nearest_n_times_div_by_2(*std::max_element(widths.begin(), widths.end()), 2);
	word_grey_resized.resize(MAX_HEIGHT * max_width);

	if ((size_t(batch_size) < rects.size()) || (in_w < max_width)) {
		batch_size = rects.size();
		in_w = max_width;
		out_w = max_width / 4 + 1;

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

		std::array<int64_t, 4> in_shape = {batch_size, 1, MAX_HEIGHT, in_w};
		std::array<int64_t, 3> out_shape = {out_w, batch_size, int(symbols.size() + 1)};

		in.resize(batch_size * 1 * MAX_HEIGHT * in_w);
		out.resize(out_w * batch_size * int(symbols.size() + 1));

		in_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, in.data(), in.size(), in_shape.data(), in_shape.size()));
		out_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, out.data(), out.size(), out_shape.data(), out_shape.size()));
	}

	std::fill(in.begin(), in.end(), 0);
	std::fill(out.begin(), out.end(), 0);

	for (size_t i = 0; i < rects.size(); ++i) {
		const Rect& rect = rects[i];

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

		stbir_resize_uint8(
			word_grey.data(), w, h, 0,
			word_grey_resized.data(), widths[i], MAX_HEIGHT, 0,
			1);

		std::string path = "tmp/" + std::to_string(i) + ".png";
		if (!stbi_write_png(path.c_str(), widths[i], MAX_HEIGHT, 1, word_grey_resized.data(), widths[i]*1)) {
			throw std::runtime_error("Cannot save image " + path + " because " + stbi_failure_reason());
		}
	}
}

}
