
#include "TextRecognizer.hpp"
#include <iostream>
#include <utf8.hpp>

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

	run_nn(image, rects);
	throw std::runtime_error("Implement me");
}

void TextRecognizer::run_nn(const stb::Image& image, const std::vector<Rect>& rects) {
	int32_t max_width = 0;

	for (auto& rect: rects) {
		float ratio = float(rect.width()) / float(rect.height());
		int32_t height = MAX_HEIGHT;
		int32_t width = ratio * height;
		if (max_width < width) {
			max_width = width;
		}
	}

	max_width = nearest_n_times_div_by_2(max_width, 2);

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
}

}
