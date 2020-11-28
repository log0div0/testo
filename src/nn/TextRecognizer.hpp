
#pragma once

#include "OnnxRuntime.hpp"
#include "TextLine.hpp"

namespace nn {

struct TextRecognizer {
	static TextRecognizer& instance();

	TextRecognizer(const TextRecognizer&) = delete;
	TextRecognizer& operator=(const TextRecognizer&) = delete;

	void recognize(const stb::Image<stb::RGB>* image, TextLine& textline);

private:
	TextRecognizer();

	void run_nn(const stb::Image<stb::RGB>* image, TextLine& textline);
	void run_postprocessing(TextLine& textline);

	std::vector<size_t> symbols_indexes;

	int in_w = 0;
	int in_c = 0;
	int out_w = 0;
	int out_c = 0;

	onnx::Model model = "TextRecognizer";
	onnx::Image in = "input";

	struct Output: onnx::Value {
		using onnx::Value::Value;

		void resize(int w, int c) {
			Value::resize({w, 1, c});
		}
		float* operator[](int x) {
			return &_buf[x * _shape[2]];
		}
	};

	Output out = "output";
};

}
