
#pragma once

#include "OnnxRuntime.hpp"
#include "TextLine.hpp"

namespace nn {

struct TextRecognizer {
	static const std::vector<std::u32string> symbols;

	static TextRecognizer& instance();

	TextRecognizer(const TextRecognizer&) = delete;
	TextRecognizer& operator=(const TextRecognizer&) = delete;

	std::vector<TextLine> recognize(const stb::Image<stb::RGB>* image, TextLine& textline, const std::string& query);

private:
	TextRecognizer();

	void run_nn(const stb::Image<stb::RGB>* image, TextLine& textline);
	std::vector<TextLine> run_postprocessing(const TextLine& textline, const std::string& query);

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
