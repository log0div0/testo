
#pragma once

#include "TextLine.hpp"
#include "OnnxRuntime.hpp"
#include <stb/Image.hpp>

namespace nn {

struct TextColorPicker {
	static TextColorPicker& instance();

	TextColorPicker(const TextColorPicker&) = delete;
	TextColorPicker& operator=(const TextColorPicker&) = delete;

	bool run(const stb::Image<stb::RGB>* image, const TextLine& textline, const std::string& fg, const std::string& bg);

private:
	TextColorPicker() = default;

	void run_nn(const stb::Image<stb::RGB>* image, const TextLine& textline);
	bool run_postprocessing(const TextLine& textline, const std::string& fg, const std::string& bg);
	bool match_color(const TextLine& textline, const std::string& color, int c_off);

	int in_w = 0;
	int in_c = 0;
	int out_w = 0;
	int out_c = 0;

	onnx::Model model = "TextColorPicker";
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
