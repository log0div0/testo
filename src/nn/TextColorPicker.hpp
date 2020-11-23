
#pragma once

#include "OnnxRuntime.hpp"
#include "OCR.hpp"

namespace nn {

struct TextColorPicker {
	static TextColorPicker& instance();

	TextColorPicker(const TextColorPicker&) = delete;
	TextColorPicker& operator=(const TextColorPicker&) = delete;

	void run(const stb::Image<stb::RGB>* image, Char& char_);

private:
	TextColorPicker() = default;

	void run_nn(const stb::Image<stb::RGB>* image, const Char& char_);
	void run_postprocessing(Char& char_);

	int in_c = 0;
	int out_c = 0;

	onnx::Model model = "TextColorPicker";
	onnx::Image in = "input";

	struct Output: onnx::Value {
		using onnx::Value::Value;
		void resize(int classes_count);
		float operator[](size_t index);
	};

	Output out = "output";
};

}
