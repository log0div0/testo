
#pragma once

#include "OnnxRuntime.hpp"
#include "OCR.hpp"

namespace nn {

struct TextColorPicker {
	static TextColorPicker& instance();
	~TextColorPicker();

	TextColorPicker(const TextColorPicker&) = delete;
	TextColorPicker& operator=(const TextColorPicker&) = delete;

	void run(Char& char_);

private:
	TextColorPicker();

	void run_nn(const Char& char_);
	void run_postprocessing(Char& char_);

	int in_w = 0;
	int in_c = 0;
	int out_w = 0;
	int out_c = 0;
	std::vector<float> in;
	std::vector<float> out;

	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Value> in_tensor;
	std::unique_ptr<Ort::Value> out_tensor;

	std::vector<uint8_t> char_img, char_img_resized;
};

}
